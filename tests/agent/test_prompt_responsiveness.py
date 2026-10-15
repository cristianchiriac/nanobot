"""Prompt filesystem work must leave the runtime loop available."""

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.hook import AgentHook
from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.context import current_request_context
from nanobot.providers.base import LLMResponse
from nanobot.runtime_context import public_history_message


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_prompt_preparation_preserves_context_and_allows_cancellation(
    tmp_path, monkeypatch, loop_factory, cancel,
) -> None:
    loop = loop_factory()
    loop.provider.provider_name = "prompt-test"
    loop.provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(content="Done"))
    runtime_loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    completed = asyncio.Event()
    release = threading.Event()
    original = loop.context.build_system_prompt
    captured = []
    finalized = []

    class Lifecycle(AgentHook):
        async def on_finally(self, context):
            finalized.append((context.stop_reason, current_request_context(), asyncio.get_running_loop()))

    loop._extra_hooks.append(Lifecycle())

    def build(**kwargs):
        captured.append(current_request_context())
        runtime_loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(5), "test did not release prompt filesystem work"
            return original(**kwargs)
        finally:
            runtime_loop.call_soon_threadsafe(completed.set)

    monkeypatch.setattr(loop.context, "build_system_prompt", build)
    task = asyncio.create_task(loop.process_direct("Hello", session_key="cli:prompt-work"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert captured[0] is not None
        assert captured[0].session_key == "cli:prompt-work"
        assert captured[0].workspace_scope.project_path == tmp_path.resolve()
        assert current_request_context() is None
        loop.provider.chat_stream_with_retry.assert_not_awaited()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
            loop.provider.chat_stream_with_retry.assert_not_awaited()
            assert len(finalized) == 1
            reason, request, final_loop = finalized[0]
            assert reason == "cancelled"
            assert request is not None and request.session_key == "cli:prompt-work"
            assert final_loop is runtime_loop
        release.set()
        await asyncio.wait_for(completed.wait(), timeout=5)
        if not cancel:
            result = await asyncio.wait_for(task, timeout=5)
            assert result is not None and result.content == "Done"
            saved = loop.sessions.read_session_snapshot("cli:prompt-work")
            assert [public_history_message(message)["content"] for message in saved.messages] == ["Hello", "Done"]
        assert current_request_context() is None
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if captured:
            await asyncio.wait_for(completed.wait(), timeout=5)
        await loop.aclose()


@pytest.mark.asyncio
async def test_cancelled_prompt_keeps_its_worker_slot_until_read_finishes() -> None:
    runner = AgentRunner()
    runtime_loop = asyncio.get_running_loop()
    entered = {name: asyncio.Event() for name in ("first", "second")}
    released = {name: threading.Event() for name in entered}

    def read(name):
        runtime_loop.call_soon_threadsafe(entered[name].set)
        assert released[name].wait(5), "test did not release prompt read"
        return name

    first = asyncio.create_task(runner.prepare_prompt(read, "first"))
    tasks = [first]
    try:
        await asyncio.wait_for(entered["first"].wait(), timeout=5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, timeout=5)
        second = asyncio.create_task(runner.prepare_prompt(read, "second"))
        tasks.append(second)
        await asyncio.sleep(0)
        # The cancelled caller's read keeps its slot until the worker exits.
        assert not entered["second"].is_set()
        released["first"].set()
        await asyncio.wait_for(entered["second"].wait(), timeout=5)
        released["second"].set()
        assert await asyncio.wait_for(second, timeout=5) == "second"
    finally:
        for release in released.values():
            release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*tuple(runner.context_governor._preparation_jobs), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["request", "usage"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_local_token_estimation_leaves_loop_available(
    tmp_path, monkeypatch, loop_factory, phase, cancel,
) -> None:
    loop = loop_factory()
    loop.provider.provider_name = "token-test"
    loop.provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(content="Done"))
    runtime_loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    completed = asyncio.Event()
    release = threading.Event()
    captured = []
    finalized = []

    class Lifecycle(AgentHook):
        async def on_finally(self, context):
            finalized.append((context.stop_reason, current_request_context(), asyncio.get_running_loop()))

    loop._extra_hooks.append(Lifecycle())

    def count(*_args, **_kwargs):
        captured.append(current_request_context())
        runtime_loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(5), "test did not release local token estimation"
            return 10, "test-counter"
        finally:
            runtime_loop.call_soon_threadsafe(completed.set)

    module = "context_governance" if phase == "request" else "runner"
    monkeypatch.setattr(f"nanobot.agent.{module}.estimate_prompt_tokens_chain", count)
    task = asyncio.create_task(loop.process_direct("Hello", session_key="cli:token-work"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert captured[0] is not None
        assert captured[0].session_key == "cli:token-work"
        assert captured[0].workspace_scope.project_path == tmp_path.resolve()
        assert current_request_context() is None
        if phase == "request":
            loop.provider.chat_stream_with_retry.assert_not_awaited()
        else:
            loop.provider.chat_stream_with_retry.assert_awaited_once()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout=1)
            assert len(finalized) == 1
            reason, request, final_loop = finalized[0]
            assert reason == "cancelled"
            assert request is not None and request.session_key == "cli:token-work"
            assert final_loop is runtime_loop
        else:
            release.set()
            result = await asyncio.wait_for(task, timeout=5)
            assert result is not None and result.content == "Done"
            saved = loop.sessions.read_session_snapshot("cli:token-work")
            assert [public_history_message(message)["content"] for message in saved.messages] == ["Hello", "Done"]
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if captured:
            await asyncio.wait_for(completed.wait(), timeout=5)
        await loop.aclose()
