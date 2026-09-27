"""Hook dispatch tests: sync/async methods, missing methods, ordering."""

from __future__ import annotations

import asyncio

from aimee.hooks import call_hook, fire


class SyncHook:
    def __init__(self, log):
        self.log = log

    def on_ping(self, value):
        self.log.append(("sync", value))
        return "sync-result"


class AsyncHook:
    def __init__(self, log):
        self.log = log

    async def on_ping(self, value):
        self.log.append(("async", value))
        return "async-result"


class Silent:
    """Defines no hooks at all."""


def test_call_hook_sync():
    log: list = []
    result = asyncio.run(call_hook(SyncHook(log), "on_ping", 42))
    assert log == [("sync", 42)]
    assert result == "sync-result"


def test_call_hook_async():
    log: list = []
    result = asyncio.run(call_hook(AsyncHook(log), "on_ping", 7))
    assert log == [("async", 7)]
    assert result == "async-result"


def test_call_hook_missing_method_is_none():
    assert asyncio.run(call_hook(Silent(), "on_ping", 1)) is None
    assert asyncio.run(call_hook(SyncHook([]), "on_other", 1)) is None


def test_fire_visits_all_hooks_in_order():
    log: list = []
    hooks = [SyncHook(log), Silent(), AsyncHook(log)]
    asyncio.run(fire(hooks, "on_ping", "x"))
    assert log == [("sync", "x"), ("async", "x")]


def test_fire_with_no_hooks():
    asyncio.run(fire([], "on_ping", "x"))  # no error
