"""Server mode (`prism --server`): one long-running process serving many
independent browser sessions.

`build_app()` already builds a fresh `AppState` and widget tree per session,
so sessions don't share UI state. What a long-running server additionally
needs -- and what lives here -- is lifecycle management: a session that
nobody has touched for a while must give its memory back, even though its
browser tab (and so its websocket) may still be open, which Bokeh's own
unused-session cleanup never considers idle.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Callable

from prism import gridded_products as gp

logger = logging.getLogger(__name__)

DEFAULT_IDLE_TIMEOUT_S = 30 * 60

_live_sessions = 0
_live_lock = threading.Lock()


def live_session_count() -> int:
    return _live_sessions


def _session_opened() -> None:
    global _live_sessions
    with _live_lock:
        _live_sessions += 1


def _session_closed() -> None:
    """Free what the closed session left behind. The process-wide dataset
    cache is only dropped once NO session is live: it is shared, so an
    active user's datasets must not be pulled out from under them."""
    global _live_sessions
    with _live_lock:
        _live_sessions = max(0, _live_sessions - 1)
        remaining = _live_sessions
    if remaining == 0:
        gp.clear_dataset_cache()
    # The panel/HoloViews object graph is full of reference cycles
    # (widget <-> watcher <-> stream <-> DynamicMap), so refcounting alone
    # won't free a closed session.
    gc.collect()
    logger.info("Session closed; %d live session(s) remain", remaining)


class SessionLifecycle:
    """Idle tracking and one-shot teardown for one user session.

    touch() marks the session as used; check() (called periodically) closes
    it once it has gone `idle_timeout_s` without one. close() is also what
    the server's own session-destroyed hook calls when the browser goes
    away, and is idempotent, so whichever comes first wins.

    `release(expired)` does the session-specific teardown (drop loaded data,
    and -- only when `expired` is True, i.e. the browser is still there --
    replace the UI with a notice). idle_timeout_s of None/0 disables expiry.
    """

    def __init__(self, idle_timeout_s: float | None, release: Callable[[bool], None],
                 clock: Callable[[], float] = time.monotonic):
        self.idle_timeout_s = idle_timeout_s
        self._release = release
        self._clock = clock
        self.last_active = clock()
        self.closed = False
        self.expired = False
        _session_opened()

    def touch(self) -> None:
        if not self.closed:
            self.last_active = self._clock()

    def idle_for(self) -> float:
        return self._clock() - self.last_active

    def check(self) -> bool:
        """True once the session is closed (whether just now or before)."""
        if self.closed:
            return True
        if self.idle_timeout_s and self.idle_for() >= self.idle_timeout_s:
            logger.info("Session idle for %.0f s (limit %.0f s); releasing it",
                        self.idle_for(), self.idle_timeout_s)
            self.close(expired=True)
        return self.closed

    def close(self, expired: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        self.expired = expired
        try:
            self._release(expired)
        except Exception:
            logger.exception("Session release failed")
        finally:
            _session_closed()
