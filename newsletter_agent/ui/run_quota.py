"""A process-wide daily cap on demo runs, shared by every browser session."""

import threading
from collections.abc import Callable
from datetime import date


class DailyRunQuota:
    """Counts runs per calendar day across all sessions; safe to share between threads."""

    def __init__(
        self, max_runs_per_day: int, today_provider: Callable[[], date] = date.today
    ) -> None:
        self._max_runs_per_day = max_runs_per_day
        self._today_provider = today_provider
        self._lock = threading.Lock()
        self._counted_date = today_provider()
        self._runs_used = 0

    def try_consume(self) -> bool:
        """Use one of today's runs; return False when none are left."""
        with self._lock:
            self._reset_if_new_day()
            if self._runs_used >= self._max_runs_per_day:
                return False
            self._runs_used += 1
            return True

    def remaining(self) -> int:
        """Return how many runs are left today."""
        with self._lock:
            self._reset_if_new_day()
            return max(self._max_runs_per_day - self._runs_used, 0)

    def _reset_if_new_day(self) -> None:
        """Start a fresh count when the date has changed. Call with the lock held."""
        today = self._today_provider()
        if today != self._counted_date:
            self._counted_date = today
            self._runs_used = 0
