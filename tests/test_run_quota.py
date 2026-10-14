from datetime import date

from newsletter_agent.ui.run_quota import DailyRunQuota


class FakeCalendar:
    """A controllable replacement for date.today."""

    def __init__(self, today: date) -> None:
        self.today = today

    def __call__(self) -> date:
        return self.today


def test_quota_grants_runs_until_the_daily_limit() -> None:
    quota = DailyRunQuota(2, today_provider=FakeCalendar(date(2026, 10, 14)))

    grants = [quota.try_consume() for _attempt in range(3)]

    assert grants == [True, True, False]
    assert quota.remaining() == 0


def test_quota_reports_remaining_runs() -> None:
    quota = DailyRunQuota(3, today_provider=FakeCalendar(date(2026, 10, 14)))

    quota.try_consume()

    assert quota.remaining() == 2


def test_quota_resets_on_a_new_day() -> None:
    fake_calendar = FakeCalendar(date(2026, 10, 14))
    quota = DailyRunQuota(1, today_provider=fake_calendar)
    quota.try_consume()

    fake_calendar.today = date(2026, 10, 15)

    assert quota.remaining() == 1
    assert quota.try_consume() is True
    assert quota.try_consume() is False
