"""
Tests for app.automation.calendar.

The Africa/Cairo cases below use real 2026 transition instants from the tz
database, not invented ones:

    DST starts  2026-04-23 22:00 UTC  (local 00:00 EET -> 01:00 EEST, 24 Apr)
    DST ends    2026-10-29 21:00 UTC  (local 00:00 EEST -> 23:00 EET, 29 Oct)

which makes 2026-04-24 a 23-hour local day and 2026-10-29 a 25-hour one.
Asia/Riyadh cases are included as the control: it has no DST, so a regression
that reintroduces a fixed offset would still pass those and fail Cairo.
"""

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.automation.calendar import (
    Interval,
    day_intervals,
    local_midnight_utc,
    reporting_day,
    reporting_day_of,
    resolve_local,
    snapshot_cutoff,
    split_intervals,
)

CAIRO = ZoneInfo("Africa/Cairo")
RIYADH = ZoneInfo("Asia/Riyadh")
UTC = timezone.utc


# ── Reporting-day boundaries ─────────────────────────────────────────────────

def test_cairo_winter_day_starts_at_utc_plus_two():
    """The bug the +3 literal in NEWFIX_CM_Chat_Search.sql would produce."""
    day = reporting_day(date(2026, 1, 15), CAIRO)
    assert day.start_utc == datetime(2026, 1, 14, 22, 0, tzinfo=UTC)
    assert day.end_utc == datetime(2026, 1, 15, 22, 0, tzinfo=UTC)
    assert day.hours == 24


def test_cairo_summer_day_starts_at_utc_plus_three():
    day = reporting_day(date(2026, 7, 15), CAIRO)
    assert day.start_utc == datetime(2026, 7, 14, 21, 0, tzinfo=UTC)
    assert day.hours == 24


def test_cairo_spring_forward_day_is_23_hours():
    day = reporting_day(date(2026, 4, 24), CAIRO)
    assert day.start_utc == datetime(2026, 4, 23, 22, 0, tzinfo=UTC)
    assert day.end_utc == datetime(2026, 4, 24, 21, 0, tzinfo=UTC)
    assert day.hours == 23
    assert day.is_irregular


def test_cairo_fall_back_day_is_25_hours():
    day = reporting_day(date(2026, 10, 29), CAIRO)
    assert day.start_utc == datetime(2026, 10, 28, 21, 0, tzinfo=UTC)
    assert day.end_utc == datetime(2026, 10, 29, 22, 0, tzinfo=UTC)
    assert day.hours == 25
    assert day.is_irregular


def test_riyadh_has_no_irregular_days():
    for d in (date(2026, 4, 24), date(2026, 10, 29), date(2026, 1, 15)):
        day = reporting_day(d, RIYADH)
        assert day.hours == 24
        assert not day.is_irregular


def test_consecutive_days_tile_without_gap_or_overlap_across_transitions():
    """No conversation may fall between two reporting days, or into both."""
    for tz in (CAIRO, RIYADH):
        cursor = date(2026, 4, 20)
        previous_end = reporting_day(cursor, tz).start_utc
        while cursor < date(2026, 11, 5):
            day = reporting_day(cursor, tz)
            assert day.start_utc == previous_end, f"gap/overlap at {cursor} in {tz}"
            previous_end = day.end_utc
            cursor += timedelta(days=1)


# ── Day assignment ───────────────────────────────────────────────────────────

def test_reporting_day_of_assigns_winter_midnight_edge_correctly():
    """22:00 UTC is already the next Cairo day in winter; 21:59 is not."""
    assert reporting_day_of(datetime(2026, 1, 14, 21, 59, tzinfo=UTC), CAIRO) == date(2026, 1, 14)
    assert reporting_day_of(datetime(2026, 1, 14, 22, 0, tzinfo=UTC), CAIRO) == date(2026, 1, 15)


def test_reporting_day_of_differs_from_fixed_plus_three_in_winter():
    """
    A conversation handled at 22:30 UTC on 14 Jan is 00:30 on 15 Jan in Cairo.
    A hard-coded +3 offset would call it 01:30 on 15 Jan — same date here, but
    the hour is wrong, which misplaces it by one sampling interval.
    """
    instant = datetime(2026, 1, 14, 22, 30, tzinfo=UTC)
    local = instant.astimezone(CAIRO)
    assert (local.hour, local.minute) == (0, 30)
    naive_plus_three = instant.replace(tzinfo=None) + timedelta(hours=3)
    assert naive_plus_three.hour == 1  # the offset bug, made explicit


def test_reporting_day_of_rejects_naive_input():
    with pytest.raises(ValueError, match="tz-aware"):
        reporting_day_of(datetime(2026, 1, 14, 22, 0), CAIRO)


def test_day_contains_is_half_open():
    day = reporting_day(date(2026, 1, 15), CAIRO)
    assert day.contains(day.start_utc)
    assert not day.contains(day.end_utc)


# ── Local wall-clock resolution ──────────────────────────────────────────────

def test_nonexistent_local_time_moves_forward_out_of_the_gap():
    """00:30 on 2026-04-24 never happens in Cairo; it resolves to 01:30 EEST."""
    resolved = resolve_local(datetime(2026, 4, 24, 0, 30), CAIRO)
    assert resolved.utcoffset() == timedelta(hours=3)
    assert (resolved.hour, resolved.minute) == (1, 30)
    assert resolved.astimezone(UTC) == datetime(2026, 4, 23, 22, 30, tzinfo=UTC)


def test_ambiguous_local_time_picks_the_earlier_occurrence_by_default():
    """23:30 on 2026-10-29 happens twice in Cairo."""
    earliest = resolve_local(datetime(2026, 10, 29, 23, 30), CAIRO)
    latest = resolve_local(datetime(2026, 10, 29, 23, 30), CAIRO, prefer_earliest=False)
    assert earliest.utcoffset() == timedelta(hours=3)   # still on DST
    assert latest.utcoffset() == timedelta(hours=2)     # after the fall back
    assert latest.astimezone(UTC) - earliest.astimezone(UTC) == timedelta(hours=1)


def test_resolve_local_rejects_aware_input():
    with pytest.raises(ValueError, match="naive"):
        resolve_local(datetime(2026, 1, 1, 9, 0, tzinfo=CAIRO), CAIRO)


def test_local_midnight_utc_matches_reporting_day_start():
    for d in (date(2026, 4, 24), date(2026, 10, 29), date(2026, 1, 15)):
        assert local_midnight_utc(d, CAIRO) == reporting_day(d, CAIRO).start_utc


# ── Interval splitting ───────────────────────────────────────────────────────

def test_regular_day_splits_into_twelve_two_hour_intervals():
    intervals = day_intervals(reporting_day(date(2026, 1, 15), CAIRO), 120)
    assert len(intervals) == 12
    assert all(i.minutes == 120 for i in intervals)


def test_25_hour_day_yields_a_short_final_interval():
    day = reporting_day(date(2026, 10, 29), CAIRO)
    intervals = day_intervals(day, 120)
    assert len(intervals) == 13
    assert [i.minutes for i in intervals] == [120] * 12 + [60]
    assert intervals[-1].end_utc == day.end_utc


def test_23_hour_day_yields_a_short_final_interval():
    day = reporting_day(date(2026, 4, 24), CAIRO)
    intervals = day_intervals(day, 120)
    assert [i.minutes for i in intervals] == [120] * 11 + [60]
    assert intervals[-1].end_utc == day.end_utc


def test_intervals_tile_the_day_exactly():
    for d in (date(2026, 4, 24), date(2026, 10, 29), date(2026, 7, 15)):
        day = reporting_day(d, CAIRO)
        intervals = day_intervals(day, 120)
        assert intervals[0].start_utc == day.start_utc
        assert intervals[-1].end_utc == day.end_utc
        for earlier, later in zip(intervals, intervals[1:]):
            assert earlier.end_utc == later.start_utc
        assert sum(i.duration.total_seconds() for i in intervals) == day.duration.total_seconds()


def test_every_instant_in_the_day_falls_in_exactly_one_interval():
    day = reporting_day(date(2026, 10, 29), CAIRO)
    intervals = day_intervals(day, 120)
    cursor = day.start_utc
    while cursor < day.end_utc:
        assert sum(1 for i in intervals if i.contains(cursor)) == 1
        cursor += timedelta(minutes=17)  # deliberately not an interval divisor


def test_intervals_advance_by_real_elapsed_time_not_wall_clock():
    """
    One interval on 2026-10-29 straddles the offset change: it runs from local
    22:00 EEST to 23:00 EET, which is two real hours but only one hour of wall
    clock. Quota allocated proportionally to duration must see 120 minutes.

    Stepping the *local* clock by two hours instead would make this interval
    claim two wall-clock hours and silently overlap its neighbour.
    """
    intervals = day_intervals(reporting_day(date(2026, 10, 29), CAIRO), 120)
    straddling = [
        i for i in intervals
        if i.start_utc.astimezone(CAIRO).utcoffset() != i.end_utc.astimezone(CAIRO).utcoffset()
    ]
    assert len(straddling) == 1
    interval = straddling[0]
    assert interval.minutes == 120

    lo, hi = interval.local_bounds(CAIRO)
    assert (lo.hour, hi.hour) == (22, 23)         # one wall-clock hour apart
    assert lo.utcoffset() == timedelta(hours=3)   # EEST
    assert hi.utcoffset() == timedelta(hours=2)   # EET

    # Subtracting two aware datetimes in the *same* zone is wall-clock
    # arithmetic in Python, so it reports one hour. Only going through UTC
    # gives the two hours that actually elapsed. This is why every duration
    # in this module is computed on UTC instants.
    assert hi - lo == timedelta(hours=1)
    assert hi.astimezone(UTC) - lo.astimezone(UTC) == timedelta(hours=2)


def test_interval_id_is_stable_and_bounds_derived():
    i = Interval(
        index=3,
        start_utc=datetime(2026, 10, 29, 6, 0, tzinfo=UTC),
        end_utc=datetime(2026, 10, 29, 8, 0, tzinfo=UTC),
    )
    assert i.interval_id == "20261029T0600Z/20261029T0800Z"


def test_interval_ids_are_unique_within_a_day():
    for d in (date(2026, 4, 24), date(2026, 10, 29), date(2026, 7, 15)):
        ids = [i.interval_id for i in day_intervals(reporting_day(d, CAIRO), 120)]
        assert len(ids) == len(set(ids))


def test_split_intervals_rejects_nonpositive_length():
    day = reporting_day(date(2026, 1, 15), CAIRO)
    with pytest.raises(ValueError):
        split_intervals(day.start_utc, day.end_utc, 0)


def test_split_intervals_of_empty_span_is_empty():
    t = datetime(2026, 1, 15, 8, 0, tzinfo=UTC)
    assert split_intervals(t, t, 120) == []


# ── Snapshot cutoff ──────────────────────────────────────────────────────────

def test_snapshot_cutoff_adds_grace_to_interval_end():
    interval = day_intervals(reporting_day(date(2026, 1, 15), CAIRO), 120)[0]
    assert snapshot_cutoff(interval, 20) == interval.end_utc + timedelta(minutes=20)
    assert snapshot_cutoff(interval, 0) == interval.end_utc


def test_snapshot_cutoff_rejects_negative_grace():
    interval = day_intervals(reporting_day(date(2026, 1, 15), CAIRO), 120)[0]
    with pytest.raises(ValueError):
        snapshot_cutoff(interval, -1)
