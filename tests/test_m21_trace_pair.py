"""Small CPU-only checks for M21 interval clipping and union."""

from benchmarks.analyze_m21_trace_pair import merge_intervals, union_duration


def test_union_clips_and_merges_overlapping_intervals():
    intervals = [(0, 8), (5, 15), (18, 30)]
    assert merge_intervals(intervals, 4, 20) == [(4, 15), (18, 20)]
    assert union_duration(intervals, 4, 20) == 13


def test_union_ignores_intervals_outside_window():
    intervals = [(-10, -1), (21, 30), (7, 7)]
    assert merge_intervals(intervals, 0, 20) == []
    assert union_duration(intervals, 0, 20) == 0


def test_union_coalesces_touching_intervals_without_double_counting():
    intervals = [(10, 14), (14, 18), (12, 16)]
    assert merge_intervals(intervals) == [(10, 18)]
    assert union_duration(intervals) == 8
