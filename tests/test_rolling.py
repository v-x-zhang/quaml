import numpy as np
import pytest

from quaml.reference import rolling


def brute_sum_count(ts, values, group, w):
    n = len(ts)
    s = np.zeros(n)
    c = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(i + 1):
            if group[j] == group[i] and ts[j] > ts[i] - w and not np.isnan(values[j]):
                s[i] += values[j]
                c[i] += 1
    return s, c


def test_window_is_half_open():
    ts = np.array([0, 5, 10, 15])
    v = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.zeros(4, dtype=np.uint32)
    # window 10: at t=10, t=0 is excluded (0 > 10 - 10 is false).
    s, c = rolling.rolling_sum_count(ts, v, g, 10)
    np.testing.assert_array_equal(s, [1, 3, 5, 7])
    np.testing.assert_array_equal(c, [1, 2, 2, 2])


def test_ties_never_see_later_rows():
    ts = np.array([7, 7, 7])
    v = np.array([1.0, 10.0, 100.0])
    g = np.zeros(3, dtype=np.uint32)
    np.testing.assert_array_equal(rolling.rolling_sum(ts, v, g, 1), [1, 11, 111])


def test_groups_are_independent():
    ts = np.array([0, 1, 2, 3])
    v = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.array([1, 2, 1, 2])
    np.testing.assert_array_equal(rolling.rolling_sum(ts, v, g, 100), [1, 2, 4, 6])


def test_nan_is_missing():
    ts = np.array([0, 1, 2])
    v = np.array([1.0, np.nan, 3.0])
    g = np.zeros(3)
    s, c = rolling.rolling_sum_count(ts, v, g, 100)
    np.testing.assert_array_equal(s, [1, 1, 4])
    np.testing.assert_array_equal(c, [1, 1, 2])
    m = rolling.rolling_mean(np.array([0]), np.array([np.nan]), np.zeros(1), 5)
    assert np.isnan(m[0])


def test_rejects_unsorted_and_bad_window():
    with pytest.raises(ValueError):
        rolling.rolling_sum([2, 1], [1.0, 1.0], [0, 0], 5)
    with pytest.raises(ValueError):
        rolling.rolling_sum([1, 2], [1.0, 1.0], [0, 0], 0)


@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 300
    ts = np.sort(rng.integers(0, 1_000, n))  # dense, many ties
    v = rng.normal(size=n)
    v[rng.random(n) < 0.1] = np.nan
    g = rng.integers(0, 4, n)
    w = int(rng.integers(1, 200))
    s, c = rolling.rolling_sum_count(ts, v, g, w)
    bs, bc = brute_sum_count(ts, v, g, w)
    np.testing.assert_allclose(s, bs, atol=1e-12)
    np.testing.assert_array_equal(c, bc)


def test_time_since_prev():
    out = rolling.time_since_prev(np.array([0, 3, 4, 10]), np.array([1, 2, 1, 1]))
    np.testing.assert_array_equal(out, [np.nan, np.nan, 4, 6])
