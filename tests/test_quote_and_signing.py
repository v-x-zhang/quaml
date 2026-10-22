import numpy as np

from quaml.reference import quote, signing

nan = np.nan
BID = np.array([100.00, 100.00, nan, 101.0, 100.0])
ASK = np.array([100.02, 100.02, 100.0, 100.0, 100.0])  # row 3 crossed, row 4 locked


def test_mid_spread_relative_spread():
    np.testing.assert_allclose(quote.mid(BID, ASK), [100.01, 100.01, nan, nan, 100.0])
    np.testing.assert_allclose(quote.spread(BID, ASK), [0.02, 0.02, nan, nan, 0.0], atol=1e-12)
    np.testing.assert_allclose(quote.relative_spread(BID, ASK)[0], 0.02 / 100.01)


def test_microprice_leans_toward_thin_side():
    # Big bid, small ask -> next move likely up -> microprice near the ask.
    mp = quote.microprice(np.array([100.0]), np.array([100.02]),
                          np.array([900]), np.array([100]))
    np.testing.assert_allclose(mp, [100.0 * 0.1 + 100.02 * 0.9])
    empty = quote.microprice(np.array([100.0]), np.array([100.02]),
                             np.array([0]), np.array([0]))
    assert np.isnan(empty[0])


def test_book_imbalance():
    np.testing.assert_allclose(
        quote.book_imbalance(np.array([3, 0, 0]), np.array([1, 5, 0])), [0.5, -1.0, nan])


def test_sign_trades_prefers_feed_side():
    side = np.array([1, 0, 0, 0, -1], dtype=np.int8)
    price = np.array([100.00, 100.02, 100.00, 100.01, 100.02])
    bid = np.full(5, 100.00)
    ask = np.full(5, 100.02)
    np.testing.assert_array_equal(signing.sign_trades(side, price, bid, ask), [1, 1, -1, 0, -1])


def test_quote_rule_without_quote_is_zero():
    assert signing.quote_rule(np.array([5.0]), np.array([nan]), np.array([5.1]))[0] == 0
