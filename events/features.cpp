#include "events/features.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace quaml::events {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Same tolerance as quaml.reference.signing.MID_TOLERANCE.
constexpr double kMidTolerance = 1e-10;

// Rows of each group in original (time) order, as a CSR layout:
// group g owns rows[offsets[g] .. offsets[g + 1]).
struct GroupIndex {
    std::vector<size_t> offsets;
    std::vector<size_t> rows;
};

GroupIndex groupRows(std::span<const uint32_t> group) {
    std::unordered_map<uint32_t, uint32_t> dense;
    std::vector<uint32_t> id(group.size());
    for (size_t i = 0; i < group.size(); i++)
        id[i] = dense.try_emplace(group[i], static_cast<uint32_t>(dense.size())).first->second;

    GroupIndex gi;
    gi.offsets.assign(dense.size() + 1, 0);
    for (uint32_t g : id) gi.offsets[g + 1]++;
    for (size_t g = 0; g < dense.size(); g++) gi.offsets[g + 1] += gi.offsets[g];

    gi.rows.resize(group.size());
    std::vector<size_t> next(gi.offsets.begin(), gi.offsets.end() - 1);
    for (size_t i = 0; i < group.size(); i++) gi.rows[next[id[i]]++] = i;
    return gi;
}

void checkInputs(std::span<const int64_t> ts, size_t n, int64_t window_ns) {
    if (window_ns <= 0) throw std::invalid_argument("window_ns must be positive");
    if (ts.size() != n) throw std::invalid_argument("all columns must have equal length");
    if (!std::is_sorted(ts.begin(), ts.end()))
        throw std::invalid_argument("ts must be non-decreasing");
}

bool validQuote(double bid, double ask) {
    return std::isfinite(bid) && std::isfinite(ask) && bid <= ask;
}

} // namespace

void rollingSumCount(std::span<const int64_t> ts,
                     std::span<const double> values,
                     std::span<const uint32_t> group,
                     int64_t window_ns,
                     std::span<double> out_sum,
                     std::span<int64_t> out_count) {
    size_t n = ts.size();
    checkInputs(ts, values.size(), window_ns);
    if (group.size() != n || out_sum.size() != n || out_count.size() != n)
        throw std::invalid_argument("all columns must have equal length");

    GroupIndex gi = groupRows(group);
    for (size_t g = 0; g + 1 < gi.offsets.size(); g++) {
        const size_t* rows = gi.rows.data() + gi.offsets[g];
        size_t m = gi.offsets[g + 1] - gi.offsets[g];
        double  sum = 0.0;
        int64_t cnt = 0;
        size_t  left = 0;
        for (size_t k = 0; k < m; k++) {
            size_t i = rows[k];
            if (!std::isnan(values[i])) { sum += values[i]; cnt++; }
            while (ts[rows[left]] <= ts[i] - window_ns) {
                double v = values[rows[left++]];
                if (!std::isnan(v)) { sum -= v; cnt--; }
            }
            out_sum[i]   = sum;
            out_count[i] = cnt;
        }
    }
}

TradeQuoteFeatures tradeQuoteFeatures(const TradeQuoteView& ev, int64_t window_ns) {
    size_t n = ev.ts_recv.size();
    checkInputs(ev.ts_recv, ev.instrument_id.size(), window_ns);
    for (size_t len : {ev.price.size(), ev.size.size(), ev.side.size(), ev.bid_px.size(),
                       ev.ask_px.size(), ev.bid_sz.size(), ev.ask_sz.size()})
        if (len != n) throw std::invalid_argument("all columns must have equal length");

    TradeQuoteFeatures f;
    for (auto* col : {&f.mid, &f.spread, &f.relative_spread, &f.microprice, &f.book_imbalance,
                      &f.sign, &f.signed_volume_w, &f.volume_w, &f.trade_count_w,
                      &f.flow_imbalance_w, &f.time_since_prev})
        col->resize(n);

    // Per-row quote features and trade sign.
    for (size_t i = 0; i < n; i++) {
        double bid = ev.bid_px[i], ask = ev.ask_px[i];
        double bsz = static_cast<double>(ev.bid_sz[i]);
        double asz = static_cast<double>(ev.ask_sz[i]);
        double total = bsz + asz;
        bool   ok = validQuote(bid, ask);

        double mid = ok ? 0.5 * (bid + ask) : kNaN;
        double spr = ok ? ask - bid : kNaN;
        f.mid[i]             = mid;
        f.spread[i]          = spr;
        f.relative_spread[i] = mid > 0 ? spr / mid : kNaN;
        f.microprice[i]      = ok && total > 0 ? (ask * bsz + bid * asz) / total : kNaN;
        f.book_imbalance[i]  = total > 0 ? (bsz - asz) / total : kNaN;

        int8_t s = ev.side[i];
        if (s == 0) {
            double d = ev.price[i] - mid;
            s = d > kMidTolerance ? 1 : (d < -kMidTolerance ? -1 : 0);
        }
        f.sign[i] = s;
    }

    // Fused rolling pass: volume, signed volume and trade count share one window.
    GroupIndex gi = groupRows(ev.instrument_id);
    for (size_t g = 0; g + 1 < gi.offsets.size(); g++) {
        const size_t* rows = gi.rows.data() + gi.offsets[g];
        size_t m = gi.offsets[g + 1] - gi.offsets[g];
        double vol = 0.0, signedVol = 0.0;
        size_t left = 0;
        for (size_t k = 0; k < m; k++) {
            size_t i = rows[k];
            double sz = static_cast<double>(ev.size[i]);
            vol       += sz;
            signedVol += f.sign[i] * sz;
            while (ev.ts_recv[rows[left]] <= ev.ts_recv[i] - window_ns) {
                size_t j = rows[left++];
                double szj = static_cast<double>(ev.size[j]);
                vol       -= szj;
                signedVol -= f.sign[j] * szj;
            }
            f.volume_w[i]         = vol;
            f.signed_volume_w[i]  = signedVol;
            f.trade_count_w[i]    = static_cast<double>(k + 1 - left);
            f.flow_imbalance_w[i] = vol > 0 ? signedVol / vol : kNaN;
            f.time_since_prev[i]  = k == 0 ? kNaN
                : static_cast<double>(ev.ts_recv[i] - ev.ts_recv[rows[k - 1]]);
        }
    }
    return f;
}

} // namespace quaml::events
