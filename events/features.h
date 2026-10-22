#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace quaml::events {

// Window semantics match python/quaml/reference/rolling.py: for row i the
// window holds rows j <= i of the same group with ts[j] > ts[i] - window_ns.
// NaN values are missing (not summed, not counted).
void rollingSumCount(std::span<const int64_t> ts,
                     std::span<const double> values,
                     std::span<const uint32_t> group,
                     int64_t window_ns,
                     std::span<double> out_sum,
                     std::span<int64_t> out_count);

struct TradeQuoteView {
    std::span<const uint32_t> instrument_id;
    std::span<const int64_t>  ts_recv;
    std::span<const double>   price;
    std::span<const uint32_t> size;
    std::span<const int8_t>   side;
    std::span<const double>   bid_px;
    std::span<const double>   ask_px;
    std::span<const uint32_t> bid_sz;
    std::span<const uint32_t> ask_sz;
};

// Same columns as quaml.reference.features.trade_quote_features, computed in
// one pass per instrument.
struct TradeQuoteFeatures {
    std::vector<double> mid;
    std::vector<double> spread;
    std::vector<double> relative_spread;
    std::vector<double> microprice;
    std::vector<double> book_imbalance;
    std::vector<double> sign;
    std::vector<double> signed_volume_w;
    std::vector<double> volume_w;
    std::vector<double> trade_count_w;
    std::vector<double> flow_imbalance_w;
    std::vector<double> time_since_prev;
};

TradeQuoteFeatures tradeQuoteFeatures(const TradeQuoteView& ev, int64_t window_ns);

} // namespace quaml::events
