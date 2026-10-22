#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace quaml::events {

// One trade per row with the best bid/offer just before it. Column semantics
// match python/quaml/events.py: prices are float (NaN if undefined), side is
// +1 buy / -1 sell / 0 unknown, rows are sorted by ts_recv.
struct TradeQuotes {
    std::vector<uint32_t> instrument_id;
    std::vector<int64_t>  ts_event;
    std::vector<int64_t>  ts_recv;
    std::vector<double>   price;
    std::vector<uint32_t> size;
    std::vector<int8_t>   side;
    std::vector<double>   bid_px;
    std::vector<double>   ask_px;
    std::vector<uint32_t> bid_sz;
    std::vector<uint32_t> ask_sz;

    size_t rows() const { return ts_recv.size(); }
};

// Reads a Databento TBBO (MBP-1 records) or TCBBO (CMBP-1 records) file,
// zstd-compressed (.dbn.zst) or plain (.dbn). Other record types are skipped.
TradeQuotes readTradeQuotes(const std::string& path);

} // namespace quaml::events
