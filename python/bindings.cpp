#include "events/dbn.h"
#include "events/features.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <span>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace quaml::events;

namespace {

template <class T>
using In = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using Out = nb::ndarray<nb::numpy, T, nb::ndim<1>>;

template <class T>
std::span<const T> view(const In<T>& a) {
    return {a.data(), a.shape(0)};
}

// Hands a vector to NumPy without copying; the array owns the buffer.
template <class T>
Out<T> toNumpy(std::vector<T>&& v) {
    auto* owned = new std::vector<T>(std::move(v));
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return Out<T>(owned->data(), {owned->size()}, owner);
}

nb::dict readTradeQuotesPy(const std::string& path) {
    TradeQuotes tq;
    {
        nb::gil_scoped_release release;
        tq = readTradeQuotes(path);
    }
    nb::dict d;
    d["instrument_id"] = toNumpy(std::move(tq.instrument_id));
    d["ts_event"]      = toNumpy(std::move(tq.ts_event));
    d["ts_recv"]       = toNumpy(std::move(tq.ts_recv));
    d["price"]         = toNumpy(std::move(tq.price));
    d["size"]          = toNumpy(std::move(tq.size));
    d["side"]          = toNumpy(std::move(tq.side));
    d["bid_px"]        = toNumpy(std::move(tq.bid_px));
    d["ask_px"]        = toNumpy(std::move(tq.ask_px));
    d["bid_sz"]        = toNumpy(std::move(tq.bid_sz));
    d["ask_sz"]        = toNumpy(std::move(tq.ask_sz));
    return d;
}

nb::tuple rollingSumCountPy(In<int64_t> ts, In<double> values, In<uint32_t> group,
                            int64_t window_ns) {
    std::vector<double>  sum(ts.shape(0));
    std::vector<int64_t> cnt(ts.shape(0));
    {
        nb::gil_scoped_release release;
        rollingSumCount(view(ts), view(values), view(group), window_ns, sum, cnt);
    }
    return nb::make_tuple(toNumpy(std::move(sum)), toNumpy(std::move(cnt)));
}

nb::dict tradeQuoteFeaturesPy(In<uint32_t> instrument_id, In<int64_t> ts_recv, In<double> price,
                              In<uint32_t> size, In<int8_t> side, In<double> bid_px,
                              In<double> ask_px, In<uint32_t> bid_sz, In<uint32_t> ask_sz,
                              int64_t window_ns) {
    TradeQuoteView ev{view(instrument_id), view(ts_recv), view(price), view(size), view(side),
                      view(bid_px), view(ask_px), view(bid_sz), view(ask_sz)};
    TradeQuoteFeatures f;
    {
        nb::gil_scoped_release release;
        f = tradeQuoteFeatures(ev, window_ns);
    }
    nb::dict d;
    d["mid"]              = toNumpy(std::move(f.mid));
    d["spread"]           = toNumpy(std::move(f.spread));
    d["relative_spread"]  = toNumpy(std::move(f.relative_spread));
    d["microprice"]       = toNumpy(std::move(f.microprice));
    d["book_imbalance"]   = toNumpy(std::move(f.book_imbalance));
    d["sign"]             = toNumpy(std::move(f.sign));
    d["signed_volume_w"]  = toNumpy(std::move(f.signed_volume_w));
    d["volume_w"]         = toNumpy(std::move(f.volume_w));
    d["trade_count_w"]    = toNumpy(std::move(f.trade_count_w));
    d["flow_imbalance_w"] = toNumpy(std::move(f.flow_imbalance_w));
    d["time_since_prev"]  = toNumpy(std::move(f.time_since_prev));
    return d;
}

} // namespace

NB_MODULE(_core, m) {
    m.doc() = "quaml native event engine";
    m.def("read_trade_quotes", &readTradeQuotesPy, "path"_a);
    m.def("rolling_sum_count", &rollingSumCountPy, "ts"_a, "values"_a, "group"_a, "window_ns"_a);
    m.def("trade_quote_features", &tradeQuoteFeaturesPy, "instrument_id"_a, "ts_recv"_a,
          "price"_a, "size"_a, "side"_a, "bid_px"_a, "ask_px"_a, "bid_sz"_a, "ask_sz"_a,
          "window_ns"_a);
}
