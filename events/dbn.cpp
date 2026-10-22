#include "events/dbn.h"

#include <zstd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace quaml::events {

namespace {

constexpr int64_t kUndefPrice  = std::numeric_limits<int64_t>::max();
constexpr double  kPriceScale  = 1e-9;
constexpr uint8_t kRtypeMbp1   = 0x01; // TBBO
constexpr uint8_t kRtypeCmbp1  = 0xC2; // TCBBO
constexpr size_t  kRecordBytes = 80;

// Field offsets shared by MBP-1 and CMBP-1 records.
constexpr size_t kOffInstrument = 4;
constexpr size_t kOffTsEvent    = 8;
constexpr size_t kOffPrice      = 16;
constexpr size_t kOffSize       = 24;
constexpr size_t kOffAction     = 28; // MBP-1 only
constexpr size_t kOffSide       = 29;
constexpr size_t kOffTsRecv     = 32;
constexpr size_t kOffBidPx      = 48;
constexpr size_t kOffAskPx      = 56;
constexpr size_t kOffBidSz      = 64;
constexpr size_t kOffAskSz      = 68;

template <class T>
T load(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> buf(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    return buf;
}

bool isZstd(const std::vector<uint8_t>& buf) {
    return buf.size() >= 4 && load<uint32_t>(buf.data()) == ZSTD_MAGICNUMBER;
}

std::vector<uint8_t> decompress(const std::vector<uint8_t>& in) {
    std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> ctx(ZSTD_createDCtx(), ZSTD_freeDCtx);
    std::vector<uint8_t> out;
    unsigned long long hint = ZSTD_getFrameContentSize(in.data(), in.size());
    if (hint != ZSTD_CONTENTSIZE_UNKNOWN && hint != ZSTD_CONTENTSIZE_ERROR)
        out.reserve(static_cast<size_t>(hint));

    std::vector<uint8_t> chunk(ZSTD_DStreamOutSize());
    ZSTD_inBuffer src{in.data(), in.size(), 0};
    size_t remaining = 0;
    for (;;) {
        ZSTD_outBuffer dst{chunk.data(), chunk.size(), 0};
        remaining = ZSTD_decompressStream(ctx.get(), &dst, &src);
        if (ZSTD_isError(remaining))
            throw std::runtime_error(std::string("zstd: ") + ZSTD_getErrorName(remaining));
        out.insert(out.end(), chunk.data(), chunk.data() + dst.pos);
        if (src.pos == src.size && dst.pos < dst.size) break;
    }
    if (remaining != 0) throw std::runtime_error("zstd: truncated input");
    return out;
}

double toPrice(int64_t px) {
    return px == kUndefPrice ? std::numeric_limits<double>::quiet_NaN()
                             : static_cast<double>(px) * kPriceScale;
}

int8_t toSide(uint8_t c) {
    if (c == 'B') return 1;
    if (c == 'A') return -1;
    return 0;
}

template <class T>
void permute(std::vector<T>& col, const std::vector<size_t>& perm) {
    std::vector<T> tmp(col.size());
    for (size_t i = 0; i < perm.size(); i++) tmp[i] = col[perm[i]];
    col.swap(tmp);
}

void sortByRecv(TradeQuotes& tq) {
    if (std::is_sorted(tq.ts_recv.begin(), tq.ts_recv.end())) return;
    std::vector<size_t> perm(tq.rows());
    std::iota(perm.begin(), perm.end(), size_t{0});
    std::stable_sort(perm.begin(), perm.end(),
                     [&](size_t a, size_t b) { return tq.ts_recv[a] < tq.ts_recv[b]; });
    permute(tq.instrument_id, perm);
    permute(tq.ts_event, perm);
    permute(tq.ts_recv, perm);
    permute(tq.price, perm);
    permute(tq.size, perm);
    permute(tq.side, perm);
    permute(tq.bid_px, perm);
    permute(tq.ask_px, perm);
    permute(tq.bid_sz, perm);
    permute(tq.ask_sz, perm);
}

} // namespace

TradeQuotes readTradeQuotes(const std::string& path) {
    std::vector<uint8_t> raw = readFile(path);
    if (isZstd(raw)) raw = decompress(raw);

    if (raw.size() < 8 || std::memcmp(raw.data(), "DBN", 3) != 0)
        throw std::runtime_error(path + ": not a DBN file");
    if (raw[3] < 1 || raw[3] > 3)
        throw std::runtime_error(path + ": unsupported DBN version " + std::to_string(raw[3]));

    size_t off = 8 + load<uint32_t>(raw.data() + 4); // skip metadata
    if (off > raw.size()) throw std::runtime_error(path + ": truncated metadata");

    TradeQuotes tq;
    size_t capacity = (raw.size() - off) / kRecordBytes;
    tq.instrument_id.reserve(capacity);
    tq.ts_event.reserve(capacity);
    tq.ts_recv.reserve(capacity);
    tq.price.reserve(capacity);
    tq.size.reserve(capacity);
    tq.side.reserve(capacity);
    tq.bid_px.reserve(capacity);
    tq.ask_px.reserve(capacity);
    tq.bid_sz.reserve(capacity);
    tq.ask_sz.reserve(capacity);

    while (off < raw.size()) {
        const uint8_t* r = raw.data() + off;
        size_t len = static_cast<size_t>(r[0]) * 4; // header length is in 4-byte words
        if (len == 0 || off + len > raw.size())
            throw std::runtime_error(path + ": truncated record at byte " + std::to_string(off));
        off += len;

        uint8_t rtype = r[1];
        if (len < kRecordBytes || (rtype != kRtypeMbp1 && rtype != kRtypeCmbp1)) continue;
        if (rtype == kRtypeMbp1 && r[kOffAction] != 'T') continue; // MBP-1 quote updates

        tq.instrument_id.push_back(load<uint32_t>(r + kOffInstrument));
        tq.ts_event.push_back(static_cast<int64_t>(load<uint64_t>(r + kOffTsEvent)));
        tq.ts_recv.push_back(static_cast<int64_t>(load<uint64_t>(r + kOffTsRecv)));
        tq.price.push_back(toPrice(load<int64_t>(r + kOffPrice)));
        tq.size.push_back(load<uint32_t>(r + kOffSize));
        tq.side.push_back(toSide(r[kOffSide]));
        tq.bid_px.push_back(toPrice(load<int64_t>(r + kOffBidPx)));
        tq.ask_px.push_back(toPrice(load<int64_t>(r + kOffAskPx)));
        tq.bid_sz.push_back(load<uint32_t>(r + kOffBidSz));
        tq.ask_sz.push_back(load<uint32_t>(r + kOffAskSz));
    }

    sortByRecv(tq);
    return tq;
}

} // namespace quaml::events
