#include "dataframe.h"
#include "../../compression/encoder/encoder.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

void DataFrame::readCSV(std::string_view csvPath) {
    std::ifstream file;
    file.open(std::string(csvPath));
    if (!file.is_open())
        throw std::runtime_error("readCSV: cannot open file");

    std::string line;
    file >> std::ws;
    std::getline(file, line); // skip header row

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;

        auto next = [&]() -> std::string {
            std::getline(ss, token, ',');
            return token;
        };

        timestamps.push_back(std::stoll(next()));
        open.push_back(std::stod(next()));
        high.push_back(std::stod(next()));
        low.push_back(std::stod(next()));
        close.push_back(std::stod(next()));
        volume.push_back(std::stoll(next()));
    }
}

// --- binary I/O helpers ---

static void writeU32(std::ofstream& f, uint32_t v) {
    f.write(reinterpret_cast<const char*>(&v), 4);
}
static void writeI64(std::ofstream& f, int64_t v) {
    f.write(reinterpret_cast<const char*>(&v), 8);
}
static uint32_t readU32(std::ifstream& f) {
    uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); return v;
}
static int64_t readI64(std::ifstream& f) {
    int64_t v; f.read(reinterpret_cast<char*>(&v), 8); return v;
}

// Pack a bit-string ("010110...") into bytes, MSB first.
static std::vector<uint8_t> packBits(const std::string& bits) {
    std::vector<uint8_t> out((bits.size() + 7) / 8, 0);
    for (size_t i = 0; i < bits.size(); i++) {
        if (bits[i] == '1')
            out[i / 8] |= static_cast<uint8_t>(0x80 >> (i % 8));
    }
    return out;
}

// Unpack bytes back into a bit-string of exactly `n_bits` bits.
static std::string unpackBits(const std::vector<uint8_t>& bytes, uint32_t n_bits) {
    std::string out;
    out.reserve(n_bits);
    for (uint32_t i = 0; i < n_bits; i++) {
        out += ((bytes[i / 8] >> (7 - (i % 8))) & 1) ? '1' : '0';
    }
    return out;
}

static void writeEncoded64(std::ofstream& f, const quaml::Encoded64& enc) {
    writeI64(f, enc.base);
    writeU32(f, static_cast<uint32_t>(enc.codebook.size()));
    for (auto& entry : enc.codebook) {
        writeI64(f, entry.symbol);
        writeU32(f, static_cast<uint32_t>(entry.bits.size()));
        f.write(entry.bits.data(), static_cast<std::streamsize>(entry.bits.size()));
    }
    uint32_t nbits = static_cast<uint32_t>(enc.bitstream.size());
    writeU32(f, nbits);
    auto packed = packBits(enc.bitstream);
    f.write(reinterpret_cast<const char*>(packed.data()),
            static_cast<std::streamsize>(packed.size()));
}

static quaml::Encoded64 readEncoded64(std::ifstream& f) {
    quaml::Encoded64 enc;
    enc.base = readI64(f);
    uint32_t n_entries = readU32(f);
    enc.codebook.resize(n_entries);
    for (auto& entry : enc.codebook) {
        entry.symbol = readI64(f);
        uint32_t codelen = readU32(f);
        entry.bits.resize(codelen);
        f.read(entry.bits.data(), static_cast<std::streamsize>(codelen));
    }
    uint32_t nbits = readU32(f);
    std::vector<uint8_t> packed((nbits + 7) / 8);
    f.read(reinterpret_cast<char*>(packed.data()),
           static_cast<std::streamsize>(packed.size()));
    enc.bitstream = unpackBits(packed, nbits);
    return enc;
}

static double bitsToDouble(int64_t bits) {
    double v; std::memcpy(&v, &bits, 8); return v;
}
static int64_t doubleToBits(double v) {
    int64_t bits; std::memcpy(&bits, &v, 8); return bits;
}

static const char QML_MAGIC[4] = {'Q','U','M','L'};

void DataFrame::saveQML(std::string_view outPath) {
    std::ofstream f;
    f.open(std::string(outPath), std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("saveQML: cannot open file for writing");

    f.write(QML_MAGIC, 4);
    writeU32(f, static_cast<uint32_t>(timestamps.size()));

    // timestamps and volume as int64
    writeEncoded64(f, quaml::Encode64(timestamps));
    writeEncoded64(f, quaml::Encode64(volume));

    // doubles: bit-cast to int64 then encode
    auto encodeDoubles = [&](const std::vector<double>& col) {
        std::vector<int64_t> bits(col.size());
        for (size_t i = 0; i < col.size(); i++) bits[i] = doubleToBits(col[i]);
        writeEncoded64(f, quaml::Encode64(bits));
    };
    encodeDoubles(open);
    encodeDoubles(high);
    encodeDoubles(low);
    encodeDoubles(close);
}

void DataFrame::readQML(std::string_view qmlPath) {
    std::ifstream f;
    f.open(std::string(qmlPath), std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("readQML: cannot open file");

    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, QML_MAGIC, 4) != 0)
        throw std::runtime_error("readQML: invalid file format");

    uint32_t n_bars = readU32(f);
    timestamps.reserve(n_bars);
    volume.reserve(n_bars);
    open.reserve(n_bars);
    high.reserve(n_bars);
    low.reserve(n_bars);
    close.reserve(n_bars);

    timestamps = quaml::Decode64(readEncoded64(f));
    volume     = quaml::Decode64(readEncoded64(f));

    auto decodeDoubles = [&](std::vector<double>& col) {
        auto bits = quaml::Decode64(readEncoded64(f));
        col.resize(bits.size());
        for (size_t i = 0; i < bits.size(); i++) col[i] = bitsToDouble(bits[i]);
    };
    decodeDoubles(open);
    decodeDoubles(high);
    decodeDoubles(low);
    decodeDoubles(close);
}

// Returns the index of ts via binary search, or throws if not found.
static size_t exactIndex(const std::vector<int64_t>& ts, int64_t target) {
    auto it = std::lower_bound(ts.begin(), ts.end(), target);
    if (it == ts.end() || *it != target)
        throw std::out_of_range("getBarExact: timestamp not found");
    return static_cast<size_t>(it - ts.begin());
}

BarView DataFrame::getNextBar(BarView& thisBar){
    //given a bar, get the next bar
    if(!hasNextBar(thisBar)) {
        throw std::out_of_range("getNextBar: next bar not found");
    }

    return BarView{*this, thisBar.idx + 1};
}

bool DataFrame::hasNextBar(BarView& thisBar) const{
    return thisBar.idx < timestamps.size() - 1;
}

BarView DataFrame::getBarExact(int64_t ts) {
    return BarView{*this, exactIndex(timestamps, ts)};
}

BarView DataFrame::getBarInexact(int64_t ts) {
    return getBarInexact(ts, NearestBarType::Nearest);
}

BarView DataFrame::getBarInexact(int64_t ts, NearestBarType type) {
    if (timestamps.empty())
        throw std::out_of_range("getBarInexact: DataFrame is empty");

    auto it = std::lower_bound(timestamps.begin(), timestamps.end(), ts);

    if (type == NearestBarType::Forward) {
        // First timestamp >= ts.
        if (it == timestamps.end())
            throw std::out_of_range("getBarInexact: no bar at or after timestamp");
        return BarView{*this, static_cast<size_t>(it - timestamps.begin())};
    }

    if (type == NearestBarType::Backward) {
        // Last timestamp <= ts.
        if (it == timestamps.begin())
            throw std::out_of_range("getBarInexact: no bar at or before timestamp");
        --it;
        return BarView{*this, static_cast<size_t>(it - timestamps.begin())};
    }

    // NearestBarType::Nearest: pick whichever of the two neighbours is closer.
    if (it == timestamps.end())
        return BarView{*this, timestamps.size() - 1};
    if (it == timestamps.begin())
        return BarView{*this, 0};

    size_t after  = static_cast<size_t>(it - timestamps.begin());
    size_t before = after - 1;
    int64_t distAfter  = timestamps[after]  - ts;
    int64_t distBefore = ts - timestamps[before];
    return BarView{*this, distBefore <= distAfter ? before : after};
}

std::vector<BarView> DataFrame::getBars(int64_t begin, int64_t end) {
    // lower_bound for begin, upper_bound for end gives [begin, end] inclusive.
    auto lo = std::lower_bound(timestamps.begin(), timestamps.end(), begin);
    auto hi = std::upper_bound(timestamps.begin(), timestamps.end(), end);

    std::vector<BarView> result;
    result.reserve(static_cast<size_t>(hi - lo));
    for (auto it = lo; it != hi; ++it)
        result.push_back(BarView{*this, static_cast<size_t>(it - timestamps.begin())});
    return result;
}
