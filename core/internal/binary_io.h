#pragma once

// Internal binary I/O helpers shared by dataframe and engine.
// Not part of the public API.

#include "../../compression/encoder/encoder.h"
#include <cstring>
#include <fstream>
#include <vector>

namespace quaml::internal {

inline void writeU32(std::ofstream& f, uint32_t v) {
    f.write(reinterpret_cast<const char*>(&v), 4);
}
inline void writeI64(std::ofstream& f, int64_t v) {
    f.write(reinterpret_cast<const char*>(&v), 8);
}
inline uint32_t readU32(std::ifstream& f) {
    uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); return v;
}
inline int64_t readI64(std::ifstream& f) {
    int64_t v; f.read(reinterpret_cast<char*>(&v), 8); return v;
}

inline std::vector<uint8_t> packBits(const std::string& bits) {
    std::vector<uint8_t> out((bits.size() + 7) / 8, 0);
    for (size_t i = 0; i < bits.size(); i++)
        if (bits[i] == '1')
            out[i / 8] |= static_cast<uint8_t>(0x80 >> (i % 8));
    return out;
}

inline std::string unpackBits(const std::vector<uint8_t>& bytes, uint32_t n_bits) {
    std::string out;
    out.reserve(n_bits);
    for (uint32_t i = 0; i < n_bits; i++)
        out += ((bytes[i / 8] >> (7 - (i % 8))) & 1) ? '1' : '0';
    return out;
}

inline void writeEncoded64(std::ofstream& f, const Encoded64& enc) {
    writeI64(f, enc.base);
    writeU32(f, static_cast<uint32_t>(enc.codebook.size()));
    for (auto& e : enc.codebook) {
        writeI64(f, e.symbol);
        writeU32(f, static_cast<uint32_t>(e.bits.size()));
        f.write(e.bits.data(), static_cast<std::streamsize>(e.bits.size()));
    }
    uint32_t nbits = static_cast<uint32_t>(enc.bitstream.size());
    writeU32(f, nbits);
    auto packed = packBits(enc.bitstream);
    f.write(reinterpret_cast<const char*>(packed.data()),
            static_cast<std::streamsize>(packed.size()));
}

inline Encoded64 readEncoded64(std::ifstream& f) {
    Encoded64 enc;
    enc.base = readI64(f);
    uint32_t n = readU32(f);
    enc.codebook.resize(n);
    for (auto& e : enc.codebook) {
        e.symbol = readI64(f);
        uint32_t codelen = readU32(f);
        e.bits.resize(codelen);
        f.read(e.bits.data(), static_cast<std::streamsize>(codelen));
    }
    uint32_t nbits = readU32(f);
    std::vector<uint8_t> packed((nbits + 7) / 8);
    f.read(reinterpret_cast<char*>(packed.data()),
           static_cast<std::streamsize>(packed.size()));
    enc.bitstream = unpackBits(packed, nbits);
    return enc;
}

inline double bitsToDouble(int64_t bits) {
    double v; std::memcpy(&v, &bits, 8); return v;
}
inline int64_t doubleToBits(double v) {
    int64_t bits; std::memcpy(&bits, &v, 8); return bits;
}

} // namespace quaml::internal
