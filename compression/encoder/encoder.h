#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace quaml {

// Codebook entry: maps a symbol (delta value) to its Huffman bit-string.
struct CodeEntry {
    int32_t     symbol;
    std::string bits;
};

// Self-contained result of encoding. Everything the decoder needs is in here —
// no shared state between encoder and decoder.
struct Encoded {
    int32_t                 base;      // first value, stored verbatim
    std::string             bitstream; // Huffman-compressed deltas
    std::vector<CodeEntry>  codebook;  // symbol -> bit-string mapping
};

// Compress a sequence of integers using delta encoding followed by Huffman coding.
// Input must have at least one element.
Encoded Encode(const std::vector<int32_t>& values);

// Reconstruct the original sequence from an Encoded result.
std::vector<int32_t> Decode(const Encoded& enc);

// 64-bit variant — used for timestamps, volume, and bit-reinterpreted doubles.
struct CodeEntry64 {
    int64_t     symbol;
    std::string bits;
};

struct Encoded64 {
    int64_t                  base;
    std::string              bitstream;
    std::vector<CodeEntry64> codebook;
};

Encoded64            Encode64(const std::vector<int64_t>& values);
std::vector<int64_t> Decode64(const Encoded64& enc);

} // namespace quaml
