#include "encoder.h"
#include "../delta_encode/delta_encode.h"
#include "../huffman_encoding/huffman.h"

#include <stdexcept>
#include <unordered_map>

namespace quaml {

Encoded Encode(const std::vector<int32_t>& values) {
    if (values.empty()) throw std::invalid_argument("Encode: input is empty");

    std::vector<int32_t> deltas = DeltaEncode(values);
    // deltas[0] is the base value; Huffman compresses everything after.
    std::vector<int32_t> delta_body(deltas.begin() + 1, deltas.end());

    Huffman huff;
    huff.BuildTree(delta_body);
    huff.BuildMap();

    Encoded result;
    result.base      = deltas[0];
    result.bitstream = huff.Encode();

    for (auto& [sym, bits] : huff.GetMap()) {
        result.codebook.push_back({sym, bits});
    }

    return result;
}

std::vector<int32_t> Decode(const Encoded& enc) {
    // Rebuild reverse map: bit-string -> symbol.
    std::unordered_map<std::string, int32_t> reverse;
    for (auto& entry : enc.codebook) {
        reverse[entry.bits] = entry.symbol;
    }

    // Walk the bitstream, collecting symbols as prefix codes complete.
    std::vector<int32_t> delta_body;
    std::string current;
    for (char b : enc.bitstream) {
        current += b;
        auto it = reverse.find(current);
        if (it != reverse.end()) {
            delta_body.push_back(it->second);
            current.clear();
        }
    }

    // Prepend base and delta-decode to recover original values.
    std::vector<int32_t> deltas;
    deltas.reserve(delta_body.size() + 1);
    deltas.push_back(enc.base);
    deltas.insert(deltas.end(), delta_body.begin(), delta_body.end());

    return DeltaDecode(deltas);
}

} // namespace quaml
