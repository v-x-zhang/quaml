#include "delta_encode.h"

std::vector<int32_t> DeltaEncode(const std::vector<int32_t>& values) {
    if (values.empty()) return {};
    std::vector<int32_t> out;
    out.reserve(values.size());
    out.push_back(values[0]);
    for (size_t i = 1; i < values.size(); i++) {
        out.push_back(values[i] - values[i - 1]);
    }
    return out;
}

std::vector<int32_t> DeltaDecode(const std::vector<int32_t>& deltas) {
    if (deltas.empty()) return {};
    std::vector<int32_t> out;
    out.reserve(deltas.size());
    out.push_back(deltas[0]);
    for (size_t i = 1; i < deltas.size(); i++) {
        out.push_back(out.back() + deltas[i]);
    }
    return out;
}
