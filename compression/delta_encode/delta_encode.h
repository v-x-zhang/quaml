#pragma once

#include <cstdint>
#include <vector>

// Converts absolute values into [base, d1, d2, ...] where each d_i = prices[i] - prices[i-1].
std::vector<int32_t> DeltaEncode(const std::vector<int32_t>& values);

// Reconstructs original values from a delta-encoded sequence.
std::vector<int32_t> DeltaDecode(const std::vector<int32_t>& deltas);
