#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace quaml::ml {

// All functions return a vector of the same length as the input.
// Entries where there is insufficient history are set to NaN.

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// Arithmetic mean over a rolling window.
std::vector<double> rollingMean(const std::vector<double>& col, int window);

// Sample standard deviation over a rolling window.
std::vector<double> rollingStd(const std::vector<double>& col, int window);

// Z-score: (value - rolling mean) / rolling std.
std::vector<double> zScore(const std::vector<double>& col, int window);

// Momentum: col[i] - col[i - lag].
std::vector<double> momentum(const std::vector<double>& col, int lag);

// Rate of change: (col[i] - col[i - lag]) / col[i - lag].
std::vector<double> roc(const std::vector<double>& col, int lag);

// Exponential moving average (span-based, like pandas ewm(span=n)).
std::vector<double> ema(const std::vector<double>& col, int span);

// Shift col forward by lag bars (positive lag = look back). Fills head with NaN.
std::vector<double> shift(const std::vector<double>& col, int lag);

} // namespace quaml::ml
