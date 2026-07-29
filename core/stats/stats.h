#pragma once

#include <vector>

namespace quaml::stats {

// Annualised Sharpe ratio. bars_per_year = 252 for daily, 252*390 for 1-min bars.
double sharpe(const std::vector<double>& equity, double bars_per_year = 252.0);

// Largest peak-to-trough decline in the equity curve (returned as a positive number).
double maxDrawdown(const std::vector<double>& equity);

// Compound annual growth rate.
double cagr(const std::vector<double>& equity, double bars_per_year = 252.0);

// Average return per bar.
double meanReturn(const std::vector<double>& equity);

} // namespace quaml::stats
