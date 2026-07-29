#include "stats.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace quaml::stats {

static std::vector<double> returns(const std::vector<double>& equity) {
    if (equity.size() < 2) return {};
    std::vector<double> r;
    r.reserve(equity.size() - 1);
    for (size_t i = 1; i < equity.size(); i++)
        r.push_back((equity[i] - equity[i - 1]) / equity[i - 1]);
    return r;
}

double meanReturn(const std::vector<double>& equity) {
    auto r = returns(equity);
    if (r.empty()) return 0.0;
    return std::accumulate(r.begin(), r.end(), 0.0) / r.size();
}

double sharpe(const std::vector<double>& equity, double bars_per_year) {
    auto r = returns(equity);
    if (r.size() < 2) return 0.0;

    double mean = std::accumulate(r.begin(), r.end(), 0.0) / r.size();
    double var  = 0.0;
    for (double x : r) var += (x - mean) * (x - mean);
    double stddev = std::sqrt(var / (r.size() - 1));

    if (stddev == 0.0) return 0.0;
    return (mean / stddev) * std::sqrt(bars_per_year);
}

double maxDrawdown(const std::vector<double>& equity) {
    double peak = equity.empty() ? 0.0 : equity[0];
    double max_dd = 0.0;
    for (double v : equity) {
        if (v > peak) peak = v;
        double dd = (peak - v) / peak;
        if (dd > max_dd) max_dd = dd;
    }
    return max_dd;
}

double cagr(const std::vector<double>& equity, double bars_per_year) {
    if (equity.size() < 2 || equity.front() == 0.0) return 0.0;
    double total_return = equity.back() / equity.front();
    double years = (equity.size() - 1) / bars_per_year;
    return std::pow(total_return, 1.0 / years) - 1.0;
}

} // namespace quaml::stats
