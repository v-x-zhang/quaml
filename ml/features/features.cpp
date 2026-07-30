#include "features.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace quaml::ml {

std::vector<double> rollingMean(const std::vector<double>& col, int window) {
    if (window <= 0) throw std::invalid_argument("rollingMean: window must be > 0");
    std::vector<double> out(col.size(), NaN);
    double sum = 0.0;
    for (int i = 0; i < (int)col.size(); i++) {
        sum += col[i];
        if (i >= window)
            sum -= col[i - window];
        if (i >= window - 1)
            out[i] = sum / window;
    }
    return out;
}

std::vector<double> rollingStd(const std::vector<double>& col, int window) {
    if (window < 2) throw std::invalid_argument("rollingStd: window must be >= 2");
    std::vector<double> out(col.size(), NaN);
    for (int i = window - 1; i < (int)col.size(); i++) {
        double mean = 0.0;
        for (int j = i - window + 1; j <= i; j++) mean += col[j];
        mean /= window;
        double var = 0.0;
        for (int j = i - window + 1; j <= i; j++) var += (col[j] - mean) * (col[j] - mean);
        out[i] = std::sqrt(var / (window - 1));
    }
    return out;
}

std::vector<double> zScore(const std::vector<double>& col, int window) {
    auto mean = rollingMean(col, window);
    auto std  = rollingStd(col, window);
    std::vector<double> out(col.size(), NaN);
    for (int i = 0; i < (int)col.size(); i++) {
        if (std::isnan(mean[i]) || std[i] == 0.0) continue;
        out[i] = (col[i] - mean[i]) / std[i];
    }
    return out;
}

std::vector<double> momentum(const std::vector<double>& col, int lag) {
    if (lag <= 0) throw std::invalid_argument("momentum: lag must be > 0");
    std::vector<double> out(col.size(), NaN);
    for (int i = lag; i < (int)col.size(); i++)
        out[i] = col[i] - col[i - lag];
    return out;
}

std::vector<double> roc(const std::vector<double>& col, int lag) {
    if (lag <= 0) throw std::invalid_argument("roc: lag must be > 0");
    std::vector<double> out(col.size(), NaN);
    for (int i = lag; i < (int)col.size(); i++) {
        if (col[i - lag] != 0.0)
            out[i] = (col[i] - col[i - lag]) / col[i - lag];
    }
    return out;
}

std::vector<double> ema(const std::vector<double>& col, int span) {
    if (span <= 0) throw std::invalid_argument("ema: span must be > 0");
    std::vector<double> out(col.size(), NaN);
    double alpha = 2.0 / (span + 1.0);
    for (int i = 0; i < (int)col.size(); i++) {
        if (i == 0) { out[i] = col[i]; continue; }
        out[i] = alpha * col[i] + (1.0 - alpha) * out[i - 1];
    }
    return out;
}

std::vector<double> shift(const std::vector<double>& col, int lag) {
    if (lag <= 0) throw std::invalid_argument("shift: lag must be > 0");
    std::vector<double> out(col.size(), NaN);
    for (int i = lag; i < (int)col.size(); i++)
        out[i] = col[i - lag];
    return out;
}

} // namespace quaml::ml
