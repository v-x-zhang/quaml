#pragma once

#include "../math/matrix.h"
#include <vector>

namespace quaml::ml {

class OLS {
    std::vector<double> w; // learned weights, one per column of X

public:
    // Fits OLS weights. Rows where any X value or y is NaN are skipped.
    void fit(const math::Matrix& X, const std::vector<double>& y);

    // Returns X * w (one prediction per row of X). NaN rows produce NaN output.
    std::vector<double> predict(const math::Matrix& X) const;

    const std::vector<double>& weights() const { return w; }

    // Coefficient of determination on the provided X and y.
    double r2(const math::Matrix& X, const std::vector<double>& y) const;
};

} // namespace quaml::ml
