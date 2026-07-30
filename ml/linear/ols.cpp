#include "ols.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace quaml::ml {

// Returns true if any value in the row, or the corresponding y, is NaN.
static bool hasNaN(const math::Matrix& X, int row, double yval) {
    if (std::isnan(yval)) return true;
    for (int j = 0; j < X.cols; j++)
        if (std::isnan(X(row, j))) return true;
    return false;
}

void OLS::fit(const math::Matrix& X, const std::vector<double>& y) {
    if ((int)y.size() != X.rows)
        throw std::invalid_argument("OLS::fit: X and y must have the same number of rows");

    // Count valid rows first so we can allocate the clean matrix in one shot.
    int valid = 0;
    for (int i = 0; i < X.rows; i++)
        if (!hasNaN(X, i, y[i])) valid++;

    if (valid < X.cols)
        throw std::runtime_error("OLS::fit: fewer valid rows than features — underdetermined");

    // Copy only valid rows into Xc and yc.
    math::Matrix Xc(valid, X.cols);
    std::vector<double> yc;
    yc.reserve(valid);

    int row = 0;
    for (int i = 0; i < X.rows; i++) {
        if (hasNaN(X, i, y[i])) continue;
        for (int j = 0; j < X.cols; j++)
            Xc(row, j) = X(i, j);
        yc.push_back(y[i]);
        row++;
    }

    // OLS closed form: w = (XᵀX)⁻¹ Xᵀy
    math::Matrix XtX    = math::gramMatrix(Xc);
    math::Matrix XtX_inv = math::inverse(XtX);
    math::Matrix Xt     = math::transpose(Xc);

    // Xᵀy — matrix × vector, result is k×1.
    w.assign(Xc.cols, 0.0);
    for (int i = 0; i < Xc.cols; i++)
        for (int j = 0; j < valid; j++)
            w[i] += Xt(i, j) * yc[j];

    // w = (XᵀX)⁻¹ * (Xᵀy)
    std::vector<double> result(Xc.cols, 0.0);
    for (int i = 0; i < Xc.cols; i++)
        for (int j = 0; j < Xc.cols; j++)
            result[i] += XtX_inv(i, j) * w[j];

    w = result;
}

std::vector<double> OLS::predict(const math::Matrix& X) const {
    if (w.empty())
        throw std::runtime_error("OLS::predict: model has not been fitted");
    if ((int)w.size() != X.cols)
        throw std::invalid_argument("OLS::predict: X has wrong number of columns");

    std::vector<double> preds(X.rows, 0.0);
    for (int i = 0; i < X.rows; i++) {
        // NaN rows propagate NaN rather than producing garbage predictions.
        bool nan_row = false;
        for (int j = 0; j < X.cols; j++)
            if (std::isnan(X(i, j))) { nan_row = true; break; }

        if (nan_row) {
            preds[i] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        for (int j = 0; j < X.cols; j++)
            preds[i] += w[j] * X(i, j);
    }
    return preds;
}

double OLS::r2(const math::Matrix& X, const std::vector<double>& y) const {
    auto preds = predict(X);

    // Compute mean of valid y values.
    double sum = 0.0;
    int valid = 0;
    for (int i = 0; i < (int)y.size(); i++) {
        if (std::isnan(y[i]) || std::isnan(preds[i])) continue;
        sum += y[i];
        valid++;
    }
    if (valid == 0) return 0.0;
    double mean = sum / valid;

    double ss_res = 0.0, ss_tot = 0.0;
    for (int i = 0; i < (int)y.size(); i++) {
        if (std::isnan(y[i]) || std::isnan(preds[i])) continue;
        double diff = y[i] - preds[i];
        ss_res += diff * diff;
        ss_tot += (y[i] - mean) * (y[i] - mean);
    }
    if (ss_tot == 0.0) return 1.0;
    return 1.0 - (ss_res / ss_tot);
}

} // namespace quaml::ml
