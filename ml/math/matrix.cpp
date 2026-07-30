#include "matrix.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quaml::math {

Matrix multiply(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows)
        throw std::invalid_argument("multiply: incompatible dimensions");

    Matrix C(A.rows, B.cols);

    for (int i = 0; i < A.rows; i++) {
        for (int k = 0; k < A.cols; k++) {
            double aik = A(i, k);
            for (int j = 0; j < B.cols; j++) {
                C(i, j) += aik * B(k, j);
            }
        }
    }
    return C;
}

Matrix transpose(const Matrix& A) {
    Matrix T(A.cols, A.rows);
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < A.cols; j++)
            T(j, i) = A(i, j);
    return T;
}

Matrix inverse(const Matrix& A) {
    if (A.rows != A.cols)
        throw std::invalid_argument("inverse: matrix must be square");

    int n = A.rows;

    Matrix aug(n, 2 * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            aug(i, j) = A(i, j);
        aug(i, n + i) = 1.0; // identity on the right half
    }

    for (int k = 0; k < n; k++) {
        int pivot = k;
        double maxVal = std::abs(aug(k, k));
        for (int i = k + 1; i < n; i++) {
            if (std::abs(aug(i, k)) > maxVal) {
                maxVal = std::abs(aug(i, k));
                pivot = i;
            }
        }
        if (maxVal < 1e-12)
            throw std::runtime_error("inverse: matrix is singular or near-singular");

        // Swap pivot row with row k (swap entire row across both halves)
        if (pivot != k)
            for (int j = 0; j < 2 * n; j++)
                std::swap(aug(k, j), aug(pivot, j));

        // Scale row k so the pivot becomes 1
        double scale = aug(k, k);
        for (int j = 0; j < 2 * n; j++)
            aug(k, j) /= scale;

        // Eliminate column k from all other rows
        for (int i = 0; i < n; i++) {
            if (i == k) continue;
            double factor = aug(i, k);
            for (int j = 0; j < 2 * n; j++)
                aug(i, j) -= factor * aug(k, j);
        }
    }

    // Extract the right half as A inverse
    Matrix inv(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            inv(i, j) = aug(i, n + j);
    return inv;
}

Matrix gramMatrix(const Matrix& X) {
    // Inner loop iterates rows of X sequentially — cache-friendly.
    int n = X.rows, k = X.cols;
    Matrix G(k, k);
    for (int row = 0; row < n; row++)
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                G(i, j) += X(row, i) * X(row, j);
    return G;
}

Matrix fromColumns(const std::vector<std::vector<double>>& cols) {
    if (cols.empty()) throw std::invalid_argument("fromColumns: no columns provided");
    int n = static_cast<int>(cols[0].size());
    int k = static_cast<int>(cols.size());
    for (auto& c : cols)
        if ((int)c.size() != n)
            throw std::invalid_argument("fromColumns: columns must all be the same length");

    Matrix M(n, k);
    for (int j = 0; j < k; j++)
        for (int i = 0; i < n; i++)
            M(i, j) = cols[j][i];
    return M;
}

} // namespace quaml::math
