#pragma once

#include <vector>
#include <stdexcept>

namespace quaml::math {

struct Matrix {
    int rows, cols;
    std::vector<double> data; // flat row-major: (i,j) -> data[i*cols + j]

    Matrix(int rows, int cols)
        : rows(rows), cols(cols), data(rows * cols, 0.0) {}

    double& operator()(int i, int j)       { return data[i * cols + j]; }
    double  operator()(int i, int j) const { return data[i * cols + j]; }
};

// C = A * B  (A: m×k, B: k×n → C: m×n)
Matrix multiply(const Matrix& A, const Matrix& B);

// Returns Aᵀ
Matrix transpose(const Matrix& A);

// Returns A⁻¹ via Gaussian elimination with partial pivoting. Throws if singular.
Matrix inverse(const Matrix& A);

// Computes XᵀX directly without materialising Xᵀ. More cache-friendly than
// multiply(transpose(X), X) for tall matrices.
Matrix gramMatrix(const Matrix& X);

// Build a Matrix from a list of column vectors (each col must be the same length).
Matrix fromColumns(const std::vector<std::vector<double>>& cols);

} // namespace quaml::math
