#include <iostream>
#include <vector>
#include <cmath>

using Matrix = std::vector<std::vector<float>>;

Matrix matmul(
    const Matrix& A,
    const Matrix& B
) {
    const int M = static_cast<int>(A.size());
    const int K = static_cast<int>(A[0].size());
    const int N = static_cast<int>(B[0].size());

    Matrix C(
        M,
        std::vector<float>(N, 0.0f)
    );

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C[i][j] +=
                    A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

void print_matrix(const Matrix& matrix) {
    for (const auto& row : matrix) {
        for (float value : row) {
            std::cout << value << " ";
        }

        std::cout << "\n";
    }
}

int main() {

    Matrix A = {
        {1, 2, 3},
        {4, 5, 6}
    };

    Matrix B = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };

    Matrix C = matmul(A, B);

    std::cout << "C = A x B\n";
    print_matrix(C);

    return 0;
}