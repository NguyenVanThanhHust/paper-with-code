#include <iostream>
#include <Eigen/Dense>

int main() {
    // Define two matrices, A (3x2) and B (2x3)
    Eigen::MatrixXd A(3, 2);
    A << 1, 2,
         3, 4,
         5, 6;

    Eigen::MatrixXd B(2, 3);
    B << 7, 8, 9,
         10, 11, 12;

    // Perform matrix multiplication: C = A * B
    // The resulting matrix C will be 3x3
    Eigen::MatrixXd C = A * B;

    // Print the result
    std::cout << "Matrix A:\n" << A << "\n\n";
    std::cout << "Matrix B:\n" << B << "\n\n";
    std::cout << "Matrix C (A * B):\n" << C << "\n";

    return 0;
}