#include <iostream>
#include <Eigen/Dense>

int main() {
    // Define an invertible square matrix
    Eigen::Matrix3d A;
    A << 1, 2, 3,
         0, 1, 4,
         5, 6, 0;

    // Compute the inverse
    Eigen::Matrix3d A_inv = A.inverse();

    // Print the matrix and its inverse
    std::cout << "Original Matrix A:\n" << A << "\n\n";
    std::cout << "Inverse Matrix A_inv:\n" << A_inv << "\n\n";

    // Verify the inverse by multiplying A * A_inv
    // The result should be the identity matrix
    Eigen::Matrix3d I = A * A_inv;
    std::cout << "A * A_inv:\n" << I << "\n";

    return 0;
}