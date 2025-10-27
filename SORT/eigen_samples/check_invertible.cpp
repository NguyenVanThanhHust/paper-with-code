#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>

int main() {
    Eigen::Matrix3d A;
    A << 1, 2, 3,
         2, 4, 6,
         0, 1, 4;

    Eigen::FullPivLU<Eigen::Matrix3d> lu(A);

    if (lu.isInvertible()) {
        std::cout << "Matrix is invertible. Rank: " << lu.rank() << "\n";
        Eigen::Matrix3d A_inv = lu.inverse();
        std::cout << "Inverse:\n" << A_inv << "\n";
    } else {
        std::cout << "Matrix is not invertible. Rank: " << lu.rank() << "\n";
    }

    return 0;
}