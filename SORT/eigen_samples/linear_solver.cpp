#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix3d A;
    A << 1, 2, 3,
         0, 1, 4,
         5, 6, 0;

    Eigen::Vector3d b(1, 1, 1);

    // Solve Ax = b for x
    // This is more stable and efficient than A.inverse() * b
    Eigen::Vector3d x = A.partialPivLu().solve(b);

    std::cout << "Solution x:\n" << x << "\n\n";
    std::cout << "Check: A * x\n" << A * x << "\n";
    
    return 0;
}