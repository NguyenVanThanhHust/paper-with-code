cmake_minimum_required(VERSION 3.10)
project(EigenMatmul)

# Find the Eigen3 package. This module is provided by CMake and finds the library and sets the EIGEN3_INCLUDE_DIRS variable.
find_package(Eigen3 REQUIRED)

# Add an executable for your C++ source file
add_executable(EigenMatmul matmul.cpp)

# Link the executable with the Eigen3 include directories.
# This makes sure the compiler knows where to find the Eigen header files.
target_include_directories(EigenMatmul PUBLIC ${EIGEN3_INCLUDE_DIRS})

# Add an executable for your C++ source file
add_executable(IsInvertible check_invertible.cpp)

# Link the executable with the Eigen3 include directories.
# This makes sure the compiler knows where to find the Eigen header files.
target_include_directories(IsInvertible PUBLIC ${EIGEN3_INCLUDE_DIRS})

# Add an executable for your C++ source file
add_executable(InverseMatrix inverse_matrix.cpp)

# Link the executable with the Eigen3 include directories.
# This makes sure the compiler knows where to find the Eigen header files.
target_include_directories(InverseMatrix PUBLIC ${EIGEN3_INCLUDE_DIRS})

# Add an executable for your C++ source file
add_executable(LinearSolver linear_solver.cpp)

# Link the executable with the Eigen3 include directories.
# This makes sure the compiler knows where to find the Eigen header files.
target_include_directories(LinearSolver PUBLIC ${EIGEN3_INCLUDE_DIRS})