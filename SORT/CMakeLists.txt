cmake_minimum_required(VERSION 3.10)

project(KalmanFilter CXX)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Add a C++ executable
add_executable(KalmanFilter main.cpp)

# Link the libraries to the executable
target_link_libraries(KalmanFilter PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
)