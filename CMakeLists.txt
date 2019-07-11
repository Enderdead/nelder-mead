cmake_minimum_required(VERSION 3.14)
project(nelder_mead)

set(CMAKE_CXX_STANDARD 11)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(nelder_mead example.cpp NelderMeadOptimizer.h)
target_link_libraries (nelder_mead Eigen3::Eigen)
