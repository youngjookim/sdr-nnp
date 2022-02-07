cmake_minimum_required(VERSION 3.10)

project(sdr
        LANGUAGES CXX
        VERSION 0.0.1
        DESCRIPTION "High-dimensional Sharpened Dimensionality Reduction (HD-SDR)"
        HOMEPAGE_URL "https://github.com/youngjookim/sdr")

set(SOURCES "main.cpp")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native")

find_package (Eigen3 3.3.9 REQUIRED NO_MODULE)
include_directories(include)

add_executable(sdr ${SOURCES})
target_link_libraries(sdr Eigen3::Eigen stdc++fs)