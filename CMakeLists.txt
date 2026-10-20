cmake_minimum_required(VERSION 3.16)
project(TinySparse VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable strict warning flags
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Header-only interface target
add_library(TinySparse INTERFACE)
target_include_directories(TinySparse INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Automated testing targets
enable_testing()

# 1. RingBuffer1D Boundary & Unwrap Unit Tests
add_executable(test_ring_buffer tests/test_ring_buffer.cpp)
target_link_libraries(test_ring_buffer PRIVATE TinySparse)
add_test(NAME test_ring_buffer COMMAND test_ring_buffer)

# 2. Host C++ Bit-Exact Numerical Verification Suite
add_executable(verify_inference examples/pc_verify/verify_inference.cpp)
target_link_libraries(verify_inference PRIVATE TinySparse)
add_test(NAME verify_inference COMMAND verify_inference)