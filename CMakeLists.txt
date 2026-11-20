cmake_minimum_required(VERSION 3.16)

project(
    MiniTensorIRRuntime
    LANGUAGES CXX
)

set(
    CMAKE_CXX_STANDARD
    17
)

set(
    CMAKE_CXX_STANDARD_REQUIRED
    ON
)

add_executable(
    phase1_matmul
    src/phase1_matmul.cpp
)