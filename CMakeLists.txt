cmake_minimum_required(VERSION 3.14)
project(AccretiveComputingMAS)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization Flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

include(FetchContent)

# Fetch pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.1
)
FetchContent_MakeAvailable(pybind11)

# Fetch Eigen3
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)

# Include PyBind11 and Eigen
pybind11_add_module(_core
    src/cpp/bindings.cpp
    src/cpp/MicroAgent.cpp
    src/cpp/MacroAgent.cpp
    src/cpp/ContinuousSubstrate.cpp
    src/cpp/LatentMicroAgent.cpp
)

target_include_directories(_core PRIVATE include)
target_link_libraries(_core PRIVATE Eigen3::Eigen)
