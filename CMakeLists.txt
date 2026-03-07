cmake_minimum_required(VERSION 3.14)
project(rans_sst LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")

# ---- Core library ----
add_library(rans_sst_core STATIC
    src/Mesh.cpp
    src/SSTModel.cpp
    src/SIMPLESolver.cpp
    src/ForwardModel.cpp
)
target_include_directories(rans_sst_core PUBLIC include)
target_link_libraries(rans_sst_core PUBLIC pthread)

# ---- CLI executable ----
add_executable(rans_sst src/main.cpp)
target_link_libraries(rans_sst PRIVATE rans_sst_core)

# ---- Optional: pybind11 Python bindings ----
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python module" OFF)
if(BUILD_PYTHON_BINDINGS)

    set(PYBIND11_FINDPYTHON ON)

    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(rans_sst_py python/Bindings.cpp)

    target_link_libraries(rans_sst_py PRIVATE rans_sst_core)

endif()