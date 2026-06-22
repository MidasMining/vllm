cmake_minimum_required(VERSION 3.20)
project(turboquant-sycl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SYCL flags applied per-target (not globally, so CPU target works)
set(SYCL_COMPILE_FLAGS "-fsycl")
set(SYCL_LINK_FLAGS "-fsycl")

# Uncomment for AOT compilation to BMG:
# list(APPEND SYCL_COMPILE_FLAGS "-fsycl-targets=intel_gpu_bmg_g31")
# list(APPEND SYCL_LINK_FLAGS "-fsycl-targets=intel_gpu_bmg_g31")

# --- CPU-only correctness test (no SYCL, compiles with any C++17 compiler) ---
add_executable(tq_correctness_cpu src/tq_correctness.cpp)
target_compile_options(tq_correctness_cpu PRIVATE -O2)

# --- SYCL correctness test (GPU + CPU validation) ---
add_executable(tq_correctness_gpu src/tq_correctness.cpp)
target_compile_definitions(tq_correctness_gpu PRIVATE USE_SYCL)
target_compile_options(tq_correctness_gpu PRIVATE ${SYCL_COMPILE_FLAGS} -O2)
target_link_options(tq_correctness_gpu PRIVATE ${SYCL_LINK_FLAGS})

# --- SYCL benchmark harness ---
add_executable(tq_bench src/tq_bench.cpp)
target_compile_options(tq_bench PRIVATE ${SYCL_COMPILE_FLAGS} -O2)
target_link_options(tq_bench PRIVATE ${SYCL_LINK_FLAGS})
