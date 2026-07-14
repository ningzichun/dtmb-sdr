cmake_minimum_required(VERSION 3.20)

project(dtmb_core VERSION 0.2.0 LANGUAGES C CXX)

option(DTMB_CORE_BUILD_TESTS "Build dtmb-core tests" ON)
option(DTMB_CORE_ENABLE_CUDA_LDPC "Build the optional CUDA LDPC backend" OFF)
option(DTMB_CORE_ENABLE_FFTW "Use FFTW3f when available" ON)

if(DTMB_CORE_ENABLE_CUDA_LDPC)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_library(dtmb_core
  src/c3780.cpp
  src/core.cpp
  src/fec.cpp
  src/frame_sync.cpp
  src/pn_channel.cpp
  src/resampler.cpp
)
add_library(dtmb::core ALIAS dtmb_core)
target_compile_features(dtmb_core PUBLIC cxx_std_20)
target_include_directories(dtmb_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(DTMB_CORE_ENABLE_FFTW)
  find_path(FFTW3F_INCLUDE_DIR NAMES fftw3.h HINTS ENV CONDA_PREFIX
    PATH_SUFFIXES include Library/include)
  find_library(FFTW3F_LIBRARY NAMES fftw3f libfftw3f-3 HINTS ENV CONDA_PREFIX
    PATH_SUFFIXES lib Library/lib)
  if(FFTW3F_INCLUDE_DIR AND FFTW3F_LIBRARY)
    target_include_directories(dtmb_core PRIVATE "${FFTW3F_INCLUDE_DIR}")
    target_link_libraries(dtmb_core PUBLIC "${FFTW3F_LIBRARY}")
    target_compile_definitions(dtmb_core PRIVATE DTMB_CORE_HAVE_FFTW3F=1)
  endif()
endif()

if(MSVC)
  target_compile_options(dtmb_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(dtmb_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(dtmb_core_ci8_resample tools/ci8_resample.cpp)
target_link_libraries(dtmb_core_ci8_resample PRIVATE dtmb::core)

add_executable(dtmb_core_c3780_extract tools/c3780_extract.cpp)
target_link_libraries(dtmb_core_c3780_extract PRIVATE dtmb::core)

add_executable(dtmb_core_deinterleave_qam64 tools/deinterleave_qam64.cpp)
target_link_libraries(dtmb_core_deinterleave_qam64 PRIVATE dtmb::core)

add_library(dtmb_ldpc_h_gate_support STATIC tools/ldpc_h_gate.cpp)
target_compile_features(dtmb_ldpc_h_gate_support PUBLIC cxx_std_20)
target_include_directories(dtmb_ldpc_h_gate_support PUBLIC tools)
target_link_libraries(dtmb_ldpc_h_gate_support PUBLIC dtmb::core)

if(DTMB_CORE_ENABLE_CUDA_LDPC)
  add_library(dtmb_ldpc_cuda_backend STATIC tools/ldpc_cuda_backend.cu)
  target_compile_definitions(dtmb_ldpc_cuda_backend PUBLIC DTMB_CORE_HAVE_CUDA_LDPC=1)
  target_link_libraries(dtmb_ldpc_cuda_backend PUBLIC dtmb::core CUDA::cudart)
else()
  add_library(dtmb_ldpc_cuda_backend STATIC tools/ldpc_cuda_backend.cpp)
  target_link_libraries(dtmb_ldpc_cuda_backend PUBLIC dtmb::core)
endif()
target_compile_features(dtmb_ldpc_cuda_backend PUBLIC cxx_std_20)
target_include_directories(dtmb_ldpc_cuda_backend PUBLIC tools)

add_executable(dtmb_core_ldpc_bch_decode tools/ldpc_bch_decode.cpp)
target_link_libraries(dtmb_core_ldpc_bch_decode PRIVATE
  dtmb_ldpc_h_gate_support
  dtmb_ldpc_cuda_backend
)

if(DTMB_CORE_BUILD_TESTS)
  enable_testing()
  add_executable(dtmb_ldpc_h_gate_tests tests/test_ldpc_h_gate.cpp)
  target_link_libraries(dtmb_ldpc_h_gate_tests PRIVATE dtmb_ldpc_h_gate_support)
  target_compile_definitions(dtmb_ldpc_h_gate_tests PRIVATE
    DTMB_LDPC_RATE2_ALIST="${CMAKE_CURRENT_SOURCE_DIR}/../../python/dtmb/data/dtmb_ldpc_rate2.alist"
  )
  add_test(NAME dtmb_ldpc_h_gate_tests COMMAND dtmb_ldpc_h_gate_tests)
endif()
