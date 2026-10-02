cmake_minimum_required(VERSION 3.20)

if(CMAKE_HOST_APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum macOS version for the native receiver")
endif()
if(CMAKE_HOST_APPLE AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "10.15")
  message(FATAL_ERROR "The native receiver requires macOS 10.15 or later; set MACOSX_DEPLOYMENT_TARGET accordingly")
endif()

project(dtmb_core VERSION 0.4.1 LANGUAGES C CXX)

option(DTMB_CORE_BUILD_TESTS "Build dtmb-core tests" ON)
option(DTMB_CORE_ENABLE_CUDA_LDPC "Build the optional CUDA LDPC backend" OFF)
option(DTMB_CORE_ENABLE_FFTW "Use FFTW3f when available" ON)

find_package(Threads REQUIRED)


add_library(dtmb_core
  src/c1.cpp
  src/c3780.cpp
  src/core.cpp
  src/nr.cpp
  src/fec.cpp
  src/frame_sync.cpp
  src/pn_channel.cpp
  src/resampler.cpp
  src/iq.cpp
  src/transport.cpp
)
add_library(dtmb::core ALIAS dtmb_core)
target_compile_features(dtmb_core PUBLIC cxx_std_20)
target_include_directories(dtmb_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(dtmb_core PUBLIC Threads::Threads)

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

# Both command names share the same streaming implementation.
add_executable(dtmb_core_deinterleave_qam tools/deinterleave_qam64.cpp)
target_link_libraries(dtmb_core_deinterleave_qam PRIVATE dtmb::core)

add_executable(dtmb_core_qam_softdemap tools/qam64_softdemap.cpp)
target_link_libraries(dtmb_core_qam_softdemap PRIVATE dtmb::core)

add_executable(dtmb_core_pipe_buffer tools/pipe_buffer.cpp)
target_compile_features(dtmb_core_pipe_buffer PRIVATE cxx_std_20)
target_link_libraries(dtmb_core_pipe_buffer PRIVATE Threads::Threads)

add_library(dtmb_ldpc_h_gate_support STATIC tools/ldpc_h_gate.cpp)
target_compile_features(dtmb_ldpc_h_gate_support PUBLIC cxx_std_20)
target_include_directories(dtmb_ldpc_h_gate_support PUBLIC tools)
target_link_libraries(dtmb_ldpc_h_gate_support PUBLIC dtmb::core)

include(cmake/CudaBackend.cmake)

add_executable(dtmb_core_ldpc_bch_decode tools/ldpc_bch_decode.cpp)
target_link_libraries(dtmb_core_ldpc_bch_decode PRIVATE
  dtmb_ldpc_h_gate_support
  dtmb_ldpc_cuda_backend
)

install(TARGETS
  dtmb_core_ci8_resample
  dtmb_core_c3780_extract
  dtmb_core_deinterleave_qam64
  dtmb_core_deinterleave_qam
  dtmb_core_qam_softdemap
  dtmb_core_ldpc_bch_decode
  dtmb_core_pipe_buffer
  RUNTIME DESTINATION dtmb/bin
)

if(DTMB_CORE_BUILD_TESTS)
  enable_testing()
  add_executable(dtmb_c1_tests tests/test_c1.cpp)
  target_link_libraries(dtmb_c1_tests PRIVATE dtmb::core)
  add_test(NAME dtmb_c1_tests COMMAND dtmb_c1_tests)
  add_executable(dtmb_nr_tests tests/test_nr.cpp)
  target_link_libraries(dtmb_nr_tests PRIVATE dtmb::core)
  add_test(NAME dtmb_nr_tests COMMAND dtmb_nr_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/nr_appendix_c.txt")
  add_executable(dtmb_pn_tests tests/test_pn.cpp)
  target_link_libraries(dtmb_pn_tests PRIVATE dtmb::core)
  add_test(NAME dtmb_pn_tests COMMAND dtmb_pn_tests)

  add_executable(dtmb_qam_tests tests/test_qam.cpp)
  target_link_libraries(dtmb_qam_tests PRIVATE dtmb::core)
  add_test(NAME dtmb_qam_tests COMMAND dtmb_qam_tests)
  add_executable(dtmb_ldpc_h_gate_tests tests/test_ldpc_h_gate.cpp)
  target_link_libraries(dtmb_ldpc_h_gate_tests PRIVATE dtmb_ldpc_h_gate_support)
  target_compile_definitions(dtmb_ldpc_h_gate_tests PRIVATE
    DTMB_LDPC_RATE2_ALIST="${CMAKE_CURRENT_SOURCE_DIR}/../../python/dtmb/data/dtmb_ldpc_rate2.alist"
  )
  add_test(NAME dtmb_ldpc_h_gate_tests COMMAND dtmb_ldpc_h_gate_tests)
endif()
