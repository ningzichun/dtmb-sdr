cmake_minimum_required(VERSION 3.20)

project(dtmb_core VERSION 0.2.0 LANGUAGES C CXX)

option(DTMB_CORE_BUILD_TESTS "Build dtmb-core C++ tests" ON)
option(DTMB_CORE_ENABLE_HACKRF "Link native live monitor against the HackRF submodule when libusb is available" ON)

add_library(dtmb_core
  src/c3780.cpp
  src/core.cpp
  src/fec.cpp
  src/frame_sync.cpp
  src/live_view.cpp
  src/pn_channel.cpp
  src/resampler.cpp
)
add_library(dtmb::core ALIAS dtmb_core)

target_compile_features(dtmb_core PUBLIC cxx_std_20)
target_include_directories(dtmb_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(MSVC)
  target_compile_options(dtmb_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(dtmb_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(dtmb_core_ci8_stats
  tools/ci8_stats.cpp
)
target_link_libraries(dtmb_core_ci8_stats PRIVATE dtmb::core)

add_executable(dtmb_core_pipe_buffer
  tools/pipe_buffer.cpp
)

add_executable(dtmb_core_live_monitor
  tools/live_monitor.cpp
)
target_link_libraries(dtmb_core_live_monitor PRIVATE dtmb::core)

add_executable(dtmb_core_hackrf_acquire
  tools/hackrf_acquire.cpp
)
target_link_libraries(dtmb_core_hackrf_acquire PRIVATE dtmb::core)

add_executable(dtmb_core_live_view_draw
  tools/live_view_draw.cpp
)
target_link_libraries(dtmb_core_live_view_draw PRIVATE dtmb::core)

add_executable(dtmb_core_live_view_scene
  tools/live_view_scene.cpp
)
target_link_libraries(dtmb_core_live_view_scene PRIVATE dtmb::core)

add_library(dtmb_live_view_host_renderer STATIC
  tools/live_view_host_renderer.cpp
)
target_compile_features(dtmb_live_view_host_renderer PUBLIC cxx_std_20)
target_include_directories(dtmb_live_view_host_renderer PUBLIC tools)

if(MSVC)
  target_compile_options(dtmb_live_view_host_renderer PRIVATE /W4 /permissive-)
else()
  target_compile_options(dtmb_live_view_host_renderer PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(dtmb_core_live_view_host
  tools/live_view_host.cpp
)
target_link_libraries(dtmb_core_live_view_host PRIVATE dtmb_live_view_host_renderer)

set(DTMB_HACKRF_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../hackrf/host/libhackrf/src")
set(DTMB_HACKRF_CMAKE_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../hackrf/host/cmake/modules")
if(DTMB_CORE_ENABLE_HACKRF AND EXISTS "${DTMB_HACKRF_LIB_DIR}/CMakeLists.txt")
  list(APPEND CMAKE_MODULE_PATH "${DTMB_HACKRF_CMAKE_MODULE_DIR}")
  find_package(Threads QUIET)
  find_package(LIBUSB QUIET)
  if(Threads_FOUND AND LIBUSB_FOUND)
    set(ENABLE_SHARED_LIB ON CACHE BOOL "Build shared libhackrf for gr-dtmb" FORCE)
    set(ENABLE_STATIC_LIB ON CACHE BOOL "Build static libhackrf for gr-dtmb" FORCE)
    set(RELEASE "gr-dtmb-submodule" CACHE STRING "libhackrf release string" FORCE)
    add_subdirectory(
      "${DTMB_HACKRF_LIB_DIR}"
      "${CMAKE_CURRENT_BINARY_DIR}/hackrf-libhackrf"
      EXCLUDE_FROM_ALL
    )
    target_compile_definitions(dtmb_core_live_monitor PRIVATE DTMB_CORE_HAVE_HACKRF=1)
    target_link_libraries(dtmb_core_live_monitor PRIVATE hackrf_static)
    target_compile_definitions(dtmb_core_hackrf_acquire PRIVATE DTMB_CORE_HAVE_HACKRF=1)
    target_link_libraries(dtmb_core_hackrf_acquire PRIVATE hackrf_static)
    message(STATUS "dtmb_core_live_monitor: direct HackRF capture enabled")
    message(STATUS "dtmb_core_hackrf_acquire: direct HackRF capture enabled")
  else()
    message(STATUS "dtmb_core_live_monitor: direct HackRF capture disabled; libusb/threads not found")
    message(STATUS "dtmb_core_hackrf_acquire: direct HackRF capture disabled; libusb/threads not found")
  endif()
else()
  message(STATUS "dtmb_core_live_monitor: direct HackRF capture disabled")
endif()

add_executable(dtmb_core_ci8_resample
  tools/ci8_resample.cpp
)
target_link_libraries(dtmb_core_ci8_resample PRIVATE dtmb::core)

add_executable(dtmb_core_qam64_softdemap
  tools/qam64_softdemap.cpp
)
target_link_libraries(dtmb_core_qam64_softdemap PRIVATE dtmb::core)

add_executable(dtmb_core_deinterleave_qam64
  tools/deinterleave_qam64.cpp
)
target_link_libraries(dtmb_core_deinterleave_qam64 PRIVATE dtmb::core)

add_executable(dtmb_core_c3780_extract
  tools/c3780_extract.cpp
)
target_link_libraries(dtmb_core_c3780_extract PRIVATE dtmb::core)

add_executable(dtmb_core_streaming_receiver
  tools/dtmb_streaming_receiver.cpp
)
target_link_libraries(dtmb_core_streaming_receiver PRIVATE dtmb::core)

add_executable(dtmb_core_ldpc_bch_decode
  tools/ldpc_bch_decode.cpp
)

add_executable(dtmb_core_ldpc_h_score
  tools/ldpc_h_score.cpp
)
target_link_libraries(dtmb_core_ldpc_h_score PRIVATE dtmb::core)

add_library(dtmb_ldpc_h_gate_support STATIC
  tools/ldpc_h_gate.cpp
)
target_compile_features(dtmb_ldpc_h_gate_support PUBLIC cxx_std_20)
target_include_directories(dtmb_ldpc_h_gate_support PUBLIC tools)
target_link_libraries(dtmb_ldpc_h_gate_support PUBLIC dtmb::core)

if(MSVC)
  target_compile_options(dtmb_ldpc_h_gate_support PRIVATE /W4 /permissive-)
else()
  target_compile_options(dtmb_ldpc_h_gate_support PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_link_libraries(dtmb_core_ldpc_bch_decode PRIVATE dtmb_ldpc_h_gate_support)

add_executable(dtmb_core_ldpc_h_gate
  tools/ldpc_h_gate_main.cpp
)
target_link_libraries(dtmb_core_ldpc_h_gate PRIVATE dtmb_ldpc_h_gate_support)

add_executable(dtmb_core_llr_frame_select
  tools/llr_frame_select.cpp
)
target_link_libraries(dtmb_core_llr_frame_select PRIVATE dtmb_ldpc_h_gate_support)

add_executable(dtmb_core_pre_ldpc_source_score
  tools/pre_ldpc_source_score.cpp
)
target_link_libraries(dtmb_core_pre_ldpc_source_score PRIVATE dtmb_ldpc_h_gate_support)

if(DTMB_CORE_BUILD_TESTS)
  enable_testing()

  add_executable(dtmb_core_tests
    tests/test_core.cpp
  )
  target_link_libraries(dtmb_core_tests PRIVATE dtmb::core)

  add_test(NAME dtmb_core_tests COMMAND dtmb_core_tests)

  add_executable(dtmb_live_view_host_tests
    tests/test_live_view_host.cpp
  )
  target_link_libraries(dtmb_live_view_host_tests PRIVATE dtmb_live_view_host_renderer)
  target_compile_definitions(dtmb_live_view_host_tests PRIVATE
    DTMB_LIVE_VIEW_HOST_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/live_view_host_scene.ndjson"
  )

  add_test(NAME dtmb_live_view_host_tests COMMAND dtmb_live_view_host_tests)

  add_executable(dtmb_ldpc_h_gate_tests
    tests/test_ldpc_h_gate.cpp
  )
  target_link_libraries(dtmb_ldpc_h_gate_tests PRIVATE dtmb_ldpc_h_gate_support)
  target_compile_definitions(dtmb_ldpc_h_gate_tests PRIVATE
    DTMB_LDPC_RATE2_ALIST="${CMAKE_CURRENT_SOURCE_DIR}/../../python/dtmb/data/dtmb_ldpc_rate2.alist"
  )

  add_test(NAME dtmb_ldpc_h_gate_tests COMMAND dtmb_ldpc_h_gate_tests)
endif()
