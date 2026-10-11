add_library(dtmb_ldpc_cuda_backend STATIC tools/ldpc_cuda_backend.cpp)
target_compile_features(dtmb_ldpc_cuda_backend PUBLIC cxx_std_20)
target_include_directories(dtmb_ldpc_cuda_backend PUBLIC tools)
target_link_libraries(dtmb_ldpc_cuda_backend PUBLIC dtmb::core ${CMAKE_DL_LIBS})

if(DTMB_CORE_ENABLE_CUDA_LDPC)
  find_package(CUDAToolkit REQUIRED)
  if(CUDAToolkit_VERSION VERSION_LESS 12)
    message(FATAL_ERROR "The packaged backend requires CUDA Toolkit 12.x or newer")
  endif()
  string(REGEX MATCH "^[0-9]+" DTMB_CUDA_MAJOR "${CUDAToolkit_VERSION}")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(DTMB_CUDA_MAJOR GREATER_EQUAL 13)
      # CUDA 13 removed Maxwell/Pascal/Volta support.
      set(CMAKE_CUDA_ARCHITECTURES "80-real;86-real;89-real;90-real;100-real;120-real;80-virtual")
    else()
      set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;75-virtual")
    endif()
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set_property(TARGET dtmb_core PROPERTY POSITION_INDEPENDENT_CODE ON)
  add_library(dtmb_cuda${DTMB_CUDA_MAJOR} SHARED tools/ldpc_cuda_backend.cu tools/ldpc_cuda_plugin.cpp)
  target_compile_features(dtmb_cuda${DTMB_CUDA_MAJOR} PRIVATE cxx_std_20)
  target_link_libraries(dtmb_cuda${DTMB_CUDA_MAJOR} PRIVATE dtmb::core CUDA::cudart)
  set_target_properties(dtmb_cuda${DTMB_CUDA_MAJOR} PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH "$ORIGIN")
  if(MSVC)
    target_compile_options(dtmb_cuda${DTMB_CUDA_MAJOR} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  endif()
  install(TARGETS dtmb_cuda${DTMB_CUDA_MAJOR} RUNTIME DESTINATION dtmb/bin LIBRARY DESTINATION dtmb/bin)
endif()

if(DTMB_CORE_BUILD_TESTS AND NOT EMSCRIPTEN)
  enable_testing()
  add_library(dtmb_cuda_test_backend SHARED tests/cuda_test_backend.cpp)
  target_include_directories(dtmb_cuda_test_backend PRIVATE tools)
  add_library(dtmb_cuda_bad_abi SHARED tests/cuda_test_backend.cpp)
  target_include_directories(dtmb_cuda_bad_abi PRIVATE tools)
  target_compile_definitions(dtmb_cuda_bad_abi PRIVATE DTMB_TEST_BAD_ABI=1)
  add_executable(dtmb_cuda_loader_tests tests/test_cuda_loader.cpp)
  target_link_libraries(dtmb_cuda_loader_tests PRIVATE dtmb_ldpc_cuda_backend)
  add_test(NAME dtmb_cuda_loader_batch COMMAND dtmb_cuda_loader_tests "$<TARGET_FILE:dtmb_cuda_test_backend>" batch)
  add_test(NAME dtmb_cuda_loader_abi COMMAND dtmb_cuda_loader_tests "$<TARGET_FILE:dtmb_cuda_bad_abi>" abi)
  add_test(NAME dtmb_cuda_loader_missing COMMAND dtmb_cuda_loader_tests "$<TARGET_FILE:dtmb_cuda_test_backend>.missing" missing)
  foreach(major IN ITEMS 12 13)
    add_library(dtmb_cuda_runtime_test${major} SHARED tests/cuda_test_runtime.cpp)
    target_include_directories(dtmb_cuda_runtime_test${major} PRIVATE tools)
    target_compile_definitions(dtmb_cuda_runtime_test${major} PRIVATE DTMB_TEST_RUNTIME_MAJOR=${major})
    set_target_properties(dtmb_cuda_runtime_test${major} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cuda-runtime-tests"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cuda-runtime-tests")
    if(WIN32)
      set_target_properties(dtmb_cuda_runtime_test${major} PROPERTIES OUTPUT_NAME cudart64_${major})
    elseif(APPLE)
      set_target_properties(dtmb_cuda_runtime_test${major} PROPERTIES OUTPUT_NAME cudart SUFFIX .${major}.dylib)
    else()
      set_target_properties(dtmb_cuda_runtime_test${major} PROPERTIES OUTPUT_NAME cudart SUFFIX .so.${major})
    endif()
    add_library(dtmb_cuda_major_test${major} SHARED tests/cuda_test_backend.cpp)
    target_include_directories(dtmb_cuda_major_test${major} PRIVATE tools)
    target_compile_definitions(dtmb_cuda_major_test${major} PRIVATE DTMB_TEST_RUNTIME_MAJOR=${major})
    target_link_libraries(dtmb_cuda_major_test${major} PRIVATE dtmb_cuda_runtime_test${major})
    set_target_properties(dtmb_cuda_major_test${major} PROPERTIES
      OUTPUT_NAME dtmb_cuda${major}
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cuda-backend-tests"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cuda-backend-tests")
    add_test(NAME dtmb_cuda_loader_runtime${major} COMMAND dtmb_cuda_loader_tests
      "$<TARGET_FILE:dtmb_cuda_major_test${major}>" runtime "$<TARGET_FILE_DIR:dtmb_cuda_runtime_test${major}>")
  endforeach()
endif()
