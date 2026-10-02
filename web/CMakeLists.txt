cmake_minimum_required(VERSION 3.20)
project(dtmb_web LANGUAGES CXX)
if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Configure with emcmake; see web/README.md")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CORE "${CMAKE_CURRENT_SOURCE_DIR}/../core/cpp")
add_library(receiver STATIC
  ${CORE}/src/core.cpp ${CORE}/src/nr.cpp ${CORE}/src/iq.cpp ${CORE}/src/resampler.cpp
  ${CORE}/src/c1.cpp ${CORE}/src/c3780.cpp ${CORE}/src/frame_sync.cpp ${CORE}/src/pn_channel.cpp
  ${CORE}/src/fec.cpp ${CORE}/src/transport.cpp)
target_include_directories(receiver PUBLIC ${CORE}/include)
target_compile_options(receiver PRIVATE -O3 -fexceptions)
foreach(stage IN ITEMS ci8_resample c3780_extract deinterleave_qam64 ldpc_bch_decode)
  add_executable(${stage} ${CORE}/tools/${stage}.cpp)
  target_link_libraries(${stage} PRIVATE receiver)
  target_compile_options(${stage} PRIVATE -O3 -fexceptions)
  target_link_options(${stage} PRIVATE -O3 -fexceptions
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sINVOKE_RUN=0 -sEXIT_RUNTIME=1
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=1073741824
    -sSTACK_SIZE=2097152 -sASYNCIFY=1 -sASYNCIFY_STACK_SIZE=262144
    "-sEXPORTED_RUNTIME_METHODS=['callMain','FS','Asyncify']"
    -sENVIRONMENT=web,worker,node)
  set_target_properties(${stage} PROPERTIES SUFFIX ".mjs")
endforeach()
target_sources(ldpc_bch_decode PRIVATE ${CORE}/tools/ldpc_h_gate.cpp ${CORE}/tools/ldpc_cuda_backend.cpp)
