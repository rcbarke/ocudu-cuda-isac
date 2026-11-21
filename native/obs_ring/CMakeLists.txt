# Sprint 3.2 (snapshot_ring): the observation record obs/v1 and the ring obs-ring/v1.
#
# This directory builds in two ways.
#   1. On its own, with a C++17 compiler and nothing else: the library, its unit target and the
#      two tools. tests/unit/test_snapshot_ring.py builds it that way, and so does a sensor that
#      links the consumer without the receiver build.
#   2. From native/ocudu_adapter/CMakeLists.txt, through scripts/native_adapter.py: the same
#      targets, so that the producer glue of the adapter links the library it was tested with.
# Nothing here includes OCUDU or CUDA.
cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(obs_ring LANGUAGES CXX)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Build the observation ring out of tree.")
  endif()
  enable_testing()
endif()

find_package(Threads REQUIRED)
add_library(obs_ring STATIC src/ring.cpp)
target_include_directories(obs_ring PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
# shm_open is in librt on older C libraries and in the C library itself on newer ones.
target_link_libraries(obs_ring PUBLIC Threads::Threads rt)
set_target_properties(obs_ring PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_options(obs_ring PRIVATE -O2 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)

# Unit target: the library only. One CTest case per unit case.
add_executable(obs_ring_test tests/obs_ring_test.cpp)
target_link_libraries(obs_ring_test PRIVATE obs_ring)
set_target_properties(obs_ring_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(obs_ring_test PRIVATE -O1 -Wall -Wextra -Wshadow)
set(OBS_RING_CASES definition capacity_two scratch_overwrite no_early_read stale_read double_release
    released_fault fifo spin_no_syscalls wait_flag wait_no_loss consumer_exit definition_refused read_only
    busy_and_drain long_path)
foreach(case IN LISTS OBS_RING_CASES)
  add_test(NAME obs_ring_${case} COMMAND obs_ring_test ${case})
endforeach()

# The consumer that keeps what it read, and the producer of known records.
foreach(tool obs_tail obs_feed)
  add_executable(${tool} tools/${tool}.cpp)
  target_link_libraries(${tool} PRIVATE obs_ring)
  set_target_properties(${tool} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  target_compile_options(${tool} PRIVATE -O2 -Wall -Wextra -Wshadow)
  add_test(NAME ${tool}_help COMMAND ${tool} --help)
  set_tests_properties(${tool}_help PROPERTIES PASS_REGULAR_EXPRESSION "--socket PATH")
endforeach()
