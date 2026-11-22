# Sprint 5.2 (native_estimate): capture of the receiver's own channel estimate, the stage O3.
#
# This directory builds in two ways.
#   1. From native/ocudu_adapter/CMakeLists.txt, through scripts/native_adapter.py: the host-free
#      capture core and its unit target. pilot_ls of observer/ links the core in every adapter
#      build and drives it only in the build against the patched receiver (ISAC_O3_ROUTE).
#   2. On its own, with a C++17 compiler and no OCUDU source, no OCUDU build and no CUDA: the
#      same, with the cores of replay/ and observer/, the producer glue of transport/ and the
#      ring of native/obs_ring/ compiled from their sources. tests/unit/test_native_estimate.py
#      builds it that way.
# The CUDA calls of the capture are not here. They are in observer/estimate_tap.h, behind the
# interface copier of the core.
cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(estimate_core LANGUAGES CXX)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Build the capture core out of tree.")
  endif()
  enable_testing()
endif()

# Inside the adapter build the cores, the glue and the ring are targets already. On its own this
# directory compiles the same sources with the same flags.
if(NOT TARGET replay_core)
  add_library(replay_core STATIC "${CMAKE_CURRENT_SOURCE_DIR}/../replay/replay_core.cpp")
  target_include_directories(replay_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../replay")
  set_target_properties(replay_core PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(replay_core PRIVATE
    -O2 -ffp-contract=off -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)
endif()
if(NOT TARGET observer_core)
  add_library(observer_core STATIC "${CMAKE_CURRENT_SOURCE_DIR}/../observer/observer_core.cpp")
  target_include_directories(observer_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../observer")
  target_link_libraries(observer_core PUBLIC replay_core)
  set_target_properties(observer_core PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(observer_core PRIVATE
    -O2 -ffp-contract=off -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)
endif()
if(NOT TARGET obs_ring)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../obs_ring" obs_ring)
endif()
if(NOT TARGET obs_publisher)
  add_library(obs_publisher STATIC "${CMAKE_CURRENT_SOURCE_DIR}/../transport/obs_publisher.cpp")
  target_include_directories(obs_publisher PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../transport")
  target_link_libraries(obs_publisher PUBLIC observer_core replay_core obs_ring)
  set_target_properties(obs_publisher PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(obs_publisher PRIVATE -O2 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)
endif()

# The core computes no estimate: the receiver does, on the device. It moves bytes, sets validity
# bits and keeps the account. It shares the host block of a staged record with the glue.
add_library(estimate_core STATIC estimate_core.cpp)
target_include_directories(estimate_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(estimate_core PUBLIC obs_publisher observer_core replay_core obs_ring)
set_target_properties(estimate_core PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_options(estimate_core PRIVATE
  -O2 -ffp-contract=off -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)

# Unit target: the core, the glue, the two cores, the ring and a fake asynchronous producer only.
# One CTest case per unit case.
add_executable(estimate_core_test tests/estimate_core_test.cpp
  "${CMAKE_CURRENT_SOURCE_DIR}/../transport/tests/alloc_count.cpp")
target_link_libraries(estimate_core_test PRIVATE estimate_core)
set_target_properties(estimate_core_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(estimate_core_test PRIVATE -O1 -ffp-contract=off -Wall -Wextra -Wshadow)
set(ESTIMATE_CORE_CASES half layout zero_and_absent refusals async_completion generation timed audit drop absent
    threads)
foreach(case IN LISTS ESTIMATE_CORE_CASES)
  add_test(NAME estimate_core_${case} COMMAND estimate_core_test ${case})
endforeach()
