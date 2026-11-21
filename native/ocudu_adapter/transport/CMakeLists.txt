# Sprint 3.2 (snapshot_ring): producer glue between the observer and the observation ring.
#
# This directory builds in two ways.
#   1. From native/ocudu_adapter/CMakeLists.txt, through scripts/native_adapter.py: the glue and
#      its unit target, which pilot_ls of observer/ links.
#   2. On its own, with a C++17 compiler and no OCUDU source or build: the same, with the cores
#      of replay/ and observer/ and the ring of native/obs_ring/ compiled from their sources.
#      tests/unit/test_snapshot_ring.py builds it that way.
cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(obs_publisher LANGUAGES CXX)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Build the producer glue out of tree.")
  endif()
  enable_testing()
endif()

# Inside the adapter build the cores and the ring are targets already. On its own this directory
# compiles the same sources with the same flags.
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

# The glue computes no estimate: observer_core does, with its own flags.
add_library(obs_publisher STATIC obs_publisher.cpp)
target_include_directories(obs_publisher PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(obs_publisher PUBLIC observer_core replay_core obs_ring)
set_target_properties(obs_publisher PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_options(obs_publisher PRIVATE -O2 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)

# Unit target: the glue, the two cores, the ring and doubles only. One CTest case per unit case.
add_executable(obs_publisher_test tests/obs_publisher_test.cpp tests/alloc_count.cpp)
target_link_libraries(obs_publisher_test PRIVATE obs_publisher)
set_target_properties(obs_publisher_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(obs_publisher_test PRIVATE -O1 -Wall -Wextra -Wshadow)
set(OBS_PUBLISHER_CASES needs timed audit drop stale_grid refusals)
foreach(case IN LISTS OBS_PUBLISHER_CASES)
  add_test(NAME obs_publisher_${case} COMMAND obs_publisher_test ${case})
endforeach()
