# Sprint 4.1 (mps_pacing): paced release of receiver jobs on the staged harness.
#
# This directory builds in two ways.
#   1. From native/ocudu_adapter/CMakeLists.txt, through scripts/native_adapter.py: the host-free
#      pacing core and its unit target. pilot_ls of observer/ links the core.
#   2. On its own, with a C++17 compiler and no OCUDU source or build: the same, with the cores
#      of replay/ and receiver/ compiled from their sources. tests/unit/test_mps_pacing.py builds
#      it that way.
cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(pacing_core LANGUAGES CXX)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Build the pacing core out of tree.")
  endif()
  enable_testing()
endif()

# Inside the adapter build the two cores are targets already. On its own this directory compiles
# the same sources with the same flags.
find_package(Threads REQUIRED)
if(NOT TARGET replay_core)
  add_library(replay_core STATIC "${CMAKE_CURRENT_SOURCE_DIR}/../replay/replay_core.cpp")
  target_include_directories(replay_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../replay")
  set_target_properties(replay_core PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(replay_core PRIVATE
    -O2 -ffp-contract=off -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)
endif()
if(NOT TARGET dispatch_core)
  add_library(dispatch_core STATIC "${CMAKE_CURRENT_SOURCE_DIR}/../receiver/dispatch_core.cpp")
  target_include_directories(dispatch_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../receiver")
  target_link_libraries(dispatch_core PUBLIC replay_core Threads::Threads)
  set_target_properties(dispatch_core PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(dispatch_core PRIVATE -O2 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)
endif()

# The pacing core includes the dispatch core and the channel application core and nothing of
# OCUDU. It reads the outcome rule and the job record of the dispatch core and leaves that core
# as it was accepted.
add_library(pacing_core STATIC pacing_core.cpp)
target_include_directories(pacing_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(pacing_core PUBLIC dispatch_core replay_core Threads::Threads)
set_target_properties(pacing_core PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_options(pacing_core PRIVATE -O2 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)

# Unit target: the three cores, doubles and the counting allocator of the ring's producer glue.
# One CTest case per unit case.
add_executable(pacing_core_test tests/pacing_core_test.cpp
  "${CMAKE_CURRENT_SOURCE_DIR}/../transport/tests/alloc_count.cpp")
target_link_libraries(pacing_core_test PRIVATE pacing_core)
set_target_properties(pacing_core_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(pacing_core_test PRIVATE -O1 -Wall -Wextra -Wshadow)
set(PACING_CORE_CASES schedule_rules releases_never_shift late_start book_decoded book_outcomes book_refusals
    book_no_allocation lead_in_rules book_lead_in lead_in_takes_the_cold_start spin_until)
foreach(case IN LISTS PACING_CORE_CASES)
  add_test(NAME pacing_core_${case} COMMAND pacing_core_test ${case})
endforeach()
