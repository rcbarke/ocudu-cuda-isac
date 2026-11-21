# Sprint 3.1 (pilot_ls): pilot least squares on the receiver grid, the stage O2.
#
# This directory builds in two ways.
#   1. From native/ocudu_adapter/CMakeLists.txt, through scripts/native_adapter.py: the host-free
#      observer core, its unit target and pilot_ls, which links the core, the dispatch core of
#      receiver/, the channel application core of replay/ and the accepted OCUDU build.
#   2. On its own, with a C++17 compiler and no OCUDU source or build: the observer core and its
#      unit target. tests/unit/test_pilot_ls.py builds it that way.
cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(observer_core LANGUAGES CXX)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Build the observer core out of tree.")
  endif()
  enable_testing()
endif()

# The channel application core of Sprint 2.3 owns the grids and their read access. Inside the
# adapter build it is the target of replay/. On its own this directory compiles the same source
# with the same flags.
if(NOT TARGET replay_core)
  add_library(replay_core STATIC "${CMAKE_CURRENT_SOURCE_DIR}/../replay/replay_core.cpp")
  target_include_directories(replay_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../replay")
  set_target_properties(replay_core PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(replay_core PRIVATE
    -O2 -ffp-contract=off -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)
endif()

# The observer core includes the channel application core and nothing of OCUDU. Its arithmetic
# is fixed by the source: no contraction and no value-changing optimization.
add_library(observer_core STATIC observer_core.cpp)
target_include_directories(observer_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(observer_core PUBLIC replay_core)
set_target_properties(observer_core PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_options(observer_core PRIVATE
  -O2 -ffp-contract=off -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)

# Unit target: the two cores and doubles only. One CTest case per unit case.
add_executable(observer_core_test tests/observer_core_test.cpp)
target_link_libraries(observer_core_test PRIVATE observer_core)
set_target_properties(observer_core_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(observer_core_test PRIVATE -O1 -ffp-contract=off -Wall -Wextra -Wshadow)
set(OBSERVER_CORE_CASES kernel amplitude_scale data_invariance pilot_hole refusals antenna_map coordinates
    record ownership binding slot_times)
foreach(case IN LISTS OBSERVER_CORE_CASES)
  add_test(NAME observer_core_${case} COMMAND observer_core_test ${case})
endforeach()

# The tool needs the pinned PUSCH processors, so it exists only in the adapter build.
if(TARGET ocudu_pin_fixture)
  # The tool is compiled exactly like replay_pusch, with the flags of the pinned test and no other.
  # It computes no estimate itself: observer::observe does, inside the core and with its flags.
  add_executable(pilot_ls pilot_ls.cpp)
  target_link_libraries(pilot_ls PRIVATE observer_core dispatch_core replay_core ocudu_pin_fixture)

  add_test(NAME pilot_ls_help COMMAND pilot_ls --help)
  set_tests_properties(pilot_ls_help PROPERTIES PASS_REGULAR_EXPRESSION "--plan FILE --output-dir DIR")
  add_test(NAME pilot_ls_refuses_missing_arguments COMMAND pilot_ls --no-grids)
  set_tests_properties(pilot_ls_refuses_missing_arguments PROPERTIES WILL_FAIL TRUE)
  # A known grant through the pinned grid and the pinned host processor, with the tap off and on.
  # It requests no CUDA processor, so the adapter build starts no work on a device.
  add_test(NAME pilot_ls_host_tap COMMAND pilot_ls --self-test)
  set_tests_properties(pilot_ls_host_tap PROPERTIES
    PASS_REGULAR_EXPRESSION "pilot_ls self-test: all checks passed" FAIL_REGULAR_EXPRESSION "FAIL  ")
endif()
