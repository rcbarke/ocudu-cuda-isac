# Sprint 2.3 (grid_apply): channel application for receiver replay.
#
# This directory builds in two ways.
#   1. From native/ocudu_adapter/CMakeLists.txt, through scripts/native_adapter.py: the host-free
#      core, its unit target and grid_apply, which links the core and the accepted OCUDU build.
#   2. On its own, with a C++17 compiler and no OCUDU source or build: the core and its unit target.
#      tests/unit/test_grid_apply.py builds it that way.
cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(replay_core LANGUAGES CXX)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Build the replay core out of tree.")
  endif()
  enable_testing()
endif()

# The core includes nothing of OCUDU and links nothing. Its flags are its own: the arithmetic is
# fixed by the source, and no claim is made about its speed.
add_library(replay_core STATIC replay_core.cpp)
target_include_directories(replay_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(replay_core PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_options(replay_core PRIVATE
  -O2 -ffp-contract=off -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)

# Unit target: the core and doubles only. One CTest case per unit case.
add_executable(replay_core_test tests/replay_core_test.cpp)
target_link_libraries(replay_core_test PRIVATE replay_core)
set_target_properties(replay_core_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(replay_core_test PRIVATE -O1 -Wall -Wextra -Wshadow)
set(REPLAY_CORE_CASES sha256 storage s03_vector unity zero_channel cancellation permutation unit_scale
    gain_reason gain_rules grid_states evidence session changed_grant refusals)
foreach(case IN LISTS REPLAY_CORE_CASES)
  add_test(NAME replay_core_${case} COMMAND replay_core_test ${case})
endforeach()

# The tool needs the pinned resource grid, so it exists only in the adapter build.
if(TARGET ocudu_pin_fixture)
  add_executable(grid_apply grid_apply.cpp)
  target_link_libraries(grid_apply PRIVATE replay_core ocudu_pin_fixture)

  add_test(NAME grid_apply_help COMMAND grid_apply --help)
  set_tests_properties(grid_apply_help PROPERTIES PASS_REGULAR_EXPRESSION "--plan FILE --output-dir DIR")
  add_test(NAME grid_apply_refuses_missing_arguments COMMAND grid_apply --no-prestore)
  set_tests_properties(grid_apply_refuses_missing_arguments PROPERTIES WILL_FAIL TRUE)
  # The unit vectors through the upstream writer and reader.
  add_test(NAME grid_apply_pinned_grid COMMAND grid_apply --self-test)
  set_tests_properties(grid_apply_pinned_grid PROPERTIES
    PASS_REGULAR_EXPRESSION "grid_apply self-test: all checks passed" FAIL_REGULAR_EXPRESSION "FAIL  ")
endif()
