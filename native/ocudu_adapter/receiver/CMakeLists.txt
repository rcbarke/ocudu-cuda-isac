# Sprint 2.4 (replay_pusch): dispatch of the receiver grid to the PUSCH processors.
#
# This directory builds in two ways.
#   1. From native/ocudu_adapter/CMakeLists.txt, through scripts/native_adapter.py: the host-free
#      dispatch core, its unit target and replay_pusch, which links the core, the channel
#      application core of replay/ and the accepted OCUDU build.
#   2. On its own, with a C++17 compiler and no OCUDU source or build: the dispatch core and its
#      unit target. tests/unit/test_replay_pusch.py builds it that way.
cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(dispatch_core LANGUAGES CXX)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Build the dispatch core out of tree.")
  endif()
  enable_testing()
endif()

# The channel application core of Sprint 2.3 owns the grids. Inside the adapter build it is the
# target of replay/. On its own this directory compiles the same source with the same flags.
if(NOT TARGET replay_core)
  add_library(replay_core STATIC "${CMAKE_CURRENT_SOURCE_DIR}/../replay/replay_core.cpp")
  target_include_directories(replay_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../replay")
  set_target_properties(replay_core PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(replay_core PRIVATE
    -O2 -ffp-contract=off -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)
endif()

# The dispatch core includes the channel application core and nothing of OCUDU. Completions
# arrive from the threads of a processor, so it needs the thread library.
find_package(Threads REQUIRED)
add_library(dispatch_core STATIC dispatch_core.cpp)
target_include_directories(dispatch_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(dispatch_core PUBLIC replay_core Threads::Threads)
set_target_properties(dispatch_core PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_options(dispatch_core PRIVATE -O2 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)

# Unit target: the two cores and doubles only. One CTest case per unit case.
add_executable(dispatch_core_test tests/dispatch_core_test.cpp)
target_link_libraries(dispatch_core_test PRIVATE dispatch_core)
set_target_properties(dispatch_core_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(dispatch_core_test PRIVATE -O1 -Wall -Wextra -Wshadow)
set(DISPATCH_CORE_CASES compare decoded valid_crc_wrong_byte crc_fail swapped_completions decoder_error
    duplicate_delivery silent_timeout owned_input changed_input timed guard accounting)
foreach(case IN LISTS DISPATCH_CORE_CASES)
  add_test(NAME dispatch_core_${case} COMMAND dispatch_core_test ${case})
endforeach()

# The tool needs the pinned PUSCH processors, so it exists only in the adapter build.
if(TARGET ocudu_pin_fixture)
  add_executable(replay_pusch replay_pusch.cpp)
  target_link_libraries(replay_pusch PRIVATE dispatch_core replay_core ocudu_pin_fixture)

  add_test(NAME replay_pusch_help COMMAND replay_pusch --help)
  set_tests_properties(replay_pusch_help PROPERTIES PASS_REGULAR_EXPRESSION "--plan FILE --output-dir DIR")
  add_test(NAME replay_pusch_refuses_missing_arguments COMMAND replay_pusch --no-grids)
  set_tests_properties(replay_pusch_refuses_missing_arguments PROPERTIES WILL_FAIL TRUE)
  # A known grant through the pinned host processor: the true callback, the comparison, the ownership.
  # It requests no CUDA processor, so the adapter build starts no work on a device.
  add_test(NAME replay_pusch_host_decode COMMAND replay_pusch --self-test)
  set_tests_properties(replay_pusch_host_decode PROPERTIES
    PASS_REGULAR_EXPRESSION "replay_pusch self-test: all checks passed" FAIL_REGULAR_EXPRESSION "FAIL  ")
endif()
