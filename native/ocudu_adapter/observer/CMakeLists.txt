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
    record ownership timed binding slot_times)
foreach(case IN LISTS OBSERVER_CORE_CASES)
  add_test(NAME observer_core_${case} COMMAND observer_core_test ${case})
endforeach()

# The tool needs the pinned PUSCH processors, so it exists only in the adapter build.
if(TARGET ocudu_pin_fixture)
  # The tool is compiled exactly like replay_pusch, with the flags of the pinned test and no other.
  # It computes no estimate itself: observer::observe does, inside the core and with its flags.
  # Sprint 3.2: it also links the producer glue of transport/ and, through it, the observation ring.
  add_executable(pilot_ls pilot_ls.cpp)
  target_link_libraries(pilot_ls PRIVATE obs_publisher observer_core dispatch_core replay_core ocudu_pin_fixture)
  # Sprint 5.2: the host-free capture core of estimate/. Every build links it and reads a plan of
  # version 3 with it. The device side, estimate_tap.h, is compiled only with ISAC_O3_ROUTE.
  target_link_libraries(pilot_ls PRIVATE estimate_core)
  if(ISAC_O3_ROUTE)
    if(NOT "ENABLE_CUDA" IN_LIST OCUDU_PIN_CXX_DEFINES)
      message(FATAL_ERROR "ISAC_O3_ROUTE needs an OCUDU build with ENABLE_CUDA: the estimate is produced on the device.")
    endif()
    set(ISAC_O3_TAP_HEADER "ocudu/phy/upper/channel_processors/pusch/pusch_resident_estimate_tap.h")
    set(ISAC_O3_TAP_FOUND OFF)
    foreach(directory IN LISTS OCUDU_PIN_INCLUDES)
      if(EXISTS "${directory}/${ISAC_O3_TAP_HEADER}")
        set(ISAC_O3_TAP_FOUND ON)
      endif()
    endforeach()
    if(NOT ISAC_O3_TAP_FOUND)
      message(FATAL_ERROR "ISAC_O3_ROUTE needs the patched receiver: no include directory of the imported OCUDU build "
                          "holds ${ISAC_O3_TAP_HEADER}. Build it with: python scripts/native_ocudu.py build --patched")
    endif()
    # The CUDA runtime header for the copy and its event. The runtime library is on the link line
    # of the imported build already, which links it for the kernels of the receiver.
    find_package(CUDAToolkit REQUIRED)
    target_include_directories(pilot_ls SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(pilot_ls PRIVATE ISAC_O3_ROUTE)
    set(ISAC_O3_RUNTIME_LINKED OFF)
    foreach(item IN LISTS OCUDU_PIN_LINK_ITEMS OCUDU_PIN_ARCHIVES)
      if(item MATCHES "cudart")
        set(ISAC_O3_RUNTIME_LINKED ON)
      endif()
    endforeach()
    if(NOT ISAC_O3_RUNTIME_LINKED)
      target_link_libraries(pilot_ls PRIVATE CUDA::cudart)
    endif()
  endif()

  add_test(NAME pilot_ls_help COMMAND pilot_ls --help)
  set_tests_properties(pilot_ls_help PROPERTIES PASS_REGULAR_EXPRESSION "--plan FILE --output-dir DIR")
  add_test(NAME pilot_ls_refuses_missing_arguments COMMAND pilot_ls --no-grids)
  set_tests_properties(pilot_ls_refuses_missing_arguments PROPERTIES WILL_FAIL TRUE)
  # Which receiver the tool is linked against. With the route it calls the accessor of the
  # receiver patch, so the test also shows that the patched objects are the linked ones. It starts
  # no work on a device.
  add_test(NAME pilot_ls_o3_route COMMAND pilot_ls --route)
  if(ISAC_O3_ROUTE)
    set_tests_properties(pilot_ls_o3_route PROPERTIES
      PASS_REGULAR_EXPRESSION "pilot_ls route: O3 from the patched receiver, tap interface version 1, no observer registered")
  else()
    set_tests_properties(pilot_ls_o3_route PROPERTIES
      PASS_REGULAR_EXPRESSION "pilot_ls route: none, linked against the receiver without the patch")
  endif()
  # A known grant through the pinned grid and the pinned host processor, with the tap off and on,
  # and once more in the mode timed through a ring without a consumer.
  # It requests no CUDA processor, so the adapter build starts no work on a device.
  add_test(NAME pilot_ls_host_tap COMMAND pilot_ls --self-test)
  set_tests_properties(pilot_ls_host_tap PROPERTIES
    PASS_REGULAR_EXPRESSION "pilot_ls self-test: all checks passed" FAIL_REGULAR_EXPRESSION "FAIL  ")
endif()
