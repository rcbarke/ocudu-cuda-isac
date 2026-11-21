# Out-of-tree build of the project's native adapter against the accepted OCUDU build.
#
# Configure through scripts/native_adapter.py. It reads the compile flags and the link line of
# pusch_e2e_pipeline_test in the accepted Sprint 1.1 build and writes them to the import file named
# by OCUDU_PIN_IMPORT. Adapter targets then compile and link exactly like that test, so they use the
# same OCUDU objects. Nothing here writes into the OCUDU source or build directory.
cmake_minimum_required(VERSION 3.18)

if(NOT OCUDU_PIN_IMPORT OR NOT EXISTS "${OCUDU_PIN_IMPORT}")
  message(FATAL_ERROR "OCUDU_PIN_IMPORT does not name an import file. Run: python scripts/native_adapter.py build")
endif()
include("${OCUDU_PIN_IMPORT}")

# The compiler that built the pinned receiver, chosen before the project enables a language.
set(CMAKE_CXX_COMPILER "${OCUDU_PIN_CXX_COMPILER}" CACHE FILEPATH "Compiler of the accepted OCUDU build" FORCE)
# Every flag comes from the pinned test. A build type would add a second optimization level.
set(CMAKE_BUILD_TYPE "" CACHE STRING "Flags come from the accepted OCUDU build" FORCE)

project(ocudu_adapter LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Build the adapter out of tree.")
endif()

# Everything pusch_e2e_pipeline_test compiles and links with. Static archives sit in one group, so
# their order and their mutual references need no maintenance here.
add_library(ocudu_pin_fixture INTERFACE)
target_compile_options(ocudu_pin_fixture INTERFACE ${OCUDU_PIN_CXX_FLAGS})
target_include_directories(ocudu_pin_fixture INTERFACE ${OCUDU_PIN_INCLUDES} "${OCUDU_PIN_TEST_SOURCE_DIR}")
if(OCUDU_PIN_CXX_DEFINES)
  target_compile_definitions(ocudu_pin_fixture INTERFACE ${OCUDU_PIN_CXX_DEFINES})
endif()
if(OCUDU_PIN_SYSTEM_INCLUDES)
  target_include_directories(ocudu_pin_fixture SYSTEM INTERFACE ${OCUDU_PIN_SYSTEM_INCLUDES})
endif()
if(OCUDU_PIN_LINK_OPTIONS)
  target_link_options(ocudu_pin_fixture INTERFACE ${OCUDU_PIN_LINK_OPTIONS})
endif()
target_link_libraries(ocudu_pin_fixture INTERFACE
  -Wl,--start-group ${OCUDU_PIN_ARCHIVES} -Wl,--end-group ${OCUDU_PIN_LINK_ITEMS})

enable_testing()
add_subdirectory(fixtures)
add_subdirectory(replay)
add_subdirectory(receiver)
