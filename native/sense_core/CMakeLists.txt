# Sprint 3.6 (sense_core): the native sensing core.
#
# This is a project of its own. It takes nothing from the OCUDU build: no include directory, no
# archive, no flag. It compiles the observation ring of native/obs_ring from its sources, for the
# record definition and for the consumer that sense_host links, and that is all it shares with the
# receiver side. scripts/native_core.py configures, builds and tests it out of tree and writes the
# receipt isac-core.json.
#
#   cmake -S native/sense_core -B build/sense-core                    the core with its CUDA device
#   cmake -S native/sense_core -B build/sense-core-host -DSENSE_CORE_HOST_ONLY=ON
#                                                                     the host twin only: no CUDA
#                                                                     compiler, for a development
#                                                                     host and the unit gate
#
# Options
#   SENSE_CORE_HOST_ONLY   ON: C and C++ only. The kernels run as their host twin.
#   SENSE_CORE_TENSORRT    AUTO (default), ON or OFF. TensorRT is an optional component: the
#                          kernels and the twin build without it. TENSORRT_ROOT, or
#                          TENSORRT_INCLUDE_DIR and TENSORRT_LIBRARY_DIR, name where it is.
#   SENSE_CORE_LIBTORCH    OFF (default) or ON: the declared fallback backend, through the C++
#                          library of the PyTorch distribution in TORCH_ROOT.
#   SENSE_CORE_PYTHON      AUTO (default), ON or OFF: the pybind11 binding. pybind11_DIR and
#                          Python_EXECUTABLE name the environment it is built for.
#
# Sources stay within C++17 and CUDA C++17. The public header compiles as C and as C++20. The one
# exception is the optional LibTorch backend, whose library headers need C++20.
cmake_minimum_required(VERSION 3.18)

option(SENSE_CORE_HOST_ONLY "Build the host twin only: no CUDA compiler and no device" OFF)
if(SENSE_CORE_HOST_ONLY)
  project(sense_core LANGUAGES C CXX)
else()
  project(sense_core LANGUAGES C CXX CUDA)
endif()
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Build the sensing core out of tree: python scripts/native_core.py build")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
enable_testing()

set(SENSE_CORE_TENSORRT "AUTO" CACHE STRING "The TensorRT backend: AUTO, ON or OFF")
option(SENSE_CORE_LIBTORCH "Build the LibTorch backend, the declared fallback" OFF)
set(SENSE_CORE_PYTHON "AUTO" CACHE STRING "The pybind11 binding: AUTO, ON or OFF")
set(TENSORRT_ROOT "" CACHE PATH "Where TensorRT is installed: include/ and lib/")
set(TENSORRT_INCLUDE_DIR "" CACHE PATH "The directory that holds NvInfer.h")
set(TENSORRT_LIBRARY_DIR "" CACHE PATH "The directory that holds libnvinfer and libnvonnxparser")
set(TORCH_ROOT "" CACHE PATH "The torch directory of the PyTorch distribution")

set(SENSE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(OBS_RING_DIR "${SENSE_ROOT}/../obs_ring")
set(SENSE_WARNINGS -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wsuggest-override)

# ------------------------------------------------------------------------------------------------
# The identity of the sources. The hash names every file of this directory and the record
# definition of the ring, so a binary says what it was built from. scripts/native_core.py computes
# the same value from the tree: the text "path:sha256" of each file, sorted, one per line.
file(GLOB_RECURSE SENSE_SOURCE_FILES RELATIVE "${SENSE_ROOT}" "${SENSE_ROOT}/*")
list(FILTER SENSE_SOURCE_FILES EXCLUDE REGEX "(^|/)__pycache__/")
list(SORT SENSE_SOURCE_FILES)
set(SENSE_SOURCE_LIST "")
foreach(file IN LISTS SENSE_SOURCE_FILES)
  file(SHA256 "${SENSE_ROOT}/${file}" digest)
  string(APPEND SENSE_SOURCE_LIST "${file}:${digest}\n")
endforeach()
file(SHA256 "${OBS_RING_DIR}/include/obs/record.h" digest)
string(APPEND SENSE_SOURCE_LIST "obs_ring/include/obs/record.h:${digest}\n")
string(SHA256 SENSE_CORE_SOURCES_SHA256 "${SENSE_SOURCE_LIST}")
list(LENGTH SENSE_SOURCE_FILES SENSE_CORE_SOURCE_COUNT)
math(EXPR SENSE_CORE_SOURCE_COUNT "${SENSE_CORE_SOURCE_COUNT} + 1")
file(SHA256 "${SENSE_ROOT}/src/kernels/units.h" SENSE_UNIT_UNITS_SHA256)
file(SHA256 "${SENSE_ROOT}/src/engine/engine.h" SENSE_UNIT_ENGINE_SHA256)
file(SHA256 "${SENSE_ROOT}/src/engine/host_reference.cpp" SENSE_UNIT_HOST_REFERENCE_SHA256)
file(SHA256 "${SENSE_ROOT}/src/engine/tensorrt.cpp" SENSE_UNIT_TENSORRT_SHA256)
file(SHA256 "${SENSE_ROOT}/src/engine/libtorch.cpp" SENSE_UNIT_LIBTORCH_SHA256)
file(SHA256 "${SENSE_ROOT}/src/plan/plan.cpp" SENSE_UNIT_PLAN_SHA256)
file(SHA256 "${SENSE_ROOT}/src/plan/smoother.cpp" SENSE_UNIT_SMOOTHER_SHA256)

if(SENSE_CORE_HOST_ONLY)
  set(SENSE_CORE_DEVICE_KIND "host")
  set(SENSE_CORE_CUDA_ARCHITECTURES_TEXT "")
  set(SENSE_CORE_CUDA_COMPILER_TEXT "")
else()
  set(SENSE_CORE_DEVICE_KIND "cuda")
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
  # One CUDA runtime in a process: the shared one, which TensorRT and LibTorch load as well.
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  string(REPLACE ";" " " SENSE_CORE_CUDA_ARCHITECTURES_TEXT "${CMAKE_CUDA_ARCHITECTURES}")
  set(SENSE_CORE_CUDA_COMPILER_TEXT "${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
  find_package(CUDAToolkit REQUIRED)
endif()
configure_file("${SENSE_ROOT}/src/core/build_info.h.in" "${CMAKE_BINARY_DIR}/generated/sense_build_info.h" @ONLY)

# ------------------------------------------------------------------------------------------------
# TensorRT: optional. The probe of scripts/native_core.py records what a host has.
set(SENSE_HAVE_TENSORRT OFF)
set(SENSE_TENSORRT_VERSION "")
if(NOT SENSE_CORE_HOST_ONLY AND NOT SENSE_CORE_TENSORRT STREQUAL "OFF")
  find_path(SENSE_TENSORRT_INCLUDE NvInfer.h
    HINTS "${TENSORRT_INCLUDE_DIR}" "${TENSORRT_ROOT}/include" ENV TENSORRT_ROOT
    PATH_SUFFIXES include "${CMAKE_LIBRARY_ARCHITECTURE}")
  find_path(SENSE_TENSORRT_PARSER_INCLUDE NvOnnxParser.h
    HINTS "${TENSORRT_INCLUDE_DIR}" "${TENSORRT_ROOT}/include" "${SENSE_TENSORRT_INCLUDE}" ENV TENSORRT_ROOT
    PATH_SUFFIXES include "${CMAKE_LIBRARY_ARCHITECTURE}")
  # A package of runtime libraries holds libnvinfer.so.10 and no development link.
  find_library(SENSE_TENSORRT_NVINFER NAMES nvinfer libnvinfer.so.10
    HINTS "${TENSORRT_LIBRARY_DIR}" "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64" ENV TENSORRT_ROOT PATH_SUFFIXES lib lib64)
  find_library(SENSE_TENSORRT_NVONNXPARSER NAMES nvonnxparser libnvonnxparser.so.10
    HINTS "${TENSORRT_LIBRARY_DIR}" "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64" ENV TENSORRT_ROOT PATH_SUFFIXES lib lib64)
  if(SENSE_TENSORRT_INCLUDE AND SENSE_TENSORRT_PARSER_INCLUDE AND SENSE_TENSORRT_NVINFER AND SENSE_TENSORRT_NVONNXPARSER)
    set(SENSE_HAVE_TENSORRT ON)
    file(STRINGS "${SENSE_TENSORRT_INCLUDE}/NvInferVersion.h" lines REGEX "#define (TRT|NV_TENSORRT)_(MAJOR|MINOR|PATCH|BUILD)")
    foreach(part MAJOR MINOR PATCH BUILD)
      set(value "")
      foreach(line IN LISTS lines)
        if(line MATCHES "#define (TRT_${part}_ENTERPRISE|NV_TENSORRT_${part}) +([0-9]+)")
          set(value "${CMAKE_MATCH_2}")
        endif()
      endforeach()
      list(APPEND SENSE_TENSORRT_PARTS "${value}")
    endforeach()
    string(REPLACE ";" "." SENSE_TENSORRT_VERSION "${SENSE_TENSORRT_PARTS}")
    message(STATUS "sense_core: TensorRT ${SENSE_TENSORRT_VERSION}: ${SENSE_TENSORRT_INCLUDE}, ${SENSE_TENSORRT_NVINFER}")
  elseif(SENSE_CORE_TENSORRT STREQUAL "ON")
    message(FATAL_ERROR "SENSE_CORE_TENSORRT is ON, and TensorRT was not found: headers '${SENSE_TENSORRT_INCLUDE}', "
                        "parser header '${SENSE_TENSORRT_PARSER_INCLUDE}', libnvinfer '${SENSE_TENSORRT_NVINFER}', "
                        "libnvonnxparser '${SENSE_TENSORRT_NVONNXPARSER}'. Name it with TENSORRT_ROOT, or with "
                        "TENSORRT_INCLUDE_DIR and TENSORRT_LIBRARY_DIR.")
  else()
    message(STATUS "sense_core: no TensorRT with headers on this host: the kernels build, the TensorRT backend does not")
  endif()
elseif(SENSE_CORE_HOST_ONLY AND SENSE_CORE_TENSORRT STREQUAL "ON")
  message(FATAL_ERROR "The TensorRT backend runs on the CUDA device: it cannot be built with SENSE_CORE_HOST_ONLY")
endif()

# LibTorch: the declared fallback, only where the build asks for it.
if(SENSE_CORE_LIBTORCH)
  if(NOT EXISTS "${TORCH_ROOT}/include/torch/script.h")
    message(FATAL_ERROR "SENSE_CORE_LIBTORCH is ON, and TORCH_ROOT '${TORCH_ROOT}' holds no include/torch/script.h: "
                        "the PyTorch distribution must ship its C++ headers")
  endif()
  set(SENSE_TORCH_LIBRARIES "")
  foreach(name torch torch_cpu c10)
    find_library(SENSE_TORCH_${name} NAMES ${name} HINTS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH)
    if(NOT SENSE_TORCH_${name})
      message(FATAL_ERROR "SENSE_CORE_LIBTORCH is ON, and ${TORCH_ROOT}/lib holds no lib${name}")
    endif()
    list(APPEND SENSE_TORCH_LIBRARIES "${SENSE_TORCH_${name}}")
  endforeach()
  foreach(name torch_cuda c10_cuda)
    find_library(SENSE_TORCH_${name} NAMES ${name} HINTS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH)
    if(SENSE_TORCH_${name} AND NOT SENSE_CORE_HOST_ONLY)
      list(APPEND SENSE_TORCH_LIBRARIES "${SENSE_TORCH_${name}}")
    endif()
  endforeach()
  message(STATUS "sense_core: LibTorch backend from ${TORCH_ROOT}")
endif()

# ------------------------------------------------------------------------------------------------
# The ring: the record definition, and the consumer of sense_host. Its own unit cases come along.
add_subdirectory("${OBS_RING_DIR}" obs_ring)

# ------------------------------------------------------------------------------------------------
# The core library. No OCUDU archive, no Aerial library, no Python library.
set(SENSE_SOURCES
  src/core/sha256.cpp
  src/core/json.cpp
  src/core/artifact.cpp
  src/core/build.cpp
  src/kernels/host_device.cpp
  src/engine/engine_manifest.cpp
  src/engine/host_reference.cpp
  src/plan/smoother.cpp
  src/plan/plan.cpp
  src/plan/lines.cpp)
if(SENSE_CORE_HOST_ONLY)
  list(APPEND SENSE_SOURCES src/kernels/no_cuda_device.cpp)
else()
  list(APPEND SENSE_SOURCES src/kernels/cuda_device.cu)
endif()
if(SENSE_HAVE_TENSORRT)
  list(APPEND SENSE_SOURCES src/engine/tensorrt.cpp)
else()
  list(APPEND SENSE_SOURCES src/engine/no_tensorrt.cpp)
endif()
if(SENSE_CORE_LIBTORCH)
  list(APPEND SENSE_SOURCES src/engine/libtorch.cpp)
  # The headers of the PyTorch distribution need C++20. This one file is compiled that way: the
  # last language option of a command line is the one that holds.
  set_source_files_properties(src/engine/libtorch.cpp PROPERTIES COMPILE_OPTIONS "-std=c++20;-Wno-shadow;-Wno-suggest-override")
else()
  list(APPEND SENSE_SOURCES src/engine/no_libtorch.cpp)
endif()

add_library(sense_core STATIC ${SENSE_SOURCES})
target_include_directories(sense_core PUBLIC "${SENSE_ROOT}/include" "${OBS_RING_DIR}/include")
target_include_directories(sense_core PRIVATE "${SENSE_ROOT}/src" "${CMAKE_BINARY_DIR}/generated")
set_target_properties(sense_core PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
# The units must run as written, on the host and on the device: no fused multiply-add on either.
target_compile_options(sense_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2 -ffp-contract=off ${SENSE_WARNINGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
if(NOT SENSE_CORE_HOST_ONLY)
  set_target_properties(sense_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_link_libraries(sense_core PUBLIC CUDA::cudart)
endif()
if(SENSE_HAVE_TENSORRT)
  target_include_directories(sense_core SYSTEM PRIVATE "${SENSE_TENSORRT_INCLUDE}" "${SENSE_TENSORRT_PARSER_INCLUDE}")
  target_link_libraries(sense_core PUBLIC "${SENSE_TENSORRT_NVINFER}")
endif()
if(SENSE_CORE_LIBTORCH)
  target_include_directories(sense_core SYSTEM PRIVATE "${TORCH_ROOT}/include" "${TORCH_ROOT}/include/torch/csrc/api/include")
  # The CUDA part of the library registers itself when it is loaded: it must not be dropped.
  target_link_libraries(sense_core PUBLIC -Wl,--no-as-needed ${SENSE_TORCH_LIBRARIES} -Wl,--as-needed)
endif()

# ------------------------------------------------------------------------------------------------
# The sensor process, and the producer that feeds it records of an archive without a receiver.
add_executable(sense_host host/sense_host.cpp host/sensor_loop.cpp)
target_include_directories(sense_host PRIVATE "${SENSE_ROOT}/host" "${SENSE_ROOT}/src")
target_link_libraries(sense_host PRIVATE sense_core obs_ring)
add_executable(sense_feed tools/sense_feed.cpp)
target_link_libraries(sense_feed PRIVATE obs_ring)
foreach(tool sense_host sense_feed)
  set_target_properties(${tool} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  target_compile_options(${tool} PRIVATE -O2 -ffp-contract=off ${SENSE_WARNINGS})
  add_test(NAME ${tool}_help COMMAND ${tool} --help)
  set_tests_properties(${tool}_help PROPERTIES PASS_REGULAR_EXPRESSION "--socket PATH")
endforeach()

# The engine builder and the probe exist where TensorRT does. The probe is one file that links
# TensorRT and the CUDA runtime and nothing of the core: scripts/native_core.py compiles the same
# file on its own before the core is built.
if(SENSE_HAVE_TENSORRT)
  add_executable(sense_engine tools/sense_engine.cpp)
  target_include_directories(sense_engine PRIVATE "${SENSE_ROOT}/src" "${CMAKE_BINARY_DIR}/generated")
  target_include_directories(sense_engine SYSTEM PRIVATE "${SENSE_TENSORRT_INCLUDE}" "${SENSE_TENSORRT_PARSER_INCLUDE}")
  target_link_libraries(sense_engine PRIVATE sense_core "${SENSE_TENSORRT_NVONNXPARSER}")
  add_executable(sense_probe tools/sense_probe.cpp)
  target_compile_definitions(sense_probe PRIVATE SENSE_PROBE_ONNX)
  target_include_directories(sense_probe SYSTEM PRIVATE "${SENSE_TENSORRT_INCLUDE}" "${SENSE_TENSORRT_PARSER_INCLUDE}")
  target_link_libraries(sense_probe PRIVATE "${SENSE_TENSORRT_NVINFER}" "${SENSE_TENSORRT_NVONNXPARSER}" CUDA::cudart)
  foreach(tool sense_engine sense_probe)
    set_target_properties(${tool} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_compile_options(${tool} PRIVATE -O2 ${SENSE_WARNINGS})
    add_test(NAME ${tool}_help COMMAND ${tool} --help)
    set_tests_properties(${tool}_help PROPERTIES PASS_REGULAR_EXPRESSION "--device N")
  endforeach()
endif()

# ------------------------------------------------------------------------------------------------
# The binding, for tests, parity checks and offline scoring. No timed run imports it.
set(SENSE_HAVE_PYTHON OFF)
if(NOT SENSE_CORE_PYTHON STREQUAL "OFF")
  set(PYBIND11_FINDPYTHON ON) # the interpreter that Python_EXECUTABLE names, through FindPython
  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND)
    pybind11_add_module(_sense_core MODULE python/binding.cpp)
    target_include_directories(_sense_core PRIVATE "${SENSE_ROOT}/src")
    target_link_libraries(_sense_core PRIVATE sense_core)
    set_target_properties(_sense_core PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
    target_compile_options(_sense_core PRIVATE -O2 -ffp-contract=off)
    set(SENSE_HAVE_PYTHON ON)
  elseif(SENSE_CORE_PYTHON STREQUAL "ON")
    message(FATAL_ERROR "SENSE_CORE_PYTHON is ON, and pybind11 was not found. Install it in the environment and pass "
                        "-Dpybind11_DIR=$(python -m pybind11 --cmakedir)")
  else()
    message(STATUS "sense_core: no pybind11: the binding is not built")
  endif()
endif()

# ------------------------------------------------------------------------------------------------
# Unit target: the library, the sensor loop and doubles. One CTest case per unit case.
add_executable(sense_core_test tests/sense_core_test.cpp tests/alloc_count.cpp host/sensor_loop.cpp)
target_include_directories(sense_core_test PRIVATE "${SENSE_ROOT}/src" "${SENSE_ROOT}/host" "${CMAKE_BINARY_DIR}/generated")
target_link_libraries(sense_core_test PRIVATE sense_core obs_ring)
set_target_properties(sense_core_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(sense_core_test PRIVATE -O1 -ffp-contract=off -Wall -Wextra -Wshadow)
set(SENSE_CORE_CASES sha256 json_canonical half_widening vectors placement o3_element aerial_same_cube artifact_hashes
    registry_refused engine_manifest no_allocation one_submission stale_record late_status skip_backlog after_radio
    gap_and_reset smoothing lines ring_backpressure)
foreach(case IN LISTS SENSE_CORE_CASES)
  add_test(NAME sense_core_${case} COMMAND sense_core_test ${case})
  set_tests_properties(sense_core_${case} PROPERTIES TIMEOUT 120)
endforeach()
if(NOT SENSE_CORE_HOST_ONLY)
  # The same vectors through the kernels on the device. A host whose runtime reaches no device
  # reports the case as skipped, and the receipt says so.
  add_test(NAME sense_core_device_vectors COMMAND sense_core_test device_vectors)
  set_tests_properties(sense_core_device_vectors PROPERTIES SKIP_RETURN_CODE 77)
endif()

# The public header compiles as C and as C++20.
add_library(sense_api_c OBJECT tests/api_c.c)
target_include_directories(sense_api_c PRIVATE "${SENSE_ROOT}/include")
set_target_properties(sense_api_c PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_compile_options(sense_api_c PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_library(sense_api_cxx20 OBJECT tests/api_cxx20.cpp)
target_include_directories(sense_api_cxx20 PRIVATE "${SENSE_ROOT}/include")
set_target_properties(sense_api_cxx20 PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(sense_api_cxx20 PRIVATE -Wall -Wextra -Wpedantic -Werror)

# What this configuration holds, for the receipt.
file(WRITE "${CMAKE_BINARY_DIR}/sense-core-config.json"
"{\"schema\": \"sense-core-config/v1\", \"sources_sha256\": \"${SENSE_CORE_SOURCES_SHA256}\", \"device\": \"${SENSE_CORE_DEVICE_KIND}\", \"cuda_architectures\": \"${SENSE_CORE_CUDA_ARCHITECTURES_TEXT}\", \"cuda_compiler\": \"${SENSE_CORE_CUDA_COMPILER_TEXT}\", \"cxx_compiler\": \"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\", \"cxx_compiler_path\": \"${CMAKE_CXX_COMPILER}\", \"tensorrt\": \"${SENSE_TENSORRT_VERSION}\", \"tensorrt_include\": \"${SENSE_TENSORRT_INCLUDE}\", \"tensorrt_library\": \"${SENSE_TENSORRT_NVINFER}\", \"libtorch\": \"${SENSE_CORE_LIBTORCH}\", \"python_binding\": \"${SENSE_HAVE_PYTHON}\", \"build_type\": \"${CMAKE_BUILD_TYPE}\"}\n")
