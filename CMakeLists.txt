cmake_minimum_required(VERSION 3.26)
project(quaml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# SKBUILD_SABI_COMPONENT is Development.SABIModule when building an abi3 wheel.
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module
             ${SKBUILD_SABI_COMPONENT})
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

include(FetchContent)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED   OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC   ON  CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_Declare(zstd
    URL https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz
    URL_HASH SHA256=8c29e06cf42aacc1eafc4077ae2ec6c6fcb96a626157e0593d5e82a34fd403c1
    SOURCE_SUBDIR build/cmake
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    EXCLUDE_FROM_ALL) # keep zstd's headers and libs out of the wheel
FetchContent_MakeAvailable(zstd)

add_library(quaml_events STATIC events/dbn.cpp events/features.cpp)
target_include_directories(quaml_events PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(quaml_events PRIVATE ${zstd_SOURCE_DIR}/lib)
target_link_libraries(quaml_events PRIVATE libzstd_static)
set_target_properties(quaml_events PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(quaml_events PRIVATE /W4 /permissive-)
else()
    target_compile_options(quaml_events PRIVATE -Wall -Wextra)
endif()

# STABLE_ABI: on Python >= 3.12 one build serves every later version (abi3).
nanobind_add_module(_core NB_STATIC STABLE_ABI python/bindings.cpp)
target_link_libraries(_core PRIVATE quaml_events)
install(TARGETS _core LIBRARY DESTINATION quaml)
