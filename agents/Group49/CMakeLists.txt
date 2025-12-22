cmake_minimum_required(VERSION 3.20)

project(Group49 LANGUAGES CXX)

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    message(STATUS "Defaulting to Release build.")
endif()

add_compile_options(-O3 -Wall)


# ONNX Runtime
# ./include/onnxruntime/...
# ./libonnxruntime.so.1

set(ONNX_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(ONNX_LIB_DIR     ${CMAKE_SOURCE_DIR})

if(NOT EXISTS "${ONNX_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "onnxruntime headers not found in ./include. "
        "Expected onnxruntime/core/session/onnxruntime_cxx_api.h"
    )
endif()

if(NOT EXISTS "${ONNX_LIB_DIR}/libonnxruntime.so.1")
    message(FATAL_ERROR
        "libonnxruntime.so.1 not found in project root"
    )
endif()

include_directories(${ONNX_INCLUDE_DIR})
link_directories(${ONNX_LIB_DIR})


# RPATH
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")


# Threads
find_package(Threads REQUIRED)


# Targets
add_executable(cpp_agent
    main.cpp
    src/Util.cpp
    src/Position.cpp
    src/Node.cpp
    src/Inference.cpp
)

add_executable(Group49SelfPlay
    self_play.cpp
    src/Util.cpp
    src/Position.cpp
    src/Node.cpp
    src/Inference.cpp
)


# Linking
target_link_libraries(cpp_agent
    onnxruntime
    Threads::Threads
)

target_link_libraries(Group49SelfPlay
    onnxruntime
    Threads::Threads
)
