cmake_minimum_required(VERSION 3.20)
project(Group49)

set(CMAKE_CXX_STANDARD 20)

# --- [1] FORCE RELEASE MODE ---
# Automatically defaults to Release builds for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "Defaulting to Release build.")
endif()

# --- [2] SAFETY FLAGS ---
# CRITICAL: Removed -march=native to ensure the binary runs on the grader's potentially older CPU.
add_compile_options(-O3 -Wall)

# --- [3] BAKED-IN DOCKER PATH CONFIG ---
# Points to the fixed location in your modified Docker image.
set(ONNX_ROOT "/opt/onnxruntime")

if(NOT EXISTS "${ONNX_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNX_ROOT}. Ensure you are building inside your modified Docker image.")
endif()

include_directories(${ONNX_ROOT}/include)
link_directories(${ONNX_ROOT}/lib)

# --- [4] PORTABILITY (RPATH) ---
# This tells the binary: "Look for .so files in the same folder as me ($ORIGIN)"
# This allows the binary to run on the grader's machine without the library being installed globally.
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# --- [5] DEPENDENCIES ---
find_package(Threads REQUIRED)

# --- [6] TARGETS ---
add_executable(Group49
        main.cpp
        src/Util.cpp src/Util.h
        src/Position.cpp src/Position.h
        src/Node.cpp src/MCTS.h
        src/Inference.cpp src/InferenceServer.h
)

add_executable(Group49SelfPlay
        self_play.cpp
        src/Util.cpp src/Util.h
        src/Position.cpp src/Position.h
        src/Node.cpp src/MCTS.h
        src/Inference.cpp src/InferenceServer.h
)

# --- [7] LINKING ---
# Link onnxruntime AND system threads (prevents linker errors)
target_link_libraries(Group49 onnxruntime Threads::Threads)
target_link_libraries(Group49SelfPlay onnxruntime Threads::Threads)

# --- [8] AUTO-COPY LIBRARIES ---
# Automatically pulls the .so file from the Docker image (/opt) and places it next to your binary.
add_custom_command(TARGET Group49 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNX_ROOT}/lib/libonnxruntime.so.1.23.2"
        $<TARGET_FILE_DIR:Group49>/libonnxruntime.so.1.23.2)