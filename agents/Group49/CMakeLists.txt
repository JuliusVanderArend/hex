cmake_minimum_required(VERSION 3.20)
project(Group49)

set(CMAKE_CXX_STANDARD 20)
# GCC Optimization flags (Now safe to use again!)
add_compile_options(-O3 -march=native -Wall)

# 1. DEFINE PATH (Linux Path inside WSL)
# TIP: Use $ENV{HOME} to represent /home/your_user
set(ONNX_ROOT "$ENV{HOME}/libs/onnxruntime")

include_directories(${ONNX_ROOT}/include)
link_directories(${ONNX_ROOT}/lib)

# 2. DEFINE EXECUTABLES
#add_executable(Group49 main.cpp src/Util.cpp src/Util.h src/Position.cpp src/Position.h src/Node.cpp src/MCTS.cpp src/MCTS.h src/Inference.cpp src/Inference.h)
add_executable(Group49 main.cpp src/Util.cpp src/Util.h src/Position.cpp src/Position.h src/Node.cpp src/MCTS.cpp src/MCTS.h)

# 3. LINK LIBRARIES (Simple and Clean)
# Linux finds libonnxruntime.so automatically via link_directories
target_link_libraries(Group49 onnxruntime)

# 4. COPY SHARED OBJECT (.so)
# In Linux, we copy .so.1.19.2 (the versioned file) or the symbolic link
add_custom_command(TARGET Group49 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNX_ROOT}/lib/libonnxruntime.so.1.23.2"
        $<TARGET_FILE_DIR:Group49>/libonnxruntime.so.1.23.2)

# --- TESTS ---
include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/heads/main.zip)
FetchContent_MakeAvailable(googletest)

add_executable(HexTests tests/tests.cpp)
#target_sources(HexTests PRIVATE src/Position.cpp src/Util.cpp src/NeuralNet.cpp)
target_sources(HexTests PRIVATE src/Position.cpp src/Util.cpp)

target_link_libraries(HexTests PRIVATE gtest_main onnxruntime)

# Copy .so for tests too
add_custom_command(TARGET HexTests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNX_ROOT}/lib/libonnxruntime.so.1.23.2"
        $<TARGET_FILE_DIR:HexTests>/libonnxruntime.so.1.23.2)

include(GoogleTest)
gtest_discover_tests(HexTests)