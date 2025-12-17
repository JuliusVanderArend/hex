cmake_minimum_required(VERSION 3.20)
project(Group49)

set(CMAKE_CXX_STANDARD 20)
# Optimization flags
if(APPLE)
    add_compile_options(-O3 -Wall)
    # macOS: Use Homebrew's ONNX Runtime
    set(ONNX_ROOT "/opt/homebrew/opt/onnxruntime")
    include_directories(${ONNX_ROOT}/include/onnxruntime)
else()
    add_compile_options(-O3 -march=native -Wall)
    # Linux: Use local ONNX Runtime
    set(ONNX_ROOT "$ENV{HOME}/libs/onnxruntime")
    include_directories(${ONNX_ROOT}/include)
endif()

link_directories(${ONNX_ROOT}/lib)

# 2. DEFINE EXECUTABLES
add_executable(Group49 main.cpp src/Util.cpp src/Util.h src/Position.cpp src/Position.h src/Node.cpp src/MCTS.h src/Inference.cpp
        src/InferenceServer.h)
add_executable(Group49SelfPlay
        self_play.cpp
        src/Util.cpp src/Util.h
        src/Position.cpp src/Position.h
        src/Node.cpp
        src/MCTS.h
        src/Inference.cpp
        src/InferenceServer.h
)

# 3. LINK LIBRARIES
target_link_libraries(Group49 onnxruntime)
target_link_libraries(Group49SelfPlay onnxruntime)

# 4. Platform-specific library handling
if(APPLE)
    # macOS: Set RPATH to find dylib at runtime
    set_target_properties(Group49 PROPERTIES
        BUILD_RPATH "${ONNX_ROOT}/lib"
        INSTALL_RPATH "${ONNX_ROOT}/lib"
    )
    set_target_properties(Group49SelfPlay PROPERTIES
        BUILD_RPATH "${ONNX_ROOT}/lib"
        INSTALL_RPATH "${ONNX_ROOT}/lib"
    )
else()
    # Linux: Copy .so file
    add_custom_command(TARGET Group49 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNX_ROOT}/lib/libonnxruntime.so.1.23.2"
            $<TARGET_FILE_DIR:Group49>/libonnxruntime.so.1.23.2)

    add_custom_command(TARGET Group49SelfPlay POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNX_ROOT}/lib/libonnxruntime.so.1.23.2"
            $<TARGET_FILE_DIR:Group49SelfPlay>/libonnxruntime.so.1.23.2)
endif()

add_executable(Arbiter arbiter.cpp src/Position.cpp src/Util.cpp)

# --- TESTS ---
include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/heads/main.zip)
FetchContent_MakeAvailable(googletest)

add_executable(HexTests tests/tests.cpp
        src/InferenceServer.h)
target_sources(HexTests PRIVATE src/Position.cpp src/Util.cpp src/Inference.cpp)

target_link_libraries(HexTests PRIVATE gtest_main onnxruntime)

if(APPLE)
    set_target_properties(HexTests PROPERTIES
        BUILD_RPATH "${ONNX_ROOT}/lib"
        INSTALL_RPATH "${ONNX_ROOT}/lib"
    )
else()
    add_custom_command(TARGET HexTests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNX_ROOT}/lib/libonnxruntime.so.1.23.2"
            $<TARGET_FILE_DIR:HexTests>/libonnxruntime.so.1.23.2)
endif()

include(GoogleTest)
gtest_discover_tests(HexTests)