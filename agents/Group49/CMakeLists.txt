cmake_minimum_required(VERSION 4.0)
project(Group49)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
if(NOT MSVC)
    add_compile_options(-march=native -O3)
endif()

add_executable(Group49 main.cpp
        src/Util.cpp
        src/Util.h
        src/Position.cpp
        src/Position.h
        src/Node.cpp
        src/MCTS.cpp
        src/MCTS.h)

# 1. Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 2. Define the Test Executable
# distinct from your main 'Group49' executable
add_executable(HexTests tests/test_win_logic.cpp)

# 3. Link your Engine Source Files
# You must include Position.cpp (and others) so the test can see them.
# ideally, make your engine a library, but listing sources works for coursework.
target_sources(HexTests PRIVATE src/Position.cpp src/Util.cpp)

# 4. Link GTest
target_link_libraries(HexTests PRIVATE gtest_main)

# 5. Enable Testing in Cmake
include(GoogleTest)
gtest_discover_tests(HexTests)