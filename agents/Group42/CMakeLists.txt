cmake_minimum_required(VERSION 4.0)
project(Group42)

set(CMAKE_CXX_STANDARD 20)

add_executable(Group42 main.cpp
        Board.cpp
        Board.h
        helper.cpp
        helper.h)
