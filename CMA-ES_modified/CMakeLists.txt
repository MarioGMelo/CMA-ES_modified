cmake_minimum_required(VERSION 3.8)
project(CMA_ES_modified)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/ObjectiveFunction.cpp
        main.cpp)

add_executable(CMA_ES_modified ${SOURCE_FILES})