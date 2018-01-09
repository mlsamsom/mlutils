cmake_minimum_required(VERSION 2.8)

# project setup
project(scenes)

# C++ version
set(CMAKE_CXX_STANDARD 11)

# dependencies
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# build
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(scenedet_tests ${SOURCES})
target_link_libraries(scenedet_tests
  ${OpenCV_LIBS})
