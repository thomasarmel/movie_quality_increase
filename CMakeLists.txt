cmake_minimum_required(VERSION 3.13)
project(movie_quality_increase)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(movie_quality_increase main.cpp SuperRes.cpp SuperRes.h)

target_include_directories(movie_quality_increase PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(movie_quality_increase ${OpenCV_LIBS} pthread)