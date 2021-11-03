

cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 14)

project(histogram)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(segmentation main preprocessing.h localization.h)

# linking
target_link_libraries(segmentation ${OpenCV_LIBS})