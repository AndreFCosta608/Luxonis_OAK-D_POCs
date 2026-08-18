# poc_detection/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(poc_detection VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(depthai REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(poc_detection main.cpp)

target_link_libraries(poc_detection
    depthai::core
    depthai::opencv
    ${OpenCV_LIBS}
)