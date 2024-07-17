cmake_minimum_required(VERSION 3.28)
project(depth_anything_demo)

set(CMAKE_CXX_STANDARD 17)
find_package(ai_instance_tensorrt REQUIRED)
find_package(kaylordut REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(depth_anything_demo main.cpp depth_anything/depth_imageprocess.cpp)
target_link_libraries(depth_anything_demo ${ai_instance_tensorrt_LIBS} ${kaylordut_LIBS} ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS})
