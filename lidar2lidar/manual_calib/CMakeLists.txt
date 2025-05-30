cmake_minimum_required(VERSION 2.8.3)
project(LidarToLidar)
set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall")
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)

set(CMAKE_CXX_STANDARD 14)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
link_directories(${Pangolin_LIBRARY_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})


find_package(Boost REQUIRED system)
find_package(OpenCV REQUIRED)
# find_package(jsoncpp REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN_ROOT})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

file(GLOB_RECURSE PARSER_PATH src/*.cpp)
add_library(${PROJECT_NAME} STATIC ${PARSER_PATH})
target_link_libraries(${PROJECT_NAME} jsoncpp ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARY})

add_executable(run_lidar2lidar src/run_lidar2lidar.cpp)
target_link_libraries(run_lidar2lidar ${PROJECT_NAME})
target_link_libraries(run_lidar2lidar ${PCL_LIBRARIES})
target_link_libraries(run_lidar2lidar ${Pangolin_LIBRARIES})
