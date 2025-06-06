cmake_minimum_required(VERSION 2.8.3)
project(LidarToLidar)
set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall")
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
# set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Get Pangolin
find_package(Pangolin 0.4 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(PCL REQUIRED)
# file(GLOB PCL_LIBRARIES /usr/local/lib/libpcl_*)
# set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.12)
find_package(Boost REQUIRED system)
find_package(OpenCV REQUIRED)

link_directories(${OpenCV_LIBRARY_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${EIGEN_ROOT})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PCL_INCLUDE_DIRS})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

file(GLOB_RECURSE PARSER_PATH src/*.cpp)
add_library(${PROJECT_NAME} STATIC ${PARSER_PATH})
target_link_libraries(${PROJECT_NAME} jsoncpp ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARY})

add_executable(run_lidar2lidar src/run_lidar2lidar.cpp)
target_link_libraries(run_lidar2lidar ${PROJECT_NAME})
target_link_libraries(run_lidar2lidar ${PCL_LIBRARIES})