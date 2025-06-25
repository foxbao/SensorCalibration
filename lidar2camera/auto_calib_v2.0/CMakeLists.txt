cmake_minimum_required(VERSION 2.8.12)
project(LidarToCamera)

# 使用 C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 构建类型与优化设置
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")

# 线程库
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# 查找依赖项
find_package(PCL REQUIRED)
find_package(Boost REQUIRED system)
find_package(OpenCV REQUIRED)

# include 目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN_ROOT})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PCL_INCLUDE_DIRS})

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# 源文件
file(GLOB_RECURSE PARSER_PATH src/*.cpp)

# 构建静态库
add_library(${PROJECT_NAME} STATIC ${PARSER_PATH})
target_link_libraries(${PROJECT_NAME}
    jsoncpp                      # ✅ 链接系统动态库 libjsoncpp.so
    ${OpenCV_LIBS}
    ${Boost_SYSTEM_LIBRARY}
    Threads::Threads
)

# 可执行程序
add_executable(run_lidar2camera src/run_lidar2camera.cpp)
target_link_libraries(run_lidar2camera
    ${PROJECT_NAME}
    ${PCL_LIBRARIES}
)
