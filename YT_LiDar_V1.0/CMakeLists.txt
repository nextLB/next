cmake_minimum_required(VERSION 3.0)
project(YT_LiDar_V1.0)

# 设置构建类型和编译选项
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -DNDEBUG")

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 包含项目的头文件目录和OpenCV头文件目录
include_directories(include ${OpenCV_INCLUDE_DIRS})

# 设置库文件的搜索路径
link_directories(lib/${CMAKE_SYSTEM_PROCESSOR})

# 设置可执行文件和库文件的输出路径
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR})



# 添加可执行文件 YT_lidar_V1.0
add_executable(YT_lidar_V1.0
  examples/YT_V1.0.cpp
)

# 链接 YT_lidar_V1.0 可执行文件所需的库
target_link_libraries(YT_lidar_V1.0  libunitree_lidar_sdk.a ${OpenCV_LIBS})
