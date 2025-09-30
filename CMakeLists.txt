cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(MyProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)
# 查找 Ceres 包
find_package(Ceres REQUIRED)
# 查找 Eigen 包
find_package(Eigen3 REQUIRED)

# 包含头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# 添加源文件
set(SOURCES 
    src/main.cpp 

)

# 添加可执行文件
add_executable(MyExecutable 
    src/main.cpp
   
)

# 链接 OpenCV, Ceres 和 Eigen
target_link_libraries(MyExecutable ${OpenCV_LIBS} ${CERES_LIBRARIES})
