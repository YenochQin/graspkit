cmake_minimum_required(VERSION 3.10)
project(CSFDescriptor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# 查找OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 核心库源文件（不包括main.cpp）
set(LIB_SOURCES
    src/csf_parser.cpp
    src/descriptor_generator.cpp
    src/parallel_processor.cpp
)

# 创建静态库
add_library(csf_lib STATIC ${LIB_SOURCES})

# 主程序源文件
set(MAIN_SOURCES src/main.cpp)

# 创建可执行文件
add_executable(csf_descriptor ${MAIN_SOURCES})

# 链接库
if(OpenMP_CXX_FOUND)
    target_link_libraries(csf_lib OpenMP::OpenMP_CXX)
    target_link_libraries(csf_descriptor csf_lib OpenMP::OpenMP_CXX)
else()
    target_link_libraries(csf_descriptor csf_lib)
endif()


# 安装规则
install(TARGETS csf_descriptor DESTINATION bin)

# 测试目标
enable_testing()
add_subdirectory(tests)