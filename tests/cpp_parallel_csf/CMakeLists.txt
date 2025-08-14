cmake_minimum_required(VERSION 3.10)
project(CSFDescriptor VERSION 1.0.0 DESCRIPTION "C++ Parallel CSF Descriptor Generator")

# 设置项目信息
set(PROJECT_VENDOR "CSF Team")
set(PROJECT_CONTACT "csf@example.com")
set(PROJECT_URL "https://github.com/csf-team/csf-descriptor")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置RPATH以便可执行文件可以找到库
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# 查找依赖
find_package(OpenMP)
find_package(HDF5 REQUIRED COMPONENTS CXX)

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${HDF5_INCLUDE_DIRS})

# 核心库源文件
set(LIB_SOURCES
    src/csf_parser.cpp
    src/descriptor_generator.cpp
    src/parallel_processor.cpp
    src/hdf5_writer.cpp
)

# 创建静态库
add_library(csf_lib STATIC ${LIB_SOURCES})
set_target_properties(csf_lib PROPERTIES OUTPUT_NAME csf)

# 主程序
add_executable(csf_descriptor src/main.cpp)

# 链接库
if(OpenMP_CXX_FOUND)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(csf_descriptor csf_lib OpenMP::OpenMP_CXX ${HDF5_CXX_LIBRARIES})
        target_link_libraries(csf_lib OpenMP::OpenMP_CXX ${HDF5_CXX_LIBRARIES})
    else()
        target_link_libraries(csf_descriptor csf_lib ${HDF5_CXX_LIBRARIES} ${OMP_LIBRARY})
        target_link_libraries(csf_lib ${HDF5_CXX_LIBRARIES} ${OMP_LIBRARY})
    endif()
else()
    target_link_libraries(csf_descriptor csf_lib ${HDF5_CXX_LIBRARIES})
    target_link_libraries(csf_lib ${HDF5_CXX_LIBRARIES})
endif()

# 安装规则
install(TARGETS csf_descriptor csf_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# 安装头文件
install(DIRECTORY include/
    DESTINATION include/csf
    FILES_MATCHING PATTERN "*.h"
)

# 安装文档
install(FILES README.md
    DESTINATION share/doc/csf-descriptor
)

# 创建pkg-config文件
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/csf.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/csf.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/csf.pc
    DESTINATION lib/pkgconfig)

# 创建CMake配置文件
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/csf-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/csf-config.cmake
    INSTALL_DESTINATION lib/cmake/csf
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/csf-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/csf-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/csf-config-version.cmake
    DESTINATION lib/cmake/csf
)

# CPack配置
set(CPACK_PACKAGE_NAME "csf-descriptor")
set(CPACK_PACKAGE_VENDOR "${PROJECT_VENDOR}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C++ Parallel CSF Descriptor Generator")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "csf-descriptor")
set(CPACK_PACKAGE_CONTACT "${PROJECT_CONTACT}")
set(CPACK_PACKAGE_HOMEPAGE_URL "${PROJECT_URL}")

# 平台特定配置
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
elseif(APPLE)
    set(CPACK_GENERATOR "TGZ;DragNDrop")
else()
    set(CPACK_GENERATOR "TGZ;DEB;RPM")
endif()

# 包含CPack模块
include(CPack)

# 测试
enable_testing()
add_subdirectory(tests)

# 创建卸载目标
add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake @ONLY)