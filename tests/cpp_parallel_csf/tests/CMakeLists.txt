cmake_minimum_required(VERSION 3.10)

# 测试可执行文件
add_executable(test_csf_parser test_parser.cpp)
target_link_libraries(test_csf_parser csf_lib)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_csf_parser OpenMP::OpenMP_CXX)
endif()

# 添加测试
add_test(NAME ParserTest COMMAND test_csf_parser)