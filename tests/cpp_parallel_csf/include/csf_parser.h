#pragma once

#include "csf_types.h"
#include <fstream>
#include <vector>
#include <string>

namespace csf {

class CSFParser {
public:
    // 从文件解析CSF数据
    static CSFFileData parse_file(const std::string& filename);
    
    // 从字符串向量解析CSF数据
    static CSFFileData parse_lines(const std::vector<std::string>& lines);

private:
    // 解析剥离子壳层列表
    static PeelSubshells parse_peel_subshells(const std::string& line);
};

} // namespace csf