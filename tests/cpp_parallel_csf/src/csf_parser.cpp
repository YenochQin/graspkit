#include "csf_parser.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace csf {

CSFFileData CSFParser::parse_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    
    return parse_lines(lines);
}

CSFFileData CSFParser::parse_lines(const std::vector<std::string>& lines) {
    CSFFileData data;
    
    if (lines.empty()) {
        throw std::runtime_error("Empty CSF file");
    }
    
    // 查找关键行
    size_t core_line = 0, peel_line = 0, csf_start = 0;
    
    for (size_t i = 0; i < lines.size(); ++i) {
        if (lines[i].find("Core subshells:") != std::string::npos) {
            core_line = i;
        } else if (lines[i].find("Peel subshells:") != std::string::npos) {
            peel_line = i;
        } else if (lines[i].find("CSF(s):") != std::string::npos) {
            csf_start = i + 1;
            break;
        }
    }
    
    if (peel_line == 0 || csf_start == 0) {
        throw std::runtime_error("Invalid GRASP CSF file format: missing required sections");
    }
    
    // 解析剥离子壳层列表
    data.peel_subshells = parse_peel_subshells(lines[peel_line + 1]);
    
    // 解析CSF数据（每3行为一个CSF）
    std::vector<CSFData> csf_block;
    
    for (size_t i = csf_start; i + 2 < lines.size(); i += 3) {
        if (lines[i].empty()) continue;
        
        CSFData csf;
        csf.subshell_line = lines[i];
        csf.intermediate_line = lines[i + 1];
        csf.coupling_line = lines[i + 2];
        
        csf_block.push_back(csf);
    }
    
    if (!csf_block.empty()) {
        data.csf_blocks.push_back(csf_block);
    }
    
    return data;
}

PeelSubshells CSFParser::parse_peel_subshells(const std::string& line) {
    PeelSubshells subshells;
    std::istringstream iss(line);
    std::string subshell;
    
    while (iss >> subshell) {
        if (!subshell.empty() && subshell != "subshells:") {
            // 清理子壳层字符串
            subshell.erase(std::remove(subshell.begin(), subshell.end(), '\t'), subshell.end());
            if (!subshell.empty()) {
                subshells.push_back(subshell);
            }
        }
    }
    
    return subshells;
}


} // namespace csf