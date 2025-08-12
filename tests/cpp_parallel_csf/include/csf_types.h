#pragma once

#include <vector>
#include <string>
#include <array>

namespace csf {

// 子壳层信息结构
struct SubshellInfo {
    int principal_quantum_num;  // 主量子数n
    int kappa;                  // kappa值
    int electron_count;         // 电子数
    double intermediate_j;      // 中间J值
    double coupled_j;           // 耦合J值
    bool is_occupied;           // 是否被占用
    bool is_full;               // 是否填满
};

// CSF数据结构（三行数据）
struct CSFData {
    std::string subshell_line;      // 第一行：子壳层和电子数
    std::string intermediate_line;  // 第二行：中间J耦合值
    std::string coupling_line;      // 第三行：最终耦合和总J值
    
    CSFData() = default;
    CSFData(const std::string& s1, const std::string& s2, const std::string& s3)
        : subshell_line(s1), intermediate_line(s2), coupling_line(s3) {}
};

// 描述符类型
using Descriptor = std::vector<double>;
using DescriptorArray = std::vector<Descriptor>;

// 剥离子壳层列表类型
using PeelSubshells = std::vector<std::string>;

// CSF文件数据结构
struct CSFFileData {
    PeelSubshells peel_subshells;
    std::vector<std::vector<CSFData>> csf_blocks;  // 按块组织的CSF数据
    
    size_t total_csfs() const {
        size_t count = 0;
        for (const auto& block : csf_blocks) {
            count += block.size();
        }
        return count;
    }
};

} // namespace csf