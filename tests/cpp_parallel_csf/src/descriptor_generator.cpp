#include "descriptor_generator.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <cctype>

namespace csf {

// 优化的字符串处理函数
static inline std::string_view trim_view(std::string_view str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string_view::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

static inline std::string_view trim(const std::string& str) {
    return trim_view(std::string_view(str));
}

// 优化的分块处理 - 避免创建临时字符串
static inline std::string_view get_chunk(std::string_view str, size_t pos, size_t chunk_size) {
    if (pos >= str.length()) return "";
    return str.substr(pos, std::min(chunk_size, str.length() - pos));
}

// 优化的J值转换函数
static inline int j_to_double_j_fixed(std::string_view j_str) {
    j_str = trim_view(j_str);
    size_t slash_pos = j_str.find('/');
    if (slash_pos != std::string_view::npos) {
        // 半整数情况，如 "3/2"
        int numerator = 0;
        for (size_t i = 0; i < slash_pos; ++i) {
            if (std::isdigit(j_str[i])) {
                numerator = numerator * 10 + (j_str[i] - '0');
            }
        }
        return numerator;
    } else {
        // 整数情况，如 "2"
        int value = 0;
        for (char c : j_str) {
            if (std::isdigit(c)) {
                value = value * 10 + (c - '0');
            }
        }
        return value * 2;
    }
}

// 辅助函数：检查子壳层是否填满
static bool is_subshell_full_fixed(const std::string& subshell, int electron_count) {
    static const std::unordered_map<std::string, int> max_electrons = {
        {"s ", 2}, {"p-", 2}, {"p ", 4}, {"d-", 4},
        {"d ", 6}, {"f-", 6}, {"f ", 8}, {"g-", 8},
        {"g ", 10}, {"h-", 10}, {"h ", 12}, {"i-", 12},
        {"i ", 14}
    };
    
    auto it = max_electrons.find(subshell);
    return (it != max_electrons.end()) ? (it->second == electron_count) : false;
}

// 预创建子壳层到索引的映射
static std::unordered_map<std::string, size_t> create_subshell_index_map(
    const PeelSubshells& peel_subshells) {
    std::unordered_map<std::string, size_t> map;
    map.reserve(peel_subshells.size());
    for (size_t i = 0; i < peel_subshells.size(); ++i) {
        map[peel_subshells[i]] = i;
    }
    return map;
}

// 辅助函数：将子壳层转换为kappa值
static int subshell_to_kappa_fixed(const std::string& subshell) {
    static const std::unordered_map<std::string, int> kappa_map = {
        {"s ", -1}, {"p-", 1}, {"p ", -2}, {"d-", 2},
        {"d ", -3}, {"f-", 3}, {"f ", -4}, {"g-", 4},
        {"g ", -5}, {"h-", 5}, {"h ", -6}, {"i-", 6},
        {"i ", -7}
    };
    
    auto it = kappa_map.find(subshell);
    return (it != kappa_map.end()) ? it->second : 0;
}

Descriptor DescriptorGenerator::generate_basic_descriptor(const CSFData& csf,
                                                         const PeelSubshells& peel_subshells) {
    const size_t descriptor_size = 3 * peel_subshells.size();
    Descriptor descriptor(descriptor_size, 0.0);
    
    // 预处理CSF的三行数据，去除末尾换行符并统一长度
    std::string subshells_line = csf.subshell_line;
    std::string middle_line_raw = csf.intermediate_line;
    std::string coupling_line_raw = csf.coupling_line;
    
    // 去除换行符和尾部空格
    subshells_line = subshells_line.substr(0, subshells_line.find_last_not_of("\r\n") + 1);
    middle_line_raw = middle_line_raw.substr(0, middle_line_raw.find_last_not_of("\r\n") + 1);
    coupling_line_raw = coupling_line_raw.substr(0, coupling_line_raw.find_last_not_of("\r\n") + 1);
    
    size_t line_length = subshells_line.length();  // 以第一行长度为标准
    
    // 左对齐并填充到指定长度
    std::string middle_line = middle_line_raw;
    if (middle_line.length() < line_length) {
        middle_line += std::string(line_length - middle_line.length(), ' ');
    }
    
    // 去除前4位和后5位，然后左对齐
    std::string coupling_line = coupling_line_raw;
    if (coupling_line.length() > 9) {
        coupling_line = coupling_line.substr(4, coupling_line.length() - 9);
    }
    if (coupling_line.length() < line_length) {
        coupling_line += std::string(line_length - coupling_line.length(), ' ');
    }
    
    // 提取最终J值（从第三行的后5位中提取）
    std::string final_J = coupling_line_raw.substr(coupling_line_raw.length() - 5, 4);
    final_J = trim(final_J);
    int final_double_J = j_to_double_j_fixed(final_J);
    
    // 将三行数据按每9个字符分块处理
    std::vector<std::string> subshell_List = chunk_string_fixed(subshells_line, 9);
    std::vector<std::string> middle_line_List = chunk_string_fixed(middle_line, 9);
    std::vector<std::string> coupling_line_List = chunk_string_fixed(coupling_line, 9);
    
    // 预创建子壳层到索引的映射
    static thread_local std::unordered_map<std::string, size_t> subshell_map;
    if (subshell_map.empty()) {
        subshell_map = create_subshell_index_map(peel_subshells);
    }
    
    // 使用固定大小的数组替代vector，避免动态分配
    std::array<bool, 50> occupied_orbits{};  // 假设最多50个轨道
    
    // 遍历每个子壳层块，提取和处理信息
    for (size_t i = 0; i < subshell_List.size(); ++i) {
        const std::string& subshell_charges = subshell_List[i];
        const std::string& middle_line_item = (i < middle_line_List.size()) ? middle_line_List[i] : "";
        const std::string& coupling_line_item = (i < coupling_line_List.size()) ? coupling_line_List[i] : "";
        
        if (subshell_charges.length() < 8) continue;
        
        // 提取子壳层名称和电子数 - 优化字符串处理
        std::string subshell = trim(subshell_charges.substr(0, 5));
        int subshell_electron_num = (subshell_charges[6] - '0') * 10 + (subshell_charges[7] - '0');
        bool is_last = (i == subshell_List.size() - 1);
        
        // 快速查找索引 - O(1)哈希查找替代O(n)线性查找
        auto it = subshell_map.find(subshell);
        if (it == subshell_map.end()) {
            continue;
        }
        size_t orbs_index = it->second;
        if (orbs_index >= peel_subshells.size()) continue;
        
        size_t descriptor_index = orbs_index * 3;
        occupied_orbits[orbs_index] = true;
        
        // 处理第二行数据（中间J耦合值）
        int temp_middle_item = 0;
        if (!middle_line_item.empty()) {
            std::string middle_item = trim(middle_line_item);
            if (!middle_item.empty()) {
                // 快速处理最后一个分号后的值
                size_t semicolon_pos = middle_item.rfind(';');
                if (semicolon_pos != std::string::npos) {
                    middle_item = trim(middle_item.substr(semicolon_pos + 1));
                }
                if (!middle_item.empty()) {
                    temp_middle_item = j_to_double_j_fixed(middle_item);
                }
            }
        }
        
        // 处理第三行数据（耦合J值）
        int temp_coupling_item = 0;
        if (!coupling_line_item.empty()) {
            std::string coupling_item = trim(coupling_line_item);
            if (!coupling_item.empty()) {
                temp_coupling_item = j_to_double_j_fixed(coupling_item);
            }
        } else if (temp_middle_item != 0) {
            temp_coupling_item = temp_middle_item;
        }

        // 特殊处理：最后一个子壳层使用最终J值
        if (is_last) {
            temp_coupling_item = final_double_J;
        }
        
        // 直接填充描述符数组
        descriptor[descriptor_index] = static_cast<double>(subshell_electron_num);
        descriptor[descriptor_index + 1] = static_cast<double>(temp_middle_item);
        descriptor[descriptor_index + 2] = static_cast<double>(temp_coupling_item);
    }
    
    // 为未占用轨道填充最终J值 - 优化处理
    for (size_t i = 0; i < peel_subshells.size(); ++i) {
        if (!occupied_orbs[i]) {
            descriptor[i * 3 + 2] = static_cast<double>(final_double_J);
        }
    }
    
    return descriptor;
}

Descriptor DescriptorGenerator::generate_extended_descriptor(const CSFData& csf,
                                                           const PeelSubshells& peel_subshells) {
    const size_t descriptor_size = 5 * peel_subshells.size();
    Descriptor descriptor(descriptor_size, 0.0);
    
    // 预处理CSF的三行数据，去除末尾换行符并统一长度
    std::string subshells_line = csf.subshell_line;
    std::string middle_line_raw = csf.intermediate_line;
    std::string coupling_line_raw = csf.coupling_line;
    
    // 去除换行符和尾部空格
    subshells_line = subshells_line.substr(0, subshells_line.find_last_not_of("\r\n") + 1);
    middle_line_raw = middle_line_raw.substr(0, middle_line_raw.find_last_not_of("\r\n") + 1);
    coupling_line_raw = coupling_line_raw.substr(0, coupling_line_raw.find_last_not_of("\r\n") + 1);
    
    size_t line_length = subshells_line.length();  // 以第一行长度为标准
    
    // 左对齐并填充到指定长度
    std::string middle_line = middle_line_raw;
    if (middle_line.length() < line_length) {
        middle_line += std::string(line_length - middle_line.length(), ' ');
    }
    
    // 去除前4位和后5位，然后左对齐
    std::string coupling_line = coupling_line_raw;
    if (coupling_line.length() > 9) {
        coupling_line = coupling_line.substr(4, coupling_line.length() - 9);
    }
    if (coupling_line.length() < line_length) {
        coupling_line += std::string(line_length - coupling_line.length(), ' ');
    }
    
    // 提取最终J值（从第三行的后5位中提取）
    std::string final_J = coupling_line_raw.substr(coupling_line_raw.length() - 5, 4);
    final_J = trim(final_J);
    int final_double_J = j_to_double_j_fixed(final_J);
    
    // 首先为所有轨道填充子壳层信息（主量子数和kappa值）
    for (size_t i = 0; i < peel_subshells.size(); ++i) {
        const auto& subshell = peel_subshells[i];
        
        // 解析主量子数（数字部分）
        int n = 0;
        for (char c : subshell) {
            if (std::isdigit(c)) {
                n = n * 10 + (c - '0');
            } else {
                break;
            }
        }
        
        // 解析轨道类型（字母部分，包括可能的'-'）
        std::string orbital_part;
        for (char c : subshell) {
            if (!std::isdigit(c)) {
                orbital_part += c;
            }
        }
        if (orbital_part.back() != '-' && orbital_part.back() != ' ') {
            orbital_part += ' ';
        }
        
        int kappa = subshell_to_kappa_fixed(orbital_part);
        
        descriptor[i * 5] = static_cast<double>(n);      // 第1位：主量子数
        descriptor[i * 5 + 1] = static_cast<double>(kappa);  // 第2位：kappa值
    }
    
    // 将三行数据按每9个字符分块处理
    std::vector<std::string> subshell_List = chunk_string_fixed(subshells_line, 9);
    std::vector<std::string> middle_line_List = chunk_string_fixed(middle_line, 9);
    std::vector<std::string> coupling_line_List = chunk_string_fixed(coupling_line, 9);
    
    // 预创建子壳层到索引的映射
    static thread_local std::unordered_map<std::string, size_t> subshell_map;
    if (subshell_map.empty()) {
        subshell_map = create_subshell_index_map(peel_subshells);
    }
    
    // 预填充子壳层信息（主量子数和kappa值）
    for (size_t i = 0; i < peel_subshells.size(); ++i) {
        const auto& subshell = peel_subshells[i];
        
        // 快速解析主量子数
        int n = 0;
        size_t j = 0;
        while (j < subshell.size() && std::isdigit(subshell[j])) {
            n = n * 10 + (subshell[j] - '0');
            ++j;
        }
        
        // 快速解析轨道类型
        std::string orbital_part = subshell.substr(j);
        if (!orbital_part.empty() && orbital_part.back() != '-' && orbital_part.back() != ' ') {
            orbital_part += ' ';
        }
        
        int kappa = subshell_to_kappa_fixed(orbital_part);
        
        descriptor[i * 5] = static_cast<double>(n);
        descriptor[i * 5 + 1] = static_cast<double>(kappa);
    }
    
    // 使用固定大小的数组跟踪已占用轨道
    std::array<bool, 50> occupied_orbits{};
    
    // 遍历每个子壳层块，提取和处理信息
    for (size_t i = 0; i < subshell_List.size(); ++i) {
        const std::string& subshell_charges = subshell_List[i];
        const std::string& middle_line_item = (i < middle_line_List.size()) ? middle_line_List[i] : "";
        const std::string& coupling_line_item = (i < coupling_line_List.size()) ? coupling_line_List[i] : "";
        
        if (subshell_charges.length() < 8) continue;
        
        // 快速提取子壳层名称和电子数
        std::string subshell = trim(subshell_charges.substr(0, 5));
        int subshell_electron_num = (subshell_charges[6] - '0') * 10 + (subshell_charges[7] - '0');
        bool is_last = (i == subshell_List.size() - 1);
        
        // 快速查找索引
        auto it = subshell_map.find(subshell);
        if (it == subshell_map.end()) continue;
        
        size_t orbs_index = it->second;
        if (orbs_index >= peel_subshells.size()) continue;
        
        bool is_full = is_subshell_full_fixed(subshell, subshell_electron_num);
        occupied_orbits[orbs_index] = true;
        
        // 处理第二行数据
        int temp_middle_item = 0;
        if (!middle_line_item.empty()) {
            std::string middle_item = trim(middle_line_item);
            if (!middle_item.empty()) {
                size_t semicolon_pos = middle_item.rfind(';');
                if (semicolon_pos != std::string::npos) {
                    middle_item = trim(middle_item.substr(semicolon_pos + 1));
                }
                if (!middle_item.empty()) {
                    temp_middle_item = j_to_double_j_fixed(middle_item);
                    if (!is_full) temp_middle_item *= 2;
                }
            }
        }
        
        // 处理第三行数据
        int temp_coupling_item = 0;
        if (!coupling_line_item.empty()) {
            std::string coupling_item = trim(coupling_line_item);
            if (!coupling_item.empty()) {
                temp_coupling_item = j_to_double_j_fixed(coupling_item);
                if (!is_full) temp_coupling_item *= 2;
            }
        } else if (temp_middle_item != 0) {
            temp_coupling_item = temp_middle_item;
        }
        
        if (is_last) {
            temp_coupling_item = final_double_J * (is_full ? 1 : 2);
        }
        
        if (is_full) {
            temp_middle_item = 0;
            temp_coupling_item = 0;
        }
        
        // 直接填充描述符
        descriptor[orbs_index * 5 + 2] = static_cast<double>(subshell_electron_num);
        descriptor[orbs_index * 5 + 3] = static_cast<double>(temp_middle_item);
        descriptor[orbs_index * 5 + 4] = static_cast<double>(temp_coupling_item);
    }
    
    // 为未占用轨道填充最终J值的二倍
    for (size_t i = 0; i < peel_subshells.size(); ++i) {
        if (!occupied_orbits[i]) {
            descriptor[i * 5 + 4] = static_cast<double>(final_double_J * 2);
        }
    }
    
    return descriptor;
}


} // namespace csf