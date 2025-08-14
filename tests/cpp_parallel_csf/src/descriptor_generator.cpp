#include "descriptor_generator.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <cctype>
#include <unordered_map>

namespace csf {

// 辅助函数：清理字符串
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

// 辅助函数：按固定长度分块
static std::vector<std::string> chunk_string_fixed(const std::string& str, size_t chunk_size) {
    std::vector<std::string> chunks;
    for (size_t i = 0; i < str.length(); i += chunk_size) {
        chunks.push_back(str.substr(i, std::min(chunk_size, str.length() - i)));
    }
    return chunks;
}

// 辅助函数：将J字符串转换为2J值
static int j_to_double_j_fixed(const std::string& j_str) {
    std::string clean_j = trim(j_str);
    size_t slash_pos = clean_j.find('/');
    if (slash_pos != std::string::npos) {
        // 半整数情况，如 "3/2"
        int numerator = std::stoi(clean_j.substr(0, slash_pos));
        return numerator;
    } else {
        // 整数情况，如 "2"
        int value = std::stoi(clean_j);
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
    
    // 初始化描述符数组和已占用轨道索引列表
    std::vector<int> orbs_occupied_indices;
    
    // 遍历每个子壳层块，提取和处理信息
    for (size_t i = 0; i < subshell_List.size(); ++i) {
        const std::string& subshell_charges = subshell_List[i];
        const std::string& middle_line_item = (i < middle_line_List.size()) ? middle_line_List[i] : "";
        const std::string& coupling_line_item = (i < coupling_line_List.size()) ? coupling_line_List[i] : "";
        
        if (subshell_charges.length() < 8) continue;
        
        // 提取子壳层名称和电子数
        std::string subshell = subshell_charges.substr(0, 5);
        subshell = trim(subshell);
        int subshell_electron_num = std::stoi(subshell_charges.substr(6, 2));
        bool is_last = (i == subshell_List.size() - 1);
        
        // 处理第二行数据（中间J耦合值）
        int temp_middle_item = 0;
        std::string middle_item = trim(middle_line_item);
        if (!middle_item.empty() && middle_item.find_first_not_of(" \t") != std::string::npos) {
            // 如果有分号分隔的多个值，取最后一个
            size_t semicolon_pos = middle_item.find_last_of(';');
            if (semicolon_pos != std::string::npos) {
                middle_item = middle_item.substr(semicolon_pos + 1);
            }
            middle_item = trim(middle_item);
            if (!middle_item.empty()) {
                temp_middle_item = j_to_double_j_fixed(middle_item);
            }
        }
        
        // 处理第三行数据（耦合J值）
        int temp_coupling_item = 0;
        std::string coupling_item = trim(coupling_line_item);
        if (!coupling_item.empty() && coupling_item.find_first_not_of(" \t") != std::string::npos) {
            coupling_item = trim(coupling_item);
            if (!coupling_item.empty()) {
                temp_coupling_item = j_to_double_j_fixed(coupling_item);
            }
        } else if (temp_middle_item != 0) {
            temp_coupling_item = temp_middle_item;  // 使用第二行的值
        }

        // 特殊处理：如果是最后一个子壳层，使用最终J值
        if (is_last) {
            temp_coupling_item = final_double_J;
        }
        
        // 在轨道列表中查找当前子壳层的索引
        auto it = std::find(peel_subshells.begin(), peel_subshells.end(), subshell);
        if (it == peel_subshells.end()) {
            continue;  // 跳过未找到的子壳层
        }
        
        size_t orbs_index = std::distance(peel_subshells.begin(), it);
        size_t descriptor_index = orbs_index * 3;
        
        // 记录已占用轨道并填充描述符数组
        orbs_occupied_indices.push_back(orbs_index);
        descriptor[descriptor_index] = static_cast<double>(subshell_electron_num);  // 电子数
        descriptor[descriptor_index + 1] = static_cast<double>(temp_middle_item);   // 中间J值
        descriptor[descriptor_index + 2] = static_cast<double>(temp_coupling_item); // 耦合J值
    }
    
    // 处理未占用的轨道（使用集合运算找到差集）
    std::vector<int> all_orbs_indices(peel_subshells.size());
    for (size_t i = 0; i < peel_subshells.size(); ++i) {
        all_orbs_indices[i] = static_cast<int>(i);
    }
    
    std::vector<int> remaining_orbs_indices;
    for (int index : all_orbs_indices) {
        if (std::find(orbs_occupied_indices.begin(), orbs_occupied_indices.end(), index) == orbs_occupied_indices.end()) {
            remaining_orbs_indices.push_back(index);
        }
    }
    
    // 为未占用轨道填充最终J值
    for (int index : remaining_orbs_indices) {
        descriptor[index * 3 + 2] = static_cast<double>(final_double_J);
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
    
    // 初始化描述符数组和已占用轨道索引列表
    std::vector<int> orbs_occupied_indices;
    
    // 预填充子壳层信息（主量子数和kappa值）
    for (size_t i = 0; i < peel_subshells.size(); ++i) {
        const auto& subshell = peel_subshells[i];
        
        // 解析主量子数
        int n = 0;
        size_t j = 0;
        while (j < subshell.size() && std::isdigit(subshell[j])) {
            n = n * 10 + (subshell[j] - '0');
            ++j;
        }
        
        // 解析轨道类型
        std::string orbital_part = subshell.substr(j);
        if (!orbital_part.empty() && orbital_part.back() != '-' && orbital_part.back() != ' ') {
            orbital_part += ' ';
        }
        
        int kappa = subshell_to_kappa_fixed(orbital_part);
        
        descriptor[i * 5] = static_cast<double>(n);
        descriptor[i * 5 + 1] = static_cast<double>(kappa);
    }
    
    // 遍历每个子壳层块，提取和处理信息
    for (size_t i = 0; i < subshell_List.size(); ++i) {
        const std::string& subshell_charges = subshell_List[i];
        const std::string& middle_line_item = (i < middle_line_List.size()) ? middle_line_List[i] : "";
        const std::string& coupling_line_item = (i < coupling_line_List.size()) ? coupling_line_List[i] : "";
        
        if (subshell_charges.length() < 8) continue;
        
        // 提取子壳层名称和电子数
        std::string subshell = subshell_charges.substr(0, 5);
        subshell = trim(subshell);
        int subshell_electron_num = std::stoi(subshell_charges.substr(6, 2));
        bool is_last = (i == subshell_List.size() - 1);
        
        // 判断轨道是否填满
        bool is_full = is_subshell_full_fixed(subshell, subshell_electron_num);
        
        // 处理第二行数据（中间J耦合值）
        int temp_middle_item = 0;
        std::string middle_item = trim(middle_line_item);
        if (!middle_item.empty() && middle_item.find_first_not_of(" \t") != std::string::npos) {
            // 如果有分号分隔的多个值，取最后一个
            size_t semicolon_pos = middle_item.find_last_of(';');
            if (semicolon_pos != std::string::npos) {
                middle_item = middle_item.substr(semicolon_pos + 1);
            }
            middle_item = trim(middle_item);
            if (!middle_item.empty()) {
                temp_middle_item = j_to_double_j_fixed(middle_item);
                // 未填满轨道J值乘以2增强特征
                if (!is_full) {
                    temp_middle_item *= 2;
                }
            }
        }
        
        // 处理第三行数据（耦合J值）
        int temp_coupling_item = 0;
        std::string coupling_item = trim(coupling_line_item);
        if (!coupling_item.empty() && coupling_item.find_first_not_of(" \t") != std::string::npos) {
            coupling_item = trim(coupling_item);
            if (!coupling_item.empty()) {
                temp_coupling_item = j_to_double_j_fixed(coupling_item);
                if (!is_full) {
                    temp_coupling_item *= 2;
                }
            }
        } else if (temp_middle_item != 0) {
            temp_coupling_item = temp_middle_item;  // 使用第二行的值
        }
        
        // 特殊处理：如果是最后一个子壳层，使用最终J值
        if (is_last) {
            temp_coupling_item = final_double_J * (is_full ? 1 : 2);
        }
        
        // 填满的轨道J值设为0
        if (is_full) {
            temp_middle_item = 0;
            temp_coupling_item = 0;
        }
        
        // 在轨道列表中查找当前子壳层的索引
        auto it = std::find(peel_subshells.begin(), peel_subshells.end(), subshell);
        if (it == peel_subshells.end()) {
            continue;  // 跳过未找到的子壳层
        }
        
        size_t orbs_index = std::distance(peel_subshells.begin(), it);
        
        // 记录已占用轨道并填充描述符数组
        orbs_occupied_indices.push_back(orbs_index);
        descriptor[orbs_index * 5 + 2] = static_cast<double>(subshell_electron_num);  // 第3位：电子数
        descriptor[orbs_index * 5 + 3] = static_cast<double>(temp_middle_item);       // 第4位：中间J值
        descriptor[orbs_index * 5 + 4] = static_cast<double>(temp_coupling_item);     // 第5位：耦合J值
    }
    
    // 处理未占用的轨道（使用集合运算找到差集）
    std::vector<int> all_orbs_indices(peel_subshells.size());
    for (size_t i = 0; i < peel_subshells.size(); ++i) {
        all_orbs_indices[i] = static_cast<int>(i);
    }
    
    std::vector<int> remaining_orbs_indices;
    for (int index : all_orbs_indices) {
        if (std::find(orbs_occupied_indices.begin(), orbs_occupied_indices.end(), index) == orbs_occupied_indices.end()) {
            remaining_orbs_indices.push_back(index);
        }
    }
    
    // 为未占用轨道填充最终J值的二倍
    for (int index : remaining_orbs_indices) {
        descriptor[index * 5 + 4] = static_cast<double>(final_double_J * 2);
    }
    
    return descriptor;
}


} // namespace csf