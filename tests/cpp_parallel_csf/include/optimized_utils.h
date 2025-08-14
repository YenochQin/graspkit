#pragma once

#include <unordered_map>
#include <string>
#include <string_view>
#include <array>
#include <algorithm>
#include <cctype>

namespace csf {

// 预编译的查找表
struct LookupTables {
    static const std::unordered_map<std::string, int>& get_max_electrons() {
        static const std::unordered_map<std::string, int> max_electrons = {
            {"s ", 2}, {"p-", 2}, {"p ", 4}, {"d-", 4},
            {"d ", 6}, {"f-", 6}, {"f ", 8}, {"g-", 8},
            {"g ", 10}, {"h-", 10}, {"h ", 12}, {"i-", 12},
            {"i ", 14}
        };
        return max_electrons;
    }
    
    static const std::unordered_map<std::string, int>& get_kappa_map() {
        static const std::unordered_map<std::string, int> kappa_map = {
            {"s ", -1}, {"p-", 1}, {"p ", -2}, {"d-", 2},
            {"d ", -3}, {"f-", 3}, {"f ", -4}, {"g-", 4},
            {"g ", -5}, {"h-", 5}, {"h ", -6}, {"i-", 6},
            {"i ", -7}
        };
        return kappa_map;
    }
    
    static const std::unordered_map<std::string, size_t>& create_subshell_index_map(
        const std::vector<std::string>& peel_subshells) {
        static thread_local std::unordered_map<std::string, size_t> map;
        if (map.empty()) {
            map.reserve(peel_subshells.size());
            for (size_t i = 0; i < peel_subshells.size(); ++i) {
                map[peel_subshells[i]] = i;
            }
        }
        return map;
    }
};

// 优化工具函数
struct StringUtils {
    static inline std::string_view trim_view(std::string_view str) {
        size_t first = str.find_first_not_of(" \t\n\r");
        if (first == std::string_view::npos) return "";
        size_t last = str.find_last_not_of(" \t\n\r");
        return str.substr(first, last - first + 1);
    }
    
    static inline int parse_integer(std::string_view str) {
        int result = 0;
        for (char c : str) {
            if (std::isdigit(c)) {
                result = result * 10 + (c - '0');
            }
        }
        return result;
    }
    
    static inline int parse_fraction(std::string_view str) {
        size_t slash_pos = str.find('/');
        if (slash_pos != std::string_view::npos) {
            return parse_integer(str.substr(0, slash_pos));
        } else {
            return parse_integer(str) * 2;
        }
    }
    
    static inline int extract_electrons(std::string_view chunk) {
        if (chunk.length() < 8) return 0;
        return (chunk[6] - '0') * 10 + (chunk[7] - '0');
    }
    
    static inline std::string_view extract_subshell(std::string_view chunk) {
        if (chunk.length() < 5) return "";
        return trim_view(chunk.substr(0, 5));
    }
};

} // namespace csf