#include <iostream>
#include <vector>
#include <chrono>
#include "csf_parser.h"

using namespace csf;

int main() {
    std::cout << "=== CSF Parser Test ===\n";
    
    try {
        // 创建GRASP格式测试数据
        std::vector<std::string> test_data = {
            "Core subshells:",
            "1s 2s 2p- 2p 3s 3p- 3p 3d- 3d 4s 4p- 4p 4d- 4d 4f- 4f 5s 5p-",
            "Peel subshells:",
            "5s 5p- 5p 5d- 5d 5f- 5f 6s 6p- 6p",
            "CSF(s):",
            "  5s( 2)  5p-( 2)  5p( 4)  5d-( 4)  5d( 6)",
            "                     1/2      1      3/2      2",
            "                          2-",
            "  5s( 2)  5p-( 2)  5p( 3)  5d-( 4)  5d( 6)",
            "                     1/2      1      3/2      2",
            "                          3/2-"
        };
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 测试解析
        auto csf_data = CSFParser::parse_lines(test_data);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Parsing completed in " << duration.count() << " microseconds\n";
        std::cout << "Peel subshells: ";
        for (const auto& subshell : csf_data.peel_subshells) {
            std::cout << subshell << " ";
        }
        std::cout << "\n";
        
        std::cout << "Blocks: " << csf_data.csf_blocks.size() << "\n";
        std::cout << "Total CSFs: " << csf_data.total_csfs() << "\n";
        
        // 验证数据
        if (!csf_data.csf_blocks.empty()) {
            const auto& first_block = csf_data.csf_blocks[0];
            if (!first_block.empty()) {
                const auto& first_csf = first_block[0];
                std::cout << "First CSF:\n";
                std::cout << "  Subshell: " << first_csf.subshell_line << "\n";
                std::cout << "  Intermediate: " << first_csf.intermediate_line << "\n";
                std::cout << "  Coupling: " << first_csf.coupling_line << "\n";
            }
        }
        
        std::cout << "✓ Parser test passed\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Parser test failed: " << e.what() << "\n";
        return 1;
    }
}