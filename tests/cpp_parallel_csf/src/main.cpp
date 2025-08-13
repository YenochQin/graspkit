#include <iostream>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include "csf_parser.h"
#include "parallel_processor.h"
#include "descriptor_generator.h"
#include "hdf5_writer.h"

using namespace csf;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options] input_file\n";
    std::cout << "Options:\n";
    std::cout << "  -e, --extended    Use extended descriptor format (5 values per orbital)\n";
    std::cout << "  -t, --threads N   Use N threads (default: auto-detect)\n";
    std::cout << "  -o, --output FILE Output file (.h5 format) (default: input_prefix_descriptors.h5)\n";
    std::cout << "  -q, --quiet       Suppress progress output\n";
    std::cout << "  -h, --help        Show this help message\n";
}

void print_performance_stats(const ParallelProcessor::ProcessingResult& result) {
    std::cout << "\n=== Performance Statistics ===\n";
    std::cout << "CSFs processed: " << result.total_csfs_processed << "\n";
    std::cout << "Processing time: " << result.processing_time_ms << " ms\n";
    std::cout << "Throughput: " << (result.total_csfs_processed * 1000.0 / result.processing_time_ms) << " CSFs/sec\n";
    std::cout << "Threads used: " << result.threads_used << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    // 解析命令行参数
    std::string input_file;
    std::string output_file;
    bool use_extended = false;
    bool quiet = false;
    int num_threads = 0;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-e" || arg == "--extended") {
            use_extended = true;
        } else if (arg == "-q" || arg == "--quiet") {
            quiet = true;
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                num_threads = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: Missing thread count after " << arg << "\n";
                return 1;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            } else {
                std::cerr << "Error: Missing output file after " << arg << "\n";
                return 1;
            }
        } else if (arg[0] != '-') {
            input_file = arg;
        }
    }
    
    if (input_file.empty()) {
        std::cerr << "Error: No input file specified\n";
        return 1;
    }
    
    try {
        // 解析CSF文件
        if (!quiet) {
            std::cout << "Parsing CSF file: " << input_file << "\n";
        }
        
        auto csf_data = CSFParser::parse_file(input_file);
        
        if (!quiet) {
            std::cout << "Found " << csf_data.total_csfs() << " CSFs in " << csf_data.csf_blocks.size() << " blocks\n";
            std::cout << "Peel subshells: ";
            for (const auto& subshell : csf_data.peel_subshells) {
                std::cout << subshell << " ";
            }
            std::cout << "\n\n";
        }
        
        // 设置处理选项
        ProcessingOptions options;
        options.include_subshell_info = use_extended;
        options.num_threads = num_threads;
        options.show_progress = !quiet;
        
        // 并行处理CSF数据
        if (!quiet) {
            std::cout << "Processing CSF data...\n";
        }
        
        auto result = ParallelProcessor::process_csf_data(csf_data, options);
        
        // 如果没有指定输出文件，生成默认文件名
        if (output_file.empty()) {
            output_file = HDF5Writer::generate_default_output_filename(input_file);
        }
        
        // 确保输出文件有.h5扩展名
        if (HDF5Writer::get_file_extension(output_file) != ".h5") {
            output_file += ".h5";
        }
        
        // 写入HDF5文件
        if (!quiet) {
            std::cout << "Writing descriptors to: " << output_file << "\n";
        }
        
        bool success = HDF5Writer::write_descriptors(output_file, result.descriptors, result.labels);
        if (!success) {
            std::cerr << "Error: Failed to write output file " << output_file << "\n";
            return 1;
        }
        
        if (!quiet) {
            print_performance_stats(result);
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}