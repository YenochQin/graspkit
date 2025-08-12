#include "parallel_processor.h"
#include <math.h>
#include <iomanip>
#include <iostream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace csf {

ParallelProcessor::ProcessingResult ParallelProcessor::process_csf_data(
    const CSFFileData& csf_data, const ProcessingOptions& options) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ProcessingResult result;
    result.total_csfs_processed = csf_data.total_csfs();
    
    // 确定线程数
    int num_threads = options.num_threads;
    if (num_threads <= 0) {
        num_threads = get_optimal_thread_count();
    }
    
    #ifdef _OPENMP
    if (options.use_openmp) {
        omp_set_num_threads(num_threads);
    }
    #endif
    
    result.threads_used = num_threads;
    
    // 计算总CSF数量
    size_t total_csfs = 0;
    for (const auto& block : csf_data.csf_blocks) {
        total_csfs += block.size();
    }
    
    result.descriptors.reserve(total_csfs);
    result.labels.reserve(total_csfs);
    
    // 并行处理每个块
    #pragma omp parallel for num_threads(num_threads) if(options.use_openmp)
    for (int block_idx = 0; block_idx < static_cast<int>(csf_data.csf_blocks.size()); ++block_idx) {
        const auto& block = csf_data.csf_blocks[block_idx];
        
        #pragma omp critical
        {
            if (options.show_progress) {
                show_progress_bar(block_idx + 1, csf_data.csf_blocks.size());
            }
        }
        
        // 处理块中的每个CSF
        for (size_t csf_idx = 0; csf_idx < block.size(); ++csf_idx) {
            const auto& csf = block[csf_idx];
            
            Descriptor descriptor;
            if (options.include_subshell_info) {
                descriptor = DescriptorGenerator::generate_extended_descriptor(csf, csf_data.peel_subshells);
            } else {
                descriptor = DescriptorGenerator::generate_basic_descriptor(csf, csf_data.peel_subshells);
            }
            
            #pragma omp critical
            {
                result.descriptors.push_back(std::move(descriptor));
                result.labels.push_back(block_idx);
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.processing_time_ms = duration.count() / 1000.0;
    
    if (options.show_progress) {
        std::cout << "\n";
    }
    
    return result;
}

void ParallelProcessor::set_num_threads(int num_threads) {
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif
    
    // 设置线程池大小（如果使用std::thread）
    // 这里可以添加自定义线程池逻辑
}

int ParallelProcessor::get_optimal_thread_count() {
    #ifdef _OPENMP
    return omp_get_max_threads();
    #else
    return std::thread::hardware_concurrency();
    #endif
}

void ParallelProcessor::show_progress_bar(size_t current, size_t total, int thread_id) {
    static std::mutex progress_mutex;
    std::lock_guard<std::mutex> lock(progress_mutex);
    
    if (total == 0) return;
    
    const int bar_width = 50;
    float progress = static_cast<float>(current) / total;
    
    std::cout << "\r[";
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "%";
    std::cout.flush();
    
    if (current == total) {
        std::cout << std::endl;
    }
}

std::string ParallelProcessor::format_duration(double milliseconds) {
    if (milliseconds < 1000) {
        return std::to_string(static_cast<int>(milliseconds)) + " ms";
    } else if (milliseconds < 60000) {
        return std::to_string(milliseconds / 1000.0) + " s";
    } else {
        return std::to_string(milliseconds / 60000.0) + " min";
    }
}

} // namespace csf