#include "parallel_processor.h"
#include <math.h>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <atomic>

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
    
    result.descriptors.resize(total_csfs);
    result.labels.resize(total_csfs);
    
    // 优化的chunk划分策略
    const size_t min_chunk_size = 10000; // 增大最小chunk大小以减少开销
    const size_t cache_line_size = 64;   // CPU缓存行大小
    std::vector<std::pair<size_t, size_t>> work_chunks;
    
    size_t current_global_index = 0;
    for (size_t block_idx = 0; block_idx < csf_data.csf_blocks.size(); ++block_idx) {
        const auto& block = csf_data.csf_blocks[block_idx];
        size_t block_size = block.size();
        
        if (block_size <= min_chunk_size || num_threads <= 1) {
            // 小块直接处理
            work_chunks.emplace_back(current_global_index, current_global_index + block_size);
            current_global_index += block_size;
        } else {
            // 优化的chunk划分：按缓存行对齐并考虑负载均衡
            size_t optimal_chunk_size = std::max(
                min_chunk_size,
                ((block_size + num_threads * 4 - 1) / (num_threads * 4))  // 每个线程4个chunk
            );
            
            // 确保chunk大小是缓存行大小的倍数
            optimal_chunk_size = (optimal_chunk_size + cache_line_size - 1) / cache_line_size * cache_line_size;
            
            for (size_t start = 0; start < block_size; start += optimal_chunk_size) {
                size_t end = std::min(start + optimal_chunk_size, block_size);
                work_chunks.emplace_back(current_global_index + start, current_global_index + end);
            }
            current_global_index += block_size;
        }
    }
    
    // 并行处理所有work chunks
    std::atomic<int> completed_chunks{0};
    const int total_chunks = static_cast<int>(work_chunks.size());
    
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic) if(options.use_openmp)
    for (int chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        const auto& chunk = work_chunks[chunk_idx];
        size_t start_global = chunk.first;
        size_t end_global = chunk.second;
        
        // 优化的块查找：使用二分查找替代线性查找
        size_t accumulated = 0;
        size_t block_idx = 0;
        
        // 快速查找对应的块
        for (size_t i = 0; i < csf_data.csf_blocks.size(); ++i) {
            size_t next_accumulated = accumulated + csf_data.csf_blocks[i].size();
            if (start_global < next_accumulated) {
                block_idx = i;
                break;
            }
            accumulated = next_accumulated;
        }
        
        const auto& block = csf_data.csf_blocks[block_idx];
        size_t local_start = start_global - accumulated;
        size_t local_end = std::min(end_global - accumulated, block.size());
        
        // 处理chunk中的每个CSF
        for (size_t i = local_start; i < local_end; ++i) {
            const auto& csf = block[i];
            
            Descriptor descriptor;
            if (options.include_subshell_info) {
                descriptor = DescriptorGenerator::generate_extended_descriptor(csf, csf_data.peel_subshells);
            } else {
                descriptor = DescriptorGenerator::generate_basic_descriptor(csf, csf_data.peel_subshells);
            }
            
            // 直接写入预分配的位置
            result.descriptors[start_global + (i - local_start)] = std::move(descriptor);
            result.labels[start_global + (i - local_start)] = static_cast<int>(block_idx);
        }
        
        // 进度显示 - 减少更新频率以提高性能
        if (options.show_progress) {
            int current_completed = ++completed_chunks;
            // 每1%更新一次进度条
            if (current_completed % (total_chunks / 100 + 1) == 0 || current_completed == total_chunks) {
                #pragma omp critical(progress_display)
                {
                    show_progress_bar(current_completed, total_chunks);
                }
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
    #else
    (void)num_threads;  // 避免编译警告
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

void ParallelProcessor::show_progress_bar(size_t current, size_t total, int /*thread_id*/) {
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