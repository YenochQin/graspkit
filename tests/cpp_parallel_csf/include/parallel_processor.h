#pragma once

#include "csf_types.h"
#include "descriptor_generator.h"
#include <vector>
#include <future>
#include <thread>
#include <chrono>

namespace csf {

struct ProcessingOptions {
    bool include_subshell_info = false;
    bool use_openmp = true;
    int num_threads = 0; // 0表示自动检测
    bool show_progress = true;
    
    ProcessingOptions() = default;
};

class ParallelProcessor {
    
public:
    struct ProcessingResult {
        DescriptorArray descriptors;
        std::vector<int> labels; // 块标签
        double processing_time_ms;
        size_t total_csfs_processed;
        size_t threads_used;
    };
    
    static ProcessingResult process_csf_data(const CSFFileData& csf_data,
                                           const ProcessingOptions& options = ProcessingOptions());
    
    static void set_num_threads(int num_threads);
    
    static int get_optimal_thread_count();

private:
    static DescriptorArray process_block_parallel(const std::vector<CSFData>& block,
                                                 const PeelSubshells& peel_subshells,
                                                 bool include_subshell_info,
                                                 int thread_id);
    
    static void show_progress_bar(size_t current, size_t total, int thread_id = 0);
    
    static std::string format_duration(double milliseconds);
};

} // namespace csf