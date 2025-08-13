#pragma once

#include "csf_types.h"
#include <string>
#include <vector>

namespace csf {

class NPYWriter {
public:
    // 写入描述符到.npy文件
    static bool write_descriptors(const std::string& filename,
                                  const DescriptorArray& descriptors,
                                  const std::vector<int>& labels = std::vector<int>());
    
    // 生成默认输出文件名（基于输入文件名）
    static std::string generate_default_output_filename(const std::string& input_filename);
    
    // 获取文件扩展名
    static std::string get_file_extension(const std::string& filename);
    
    // 移除文件扩展名
    static std::string remove_extension(const std::string& filename);

private:
    // 辅助函数：从文件路径中提取文件名
    static std::string get_filename_from_path(const std::string& filepath);
};

} // namespace csf