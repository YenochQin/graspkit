#include "npy_writer.h"
#include "cnpy.h"
#include <algorithm>
#include <sstream>

namespace csf {

bool NPYWriter::write_descriptors(const std::string& filename,
                                  const DescriptorArray& descriptors,
                                  const std::vector<int>& labels) {
    try {
        if (descriptors.empty()) {
            return false;
        }
        
        // 获取描述符的维度
        size_t num_csfs = descriptors.size();
        size_t descriptor_size = descriptors[0].size();
        
        // 创建2D数组存储数据
        std::vector<double> data_2d;
        data_2d.reserve(num_csfs * descriptor_size);
        
        // 填充数据（跳过第一列的CSF_index_J，只存储纯描述符数值）
        for (const auto& descriptor : descriptors) {
            for (double value : descriptor) {
                data_2d.push_back(value);
            }
        }
        
        // 创建形状向量
        std::vector<size_t> shape = {num_csfs, descriptor_size};
        
        // 写入.npy文件
        cnpy::npy_save(filename, data_2d.data(), shape, "w");
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

std::string NPYWriter::generate_default_output_filename(const std::string& input_filename) {
    if (input_filename.empty()) {
        return "descriptors.npy";
    }
    
    // 获取文件名部分（不含路径）
    std::string filename = get_filename_from_path(input_filename);
    
    // 移除扩展名
    std::string basename = remove_extension(filename);
    
    // 添加_descriptors.npy后缀
    return basename + "_descriptors.npy";
}

std::string NPYWriter::get_file_extension(const std::string& filename) {
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos != std::string::npos) {
        return filename.substr(dot_pos);
    }
    return "";
}

std::string NPYWriter::remove_extension(const std::string& filename) {
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos != std::string::npos) {
        return filename.substr(0, dot_pos);
    }
    return filename;
}

std::string NPYWriter::get_filename_from_path(const std::string& filepath) {
    // 使用字符串处理
    size_t pos = filepath.find_last_of("/\\");
    if (pos != std::string::npos) {
        return filepath.substr(pos + 1);
    }
    return filepath;
}

} // namespace csf