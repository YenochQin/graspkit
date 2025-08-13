#include "hdf5_writer.h"
#include <H5Cpp.h>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace csf {

bool HDF5Writer::write_descriptors(const std::string& filename,
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
        
        try {
            // 创建HDF5文件
            H5::H5File file(filename, H5F_ACC_TRUNC);
            
            // 创建数据空间
            hsize_t dims[2] = {num_csfs, descriptor_size};
            H5::DataSpace dataspace(2, dims);
            
            // 创建数据集
            H5::DataSet dataset = file.createDataSet("descriptors", H5::PredType::NATIVE_DOUBLE, dataspace);
            
            // 写入数据
            dataset.write(data_2d.data(), H5::PredType::NATIVE_DOUBLE);
            
            // 如果有标签数据，也写入
            if (!labels.empty() && labels.size() == num_csfs) {
                hsize_t label_dims[1] = {num_csfs};
                H5::DataSpace label_dataspace(1, label_dims);
                H5::DataSet label_dataset = file.createDataSet("labels", H5::PredType::NATIVE_INT, label_dataspace);
                label_dataset.write(labels.data(), H5::PredType::NATIVE_INT);
            }
            
            // 添加属性信息
            H5::Attribute attr = dataset.createAttribute("rows", H5::PredType::NATIVE_ULLONG, H5::DataSpace());
            attr.write(H5::PredType::NATIVE_ULLONG, &num_csfs);
            
            attr = dataset.createAttribute("cols", H5::PredType::NATIVE_ULLONG, H5::DataSpace());
            attr.write(H5::PredType::NATIVE_ULLONG, &descriptor_size);
            
            file.close();
            return true;
            
        } catch (const H5::Exception& e) {
            std::cerr << "HDF5 Error: " << e.getDetailMsg() << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

std::string HDF5Writer::generate_default_output_filename(const std::string& input_filename) {
    if (input_filename.empty()) {
        return "descriptors.h5";
    }
    
    // 获取文件名部分（不含路径）
    std::string filename = get_filename_from_path(input_filename);
    
    // 移除扩展名
    std::string basename = remove_extension(filename);
    
    // 添加_descriptors.h5后缀
    return basename + "_descriptors.h5";
}

std::string HDF5Writer::get_file_extension(const std::string& filename) {
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos != std::string::npos) {
        return filename.substr(dot_pos);
    }
    return "";
}

std::string HDF5Writer::remove_extension(const std::string& filename) {
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos != std::string::npos) {
        return filename.substr(0, dot_pos);
    }
    return filename;
}

std::string HDF5Writer::get_filename_from_path(const std::string& filepath) {
    // 使用字符串处理
    size_t pos = filepath.find_last_of("/\\");
    if (pos != std::string::npos) {
        return filepath.substr(pos + 1);
    }
    return filepath;
}

} // namespace csf