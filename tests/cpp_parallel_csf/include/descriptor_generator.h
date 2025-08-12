#pragma once

#include "csf_types.h"
#include <vector>
#include <cmath>

namespace csf {

class DescriptorGenerator {
public:
    // 基础描述符生成（每个轨道3个数值）
    static Descriptor generate_basic_descriptor(const CSFData& csf, 
                                               const PeelSubshells& peel_subshells);
    
    // 扩展描述符生成（每个轨道5个数值，包含子壳层信息）
    static Descriptor generate_extended_descriptor(const CSFData& csf,
                                                  const PeelSubshells& peel_subshells);
    
    // 批量生成描述符
    static DescriptorArray generate_descriptors_parallel(const CSFFileData& csf_data,
                                                        bool include_subshell_info = false);

private:
};

} // namespace csf