#pragma once

#include<cstdio>
#include<string>
#include<vector>
#include<stdint.h>
#include<stdexcept>
#include<cstdlib>
#include<algorithm>

namespace cnpy {

char BigEndianTest() {
    int x = 1;
    return (((char *)&x)[0]) ? '<' : '>';
}

char map_type(const std::type_info& t) {
    if(t == typeid(float) ) return 'f';
    if(t == typeid(double) ) return 'f';
    if(t == typeid(long double) ) return 'f';
    if(t == typeid(int) ) return 'i';
    if(t == typeid(unsigned int) ) return 'u';
    if(t == typeid(long long) ) return 'i';
    if(t == typeid(unsigned long long) ) return 'u';
    if(t == typeid(long) ) return 'i';
    if(t == typeid(unsigned long) ) return 'u';
    if(t == typeid(short) ) return 'i';
    if(t == typeid(unsigned short) ) return 'u';
    if(t == typeid(char) ) return 'i';
    if(t == typeid(unsigned char) ) return 'u';
    if(t == typeid(bool) ) return 'b';
    return '?';
}

std::string create_npy_header(const std::vector<size_t>& shape, size_t word_size, bool fortran_order) {
    std::string dict;
    dict += "{'descr': '<f8', 'fortran_order': ";
    dict += (fortran_order) ? "True" : "False";
    dict += ", 'shape': (";
    
    for(size_t i = 0;i < shape.size();i++) {
        dict += std::to_string(shape[i]);
        if(i < shape.size()-1) dict += ", ";
    }
    dict += "), }";
    
    int padlen = 64 - (dict.length() + 10) % 64;
    if(padlen < 32) padlen += 64;
    
    std::string header;
    header += (char)0x93;
    header += "NUMPY";
    header += (char)0x01;
    header += (char)0x00;
    
    uint16_t header_len = dict.length() + padlen + 1;
    header += (char)(header_len & 0xFF);
    header += (char)((header_len >> 8) & 0xFF);
    
    header += dict;
    for(int i = 0; i < padlen; i++) header += (char)0x20;
    header += (char)0x0a;
    
    return header;
}

template<typename T>
void npy_save(std::string fname, const T* data, const std::vector<size_t> shape, std::string mode = "w") {
    FILE* fp = NULL;
    
    fp = fopen(fname.c_str(),"wb");
    if(!fp) throw std::runtime_error("npy_save: Unable to open file " + fname);
    
    std::string header = create_npy_header(shape, sizeof(T), true);
    fwrite(header.c_str(), 1, header.size(), fp);
    
    size_t total_elements = 1;
    for(auto s : shape) total_elements *= s;
    
    fwrite(data, sizeof(T), total_elements, fp);
    fclose(fp);
}

// 特化模板用于double类型
template void npy_save<double>(std::string fname, const double* data, const std::vector<size_t> shape, std::string mode);

}