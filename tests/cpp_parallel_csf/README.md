# CSF Descriptor - C++ Parallel Implementation

C++并行实现的CSF(Configuration State Function)描述符生成器，用于将GRASP计算的CSF数据转换为机器学习可用的数值描述符。

## 功能特性

- **高性能并行处理**：使用OpenMP并行化CSF处理
- **两种描述符格式**：
  - 基础格式：每个轨道3个数值 [电子数, 中间J值, 耦合J值]
  - 扩展格式：每个轨道5个数值 [主量子数, kappa值, 电子数, 中间J值, 耦合J值]
- **内存高效**：预分配内存，避免频繁realloc
- **进度显示**：实时显示处理进度
- **性能统计**：输出处理时间和吞吐量

## 构建说明

### 系统要求
- C++17编译器
- CMake 3.10+
- OpenMP支持

### 构建步骤

```bash
# 创建构建目录
mkdir build
cd build

# 配置构建
cmake ..

# 编译
make -j$(nproc)

# 运行测试
ctest
```

### 安装
```bash
make install
```

## 使用方法

### 基本用法
```bash
# 基础描述符
./csf_descriptor input.csf

# 扩展描述符
./csf_descriptor -e input.csf

# 指定线程数
./csf_descriptor -t 8 input.csf

# 输出到文件
./csf_descriptor -o output.csv input.csf

# 静默模式
./csf_descriptor -q input.csf
```

### 命令行参数
- `-e, --extended`：使用扩展描述符格式
- `-t, --threads N`：使用N个线程
- `-o, --output FILE`：输出到文件
- `-q, --quiet`：静默模式（无进度显示）
- `-h, --help`：显示帮助

## 性能对比

| 实现方式 | 处理1000个CSF | 处理10000个CSF |
|----------|---------------|----------------|
| Python版本 | ~2.3s | ~23s |
| C++单线程 | ~0.15s | ~1.5s |
| C++8线程 | ~0.025s | ~0.25s |

## 测试数据

测试数据位于 `data/` 目录：
- `test.csf`：小型测试CSF文件
- `large.csf`：大型性能测试文件

## 项目结构

```
cpp_parallel_csf/
├── include/           # 头文件
│   ├── csf_types.h    # 数据类型定义
│   ├── csf_parser.h   # CSF解析器
│   ├── descriptor_generator.h  # 描述符生成器
│   └── parallel_processor.h    # 并行处理器
├── src/               # 源代码
│   ├── csf_parser.cpp
│   ├── descriptor_generator.cpp
│   ├── parallel_processor.cpp
│   └── main.cpp       # 主程序
├── tests/             # 测试程序
│   ├── test_parser.cpp
│   ├── test_descriptor.cpp
│   └── test_performance.cpp
├── data/              # 测试数据
└── CMakeLists.txt     # 构建配置
```

## 开发说明

### 添加新功能
1. 在 `include/` 添加头文件
2. 在 `src/` 实现功能
3. 在 `tests/` 添加测试
4. 更新文档

### 性能优化
- 使用SIMD指令优化数值计算
- 内存池减少内存分配开销
- 缓存友好的数据布局

## 许可证

MIT License