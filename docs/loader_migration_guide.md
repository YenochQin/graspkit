# graspkit 加载器迁移指南

本文档说明如何从旧的 `GraspFileLoad` 类迁移到新的专用加载器类。

## 目录

- [概述](#概述)
- [为什么迁移](#为什么迁移)
- [新旧 API 对比](#新旧-api-对比)
- [迁移步骤](#迁移步骤)
- [详细示例](#详细示例)
- [API 参考](#api-参考)

---

## 概述

### 旧架构（已弃用）

```python
from graspkit.data_IO import GraspFileLoad

# 所有功能集中在一个类中
loader = GraspFileLoad(config)
rwfn = loader.load_rwfn_bin("path/to/file.w")
mix = loader.load_rmix_data("path/to/file.m")
csf = loader.load_csf_data("path/to/file.c")
```

### 新架构（推荐）

```python
from graspkit.data_IO.loaders import (
    RWFNFileLoader,
    MixCoefLoader,
    CSFLoader,
    EnergyFileLoader,
    TransitionLoader,
    LSJCompLoader,
)

# 每种文件类型有专用的加载器
rwfn_loader = RWFNFileLoader("path/to/file.w")
rwfn_df = rwfn_loader.load()

mix_loader = MixCoefLoader("path/to/file.m")
mix_data = mix_loader.load()
```

---

## 为什么迁移

### 新架构的优势

1. **单一职责原则**：每个加载器只负责一种文件类型
2. **更好的类型安全**：使用泛型和明确的返回类型
3. **正确的 Fortran 二进制支持**：正确处理记录标记
4. **更清晰的 API**：方法名和返回值更明确
5. **更容易测试**：每个加载器可以独立测试
6. **更好的错误处理**：具体的错误消息和验证

---

## 新旧 API 对比

### CSF 文件加载

#### 旧方式
```python
from graspkit.data_IO import GraspFileLoad

loader = GraspFileLoad(config)
csf_data = loader.load_csf_data("path/to/file.c")
# 返回格式不明确，需要查看源码
```

#### 新方式
```python
from graspkit.data_IO.loaders import CSFLoader

csf_loader = CSFLoader("path/to/file.c")
csf_data = csf_loader.load()
# 返回类型明确：CSFs

# 访问特定块
block_data = csf_loader.get_block_data(block_idx=0)
total_count = csf_loader.get_total_csfs_count()
```

### 径向波函数加载

#### 旧方式
```python
from graspkit.data_IO import GraspFileLoad

loader = GraspFileLoad(config)
rwfn_df = GraspFileLoad.load_rwfn_bin("path/to/file.w")
# 静态方法，代码有 Fortran 记录处理 bug
```

#### 新方式
```python
from graspkit.data_IO.loaders import RWFNFileLoader

rwfn_loader = RWFNFileLoader("path/to/file.w")
rwfn_df = rwfn_loader.load()
# 实例方法，正确处理 Fortran 记录标记

# 访问特定轨道
pq_data = rwfn_loader.get_functions_for_orbital(orbital_n=4, orbital_l=0)
grid = rwfn_loader.get_grid()
orbitals = rwfn_loader.get_all_orbitals()
```

### 混合系数加载

#### 旧方式
```python
from graspkit.data_IO import GraspFileLoad

loader = GraspFileLoad(config)
mix_data = GraspFileLoad.load_rmix_data("path/to/file.m")
# 静态方法，返回格式不明确
```

#### 新方式
```python
from graspkit.data_IO.loaders import MixCoefLoader

mix_loader = MixCoefLoader("path/to/file.m")
mix_data = mix_loader.load()
# 返回类型明确：MixCoefficientData

# 访问特定块
block_data = mix_loader.get_block_data(block_idx=0)
block_count = mix_loader.get_block_count()
```

### 能级文件加载

#### 旧方式
```python
from graspkit.data_IO import EnergyFile2csv

# 需要先转换为 CSV
converter = EnergyFile2csv(config)
converter.energy2csv()
# 然后读取 CSV 文件
```

#### 新方式
```python
from graspkit.data_IO.loaders import EnergyFileLoader

energy_loader = EnergyFileLoader("path/to/file.level")
# 直接解析为 DataFrame，无需中间 CSV 文件
df = energy_loader.get_energy_levels()
total_energy = energy_loader.get_total_energy()
```

---

## 迁移步骤

### 步骤 1：识别使用的功能

检查代码中 `GraspFileLoad` 的使用方式：

```python
# 查找所有使用
grep -r "GraspFileLoad" src/
grep -r "load_rwfn_bin\|load_rmix_data\|load_csf_data" src/
```

### 步骤 2：导入新加载器

```python
# 旧导入
from graspkit.data_IO import GraspFileLoad

# 新导入
from graspkit.data_IO.loaders import (
    RWFNFileLoader,  # 替代 load_rwfn_bin
    MixCoefLoader,              # 替代 load_rmix_data
    CSFLoader,                  # 替代 load_csf_data
    EnergyFileLoader,           # 替代 EnergyFile2csv
    TransitionLoader,           # 新增：跃迁数据
    LSJCompLoader,                  # 新增：LSJ 耦合数据
)
```

### 步骤 3：更新实例化代码

```python
# 旧方式：需要配置字典
config = {
    "file_dir": "/path/to/files",
    "file_name": "example",
    "level_parameter": "_001",
    "this_as": "001",
}
loader = GraspFileLoad(config)

# 新方式：直接传入文件路径
loader = RWFNFileLoader("/path/to/files/example_001_001.w")
```

### 步骤 4：更新方法调用

```python
# 旧方式：静态方法
rwfn_df = GraspFileLoad.load_rwfn_bin("path/to/file.w")

# 新方式：实例方法
loader = RWFNFileLoader("path/to/file.w")
rwfn_df = loader.load()
```

### 步骤 5：更新数据访问

```python
# 旧方式：直接访问 DataFrame 列
pg_column = rwfn_df["P(4s)"]

# 新方式：使用专用方法
pq_data = loader.get_functions_for_orbital(orbital_n=4, orbital_l=0)
pg_values = pq_data["P_r"]
```

---

## 详细示例

### 示例 1：完整的波函数分析工作流

#### 旧代码
```python
from graspkit.data_IO import GraspFileLoad

# 配置加载器
config = {
    "file_dir": "/data/grasp/run1",
    "file_name": "atom",
    "level_parameter": "_001",
    "this_as": "001",
}
loader = GraspFileLoad(config)

# 加载波函数
rwfn_df = GraspFileLoad.load_rwfn_bin("/data/grasp/run1/atom_001_001.w")

# 分析数据
for col in rwfn_df.columns:
    if col.startswith("P("):
        print(f"Found orbital: {col}")
```

#### 新代码
```python
from graspkit.data_IO.loaders import RWFNFileLoader

# 直接实例化
loader = RWFNFileLoader("/data/grasp/run1/atom_001_001.w")

# 加载波函数
rwfn_df = loader.load()

# 使用专用方法分析
orbitals = loader.get_all_orbitals()
for n, l in orbitals:
    print(f"Found orbital: n={n}, l={l}")
    pq_data = loader.get_functions_for_orbital(n, l)
    print(f"  P_r shape: {pq_data['P_r'].shape}")
    print(f"  Q_r shape: {pq_data['Q_r'].shape}")
```

### 示例 2：CSF 分析

#### 旧代码
```python
from graspkit.data_IO import GraspFileLoad

loader = GraspFileLoad(config)
csf_data = loader.load_csf_data("path/to/file.c")

# 需要知道返回格式的结构
print(f"Total blocks: {csf_data['block_num']}")
print(f"Total CSFs: {sum(csf_data['CSFs_block_length'])}")
```

#### 新代码
```python
from graspkit.data_IO.loaders import CSFLoader

csf_loader = CSFLoader("path/to/file.c")
csf_data = csf_loader.load()

# 使用专用方法，更清晰
print(f"Total blocks: {csf_loader.get_blocks_count()}")
print(f"Total CSFs: {csf_loader.get_total_csfs_count()}")
print(f"Peel subshells: {csf_loader.get_peel_subshells()}")

# 访问特定块
block_data = csf_loader.get_block_data(block_idx=0)
print(f"First block has {len(block_data)} CSFs")
```

### 示例 3：混合系数分析

#### 旧代码
```python
from graspkit.data_IO import GraspFileLoad

mix_data = GraspFileLoad.load_rmix_data("path/to/file.m")

# 需要了解列表索引的含义
print(f"Blocks: {mix_data[0]}")
print(f"First block mix coef: {mix_data[8][0]}")
```

#### 新代码
```python
from graspkit.data_IO.loaders import MixCoefLoader

mix_loader = MixCoefLoader("path/to/file.m")
mix_data = mix_loader.load()

# 使用有意义的属性名
print(f"Blocks: {len(mix_data.blocks)}")
print(f"First block mix coef shape: {mix_data.blocks[0].mix_coefficients.shape}")

# 使用专用方法
block_data = mix_loader.get_block_data(block_idx=0)
print(f"Block 0 evec shape: {block_data['evec'].shape}")
```

---

## API 参考

### CSFLoader

加载 GRASP2018 CSF 配置文件（.c）。

```python
from graspkit.data_IO.loaders import CSFLoader

loader = CSFLoader("path/to/file.c")
csf_data = loader.load()

# 方法
loader.get_block_data(block_idx: int) -> list[list[str]]
loader.get_total_csfs_count() -> int
loader.get_blocks_count() -> int
loader.get_peel_subshells() -> list[str]
```

### RWFNFileLoader

加载 GRASP2018 径向波函数文件（.w）。

```python
from graspkit.data_IO.loaders import RWFNFileLoader

loader = RWFNFileLoader("path/to/file.w")
rwfn_df = loader.load()

# 方法
loader.get_functions_for_orbital(orbital_n: int, orbital_l: int) -> dict
loader.get_grid() -> NDArray[np.float64]
loader.get_all_orbitals() -> list[tuple[int, int]]
```

### MixCoefLoader

加载 GRASP2018 混合系数文件（.m）。

```python
from graspkit.data_IO.loaders import MixCoefLoader

loader = MixCoefLoader("path/to/file.m")
mix_data = loader.load()

# 方法
loader.get_block_data(block_idx: int) -> dict[str, Any]
loader.get_block_count() -> int
```

### EnergyFileLoader

加载 GRASP2018 能级文件（.level）。

```python
from graspkit.data_IO.loaders import EnergyFileLoader

loader = EnergyFileLoader("path/to/file.level")

# 方法
loader.get_energy_levels() -> pd.DataFrame
loader.get_total_energy() -> float
loader.get_number_of_levels() -> int
loader.get_configurations() -> pd.Series
```

### TransitionLoader

加载 GRASP2018 跃迁数据文件。

```python
from graspkit.data_IO.loaders import TransitionLoader

loader = TransitionLoader("path/to/file")
data = loader.load()
```

### LSJCompLoader

加载 GRASP2018 LSJ 耦合数据文件。

```python
from graspkit.data_IO.loaders import LSJCompLoader

loader = LSJCompLoader("path/to/file")
data = loader.load()
```

---

## 常见问题

### Q: 旧代码还能用吗？

A: 可以，但已标记为弃用。建议在新项目中使用新 API。

### Q: 返回数据格式有变化吗？

A: 核心数据格式保持兼容，但新 API 使用更明确的类型注解和命名。

### Q: 如何处理文件路径？

A: 新加载器接受绝对路径或相对路径（相对于当前工作目录），无需复杂的配置字典。

### Q: 性能有差异吗？

A: 新加载器性能相同或更好，特别是对于大型二进制文件。

### Q: 我可以在过渡期混合使用新旧 API 吗？

A: 技术上可以，但不推荐。建议一次性完成迁移。

---

## 迁移检查清单

- [ ] 识别所有使用 `GraspFileLoad` 的代码
- [ ] 识别所有使用 `EnergyFile2csv` 的代码
- [ ] 更新 import 语句
- [ ] 更新实例化代码
- [ ] 更新方法调用
- [ ] 更新数据访问代码
- [ ] 运行测试验证
- [ ] 更新文档

---

## 获取帮助

如有问题或建议，请：

1. 查看各加载器的 docstring
2. 查看单元测试（tests/ 目录）
3. 提交 Issue 到 GitHub 仓库

---

**文档版本**: 1.0
**最后更新**: 2026-01-19
**适用于**: graspkit >= 3.0
