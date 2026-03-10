# Strict Mypy Cleanup Across `src/`

**日期**: 2026-03-10
**提交**: `5eef438`

## 背景

本轮工作目标是把 `src/` 目录下的严格类型检查问题集中清理完成，确保：

- `mypy src/` 全量通过
- 最近修改文件的函数签名、返回值和调用链具备更好的静态校验能力
- 对缺少类型 stub 的第三方库采用定向豁免，而不是降低全局检查强度

## 主要修改

### 1. `mypy` 配置收敛

在 `pyproject.toml` 中新增模块级定向豁免：

- `joblib`
- `sklearn`
- `sklearn.*`
- `seaborn`
- `scipy`
- `scipy.*`

这样可以避免第三方库缺少 `py.typed` 或类型 stub 时阻塞项目自身的严格类型检查。

### 2. `ml_module` 类型修复

新增 `src/graspkit/ml_module/ml_types.py`，统一承载若干 `TypedDict` 结果结构，包括：

- 预测输出
- 概率输出
- 标签输出
- 指标输出
- 评估结果

同时修复了以下类型问题：

- 训练与评估函数缺少返回类型注解
- `numpy` 标量与 `float` 的类型不兼容
- `Tensor | None`、`nn.Module` 等属性缺少显式注解
- 局部变量类型收窄不足导致的 `mypy` 误报

涉及文件：

- `src/graspkit/ml_module/ml_trainer.py`
- `src/graspkit/ml_module/ml_results_analyzer.py`
- `src/graspkit/ml_module/ml_regression_model.py`
- `src/graspkit/ml_module/neural_network.py`
- `src/graspkit/ml_module/ml_types.py`

### 3. `utils` 类型修复

修复了图表工具和通用工具中的严格类型问题，主要包括：

- `**kwargs` 缺少注解
- `Axes` 返回值没有做类型收窄
- `Literal` 共享轴参数不匹配
- `add_axes()` 的入参类型不满足重载定义
- 局部列表、嵌套函数和标题/标签分支缺少精确类型

涉及文件：

- `src/graspkit/utils/fig_settings.py`
- `src/graspkit/utils/plot_functions.py`
- `src/graspkit/utils/quadrupole_deformation.py`

### 4. `data_IO` 与 `CSFs_processor` 类型修复

收敛了读取器、写出器和 CSF 处理逻辑中的历史类型问题，主要包括：

- 写出函数缺少 `-> None`
- loader 中字典嵌套结构缺少显式收窄
- `polars` 返回值需要显式转换为 `float`
- `str` 与 `int` 临时变量复用导致的参数类型错误

涉及文件：

- `src/graspkit/data_IO/produced_data_writor.py`
- `src/graspkit/data_IO/loaders/lsj_comp_loader.py`
- `src/graspkit/data_IO/loaders/energy_file_loader.py`
- `src/graspkit/CSFs_processor/CSFs_compress_extract.py`

## 验证结果

本轮完成后，已确认以下检查通过：

```bash
mypy src/
```

输出结果：

```text
Success: no issues found in 37 source files
```

同时，相关改动文件的 `ruff check` 也已通过。

## 结果与收益

本次清理带来的直接收益：

1. `src/` 全量恢复严格类型检查可用状态
2. 最近签名改动已能被 `mypy` 直接校验
3. 第三方库缺少 stub 不再污染项目自身问题定位
4. 若后续再次引入类型回归，CI/本地检查可以更快发现

## 建议

后续开发中建议继续保持以下习惯：

- 新增公共函数时同步补全返回类型
- 对复杂嵌套结构优先引入 `TypedDict` 或显式局部变量注解
- 对三方库缺 stub 的情况优先采用模块级 override，而不是全局放宽
- 在涉及签名调整的改动后，优先执行：

```bash
ruff check .
mypy src/
```

