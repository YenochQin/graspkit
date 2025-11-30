#!/usr/bin/env python3
"""
简单测试 validate_csf_descriptors_coverage 函数
"""

import numpy as np
import sys
from pathlib import Path

# 添加graspkit模块路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import graspkit as gk
except ImportError as e:
    print(f"无法导入graspkit模块: {e}")
    sys.exit(1)

def test_simple_coverage():
    """简单测试覆盖检测函数"""

    print("=== 简单测试 validate_csf_descriptors_coverage 函数 ===\n")

    # 测试无子壳层信息：2个轨道，3个CSF
    # 每个轨道3个值: [电子数, j值, occupancy]
    # 轨道0: 有电子 (CSF0=2, CSF1=1, CSF2=0) -> 覆盖
    # 轨道1: 无电子 (CSF0=0, CSF1=0, CSF2=0) -> 未覆盖
    test_descriptors = np.array([
        [2, 1, 1,  0, 2, 0],  # CSF0: 轨道0=2, 轨道1=0
        [1, 2, 1,  0, 1, 0],  # CSF1: 轨道0=1, 轨道1=0
        [0, 3, 1,  0, 1, 0],  # CSF2: 轨道0=0, 轨道1=0
    ])

    print("测试数据形状:", test_descriptors.shape)
    print("描述符结构: 每个轨道3个值 [电子数, j值, occupancy]")
    print("电子数信息:")
    print("  轨道0:", test_descriptors[:, 0])  # [2, 1, 0]
    print("  轨道1:", test_descriptors[:, 3])  # [0, 0, 0]
    print()

    # 测试无子壳层信息
    is_covered, uncovered_orbitals = gk.validate_csf_descriptors_coverage(
        test_descriptors
    )

    print("测试结果:")
    print(f"  是否所有轨道都被覆盖: {is_covered}")
    print(f"  未覆盖的轨道索引: {uncovered_orbitals}")
    print(f"  预期: False, [1]")
    print(f"  结果: {'✅ PASS' if not is_covered and uncovered_orbitals == [1] else '❌ FAIL'}")
    print()

    # 手动验证函数逻辑
    print("手动验证函数逻辑:")
    print("  无子壳层信息:")
    print("    - values_per_orbital = 3")
    print("    - electron_index_in_orbital = 0")
    print("    - actual_n_orbitals = descriptors.shape[1] // 3 = 6 // 3 = 2")
    print("    - electron_indices = [0, 3]")  # 0, 3, 6,... < 6
    print("    - electron_counts = descriptors[:, [0, 3]]")
    print("      =", test_descriptors[:, [0, 3]])
    print("    - has_nonzero_electrons = np.any(electron_counts > 0, axis=0)")
    print("      =", np.any(test_descriptors[:, [0, 3]] > 0, axis=0))
    print("      = [True, False]")
    print("    - uncovered_orbitals = np.where(~has_nonzero_electrons)[0]")
    print("      = np.where([False, True])[0]")
    print("      = [1]")
    print()

if __name__ == "__main__":
    test_simple_coverage()