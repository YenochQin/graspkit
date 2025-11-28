#!/usr/bin/env python3
"""
测试 validate_csf_descriptors_coverage 函数的正确性
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

def test_coverage_function():
    """测试覆盖检测函数的正确性"""

    print("=== 测试 validate_csf_descriptors_coverage 函数 ===\n")

    # 测试场景1: 无子壳层信息（每个轨道3个值：电子数、j值、occupancy）
    print("测试场景1: 无子壳层信息")
    print("描述符结构: [电子数, j值, occupancy, 电子数, j值, occupancy, ...]")

    # 创建测试数据：3个轨道，4个CSF
    # 轨道0: CSF0有2个电子，CSF1有0个，CSF2有1个，CSF3有0个
    # 轨道1: 所有CSF都有0个电子 -> 应该被识别为未覆盖
    # 轨道2: CSF0有1个电子，CSF1有3个，CSF2有0个，CSF3有2个
    test_descriptors_no_subshell = np.array([
        [2, 1, 1,  0, 2, 0,  1, 3, 1],  # CSF0
        [1, 2, 1,  0, 1, 0,  3, 1, 2],  # CSF1
        [0, 3, 0,  0, 0, 0,  0, 2, 0],  # CSF2
        [2, 1, 1,  0, 2, 0,  2, 0, 1],  # CSF3
    ])

    print(f"测试数据形状: {test_descriptors_no_subshell.shape}")
    print("电子数信息（每个CSF，每个轨道的第一个值）:")
    print(test_descriptors_no_subshell[:, [0, 3, 6]])  # 轨道0,1,2的电子数
    print()

    # 测试函数
    is_covered, uncovered_orbitals = gk.validate_csf_descriptors_coverage(
        test_descriptors_no_subshell, with_subshell_info=False
    )

    print(f"是否所有轨道都被覆盖: {is_covered}")
    print(f"未覆盖的轨道索引: {uncovered_orbitals}")
    print(f"预期结果: False, [1] (轨道1没有被任何CSF占据)")
    print(f"测试结果: {'✅ PASS' if not is_covered and uncovered_orbitals == [1] else '❌ FAIL'}")
    print()

    # 测试场景2: 有子壳层信息（每个轨道5个值）
    print("测试场景2: 有子壳层信息")
    print("描述符结构: [子壳层, 电子数, j值, occupancy, 其他, 子壳层, 电子数, j值, occupancy, 其他, ...]")

    # 创建测试数据：3个轨道，3个CSF，每个轨道5个值，电子数在第2个位置（索引1）
    # 轨道0: CSF0有2个电子，CSF1有0个，CSF2有1个
    # 轨道1: CSF0有0个，CSF1有3个，CSF2有0个 -> 被CSF1覆盖
    # 轨道2: 所有CSF都有0个电子 -> 应该被识别为未覆盖
    test_descriptors_with_subshell = np.array([
        [1, 2, 1, 1, 0,  2, 0, 2, 0, 1,  3, 1, 3, 1, 0],  # CSF0
        [1, 0, 2, 1, 1,  2, 3, 1, 2, 0,  3, 0, 2, 0, 1],  # CSF1
        [1, 1, 3, 0, 0,  2, 0, 0, 1, 1,  3, 0, 1, 2, 0],  # CSF2
    ])

    print(f"测试数据形状: {test_descriptors_with_subshell.shape}")
    print("电子数信息（每个CSF，每个轨道的第二个值）:")
    print(test_descriptors_with_subshell[:, [1, 6, 11]])  # 轨道0,1,2的电子数
    print()

    # 测试函数
    is_covered, uncovered_orbitals = gk.validate_csf_descriptors_coverage(
        test_descriptors_with_subshell, with_subshell_info=True
    )

    print(f"是否所有轨道都被覆盖: {is_covered}")
    print(f"未覆盖的轨道索引: {uncovered_orbitals}")
    print(f"预期结果: False, [2] (轨道2没有被任何CSF占据)")
    print(f"测试结果: {'✅ PASS' if not is_covered and uncovered_orbitals == [2] else '❌ FAIL'}")
    print()

    # 测试场景3: 完全覆盖的情况
    print("测试场景3: 完全覆盖情况")
    fully_covered_descriptors = np.array([
        [2, 1, 1,  1, 2, 1],  # 轨道0有2个电子，轨道1有1个电子
        [0, 2, 0,  3, 1, 2],  # 轨道0有0个，轨道1有3个
        [1, 3, 1,  0, 0, 0],  # 轨道0有1个，轨道1有0个
    ])

    is_covered, uncovered_orbitals = gk.validate_csf_descriptors_coverage(
        fully_covered_descriptors, with_subshell_info=False
    )

    print(f"是否所有轨道都被覆盖: {is_covered}")
    print(f"未覆盖的轨道索引: {uncovered_orbitals}")
    print(f"预期结果: True, []")
    print(f"测试结果: {'✅ PASS' if is_covered and uncovered_orbitals == [] else '❌ FAIL'}")
    print()

    # 测试场景4: 空数据
    print("测试场景4: 空数据")
    empty_descriptors = np.array([])

    is_covered, uncovered_orbitals = gk.validate_csf_descriptors_coverage(
        empty_descriptors, with_subshell_info=False
    )

    print(f"是否所有轨道都被覆盖: {is_covered}")
    print(f"未覆盖的轨道索引: {uncovered_orbitals}")
    print(f"预期结果: False, []")
    print(f"测试结果: {'✅ PASS' if not is_covered and uncovered_orbitals == [] else '❌ FAIL'}")
    print()

    print("=== 函数逻辑分析 ===")
    print("1. 无子壳层信息时:")
    print("   - 每个轨道3个值: [电子数, j值, occupancy]")
    print("   - 电子数索引: 0, 3, 6, 9, ... (间隔为3)")
    print("   - 轨道数量 = descriptors.shape[1] // 3")
    print()
    print("2. 有子壳层信息时:")
    print("   - 每个轨道5个值: [子壳层, 电子数, j值, occupancy, 其他]")
    print("   - 电子数索引: 1, 6, 11, 16, ... (间隔为5)")
    print("   - 轨道数量 = descriptors.shape[1] // 5")
    print()
    print("3. 覆盖条件:")
    print("   - 对于每个轨道，至少有一个CSF的电子数 > 0")
    print("   - 使用 np.any(electron_counts > 0, axis=0) 检查覆盖情况")

if __name__ == "__main__":
    test_coverage_function()