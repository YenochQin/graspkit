# -*- encoding: utf-8 -*-
import math


def calculate_deformation(
    Z: int,
    A: int,
    spin_I: float,
    Q_s: float,
) -> dict[str, float]:
    """Calculate intrinsic quadrupole moment and deformation parameter.

    Args:
        Z: Atomic number, equal to the proton count.
        A: Mass number.
        spin_I: Nuclear spin quantum number.
        Q_s: Spectroscopic quadrupole moment in barns.

    Returns:
        Dictionary containing ``Q_0`` and ``beta_2``.
    """

    # 常数定义
    R0_CONST = 1.2  # 单位: fm
    BARN_TO_FM2 = 100.0  # 1 barn = 100 fm²

    # 将Q_s从靶恩转换为fm²
    Q_s_fm2 = Q_s * BARN_TO_FM2

    # 计算等效核半径 R0 和 R0²
    R0 = R0_CONST * (A ** (1 / 3))
    R0_sq = R0**2

    # 从光谱四极矩 Q_s 计算内禀电四极矩 Q0
    # 使用公式: Q_s = [I(2I-1)/(I+1)(2I+3)] * Q0
    conversion_factor = (spin_I * (2 * spin_I - 1)) / ((spin_I + 1) * (2 * spin_I + 3))
    Q0_fm2 = Q_s_fm2 / conversion_factor

    # 将Q0转换回靶恩单位用于输出
    Q0_b = Q0_fm2 / BARN_TO_FM2

    # 计算四极形变参数 β2
    # 使用公式: β2 ≈ (√(5π) * Q0) / (3 * Z * R0²)
    numerator = math.sqrt(5 * math.pi) * Q0_fm2
    denominator = 3 * Z * R0_sq
    beta_2 = numerator / denominator

    # 返回结果
    return {
        "intrinsic_quadrupole_moment_Q0(b)": Q0_b,
        "quadrupole_deformation_beta2": beta_2,
        "nuclear_radius_R0(fm)": R0,
        "conversion_factor": conversion_factor,
    }


if __name__ == "__main__":
    Z = 60  # 原子序数
    A = 152  # 原子质量数
    I = 0  # 核自旋
    Q = 0  # K 值，通常2+态的 K=0
    beta2 = calculate_deformation(Z, A, I, Q)

    print(beta2)
