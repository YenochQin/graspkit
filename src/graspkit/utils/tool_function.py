# -*- encoding: utf-8 -*-
"""
@Id :tool_function.py
@date :2024/05/07 11:11:09
@author :YenochQin (秦毅)
"""


def str_subshell_2_kappa(str_subshell: str) -> int:
    r"""
    $j = l + 1/2, \kappa = -(l+1)$
    $j = l - 1/2, \kappa = +l $
    """
    kappa_value = {
        "s ": -1,
        "p-": 1,
        "p ": -2,
        "d-": 2,
        "d ": -3,
        "f-": 3,
        "f ": -4,
        "g-": 4,
        "g ": -5,
        "h-": 5,
        "h ": -6,
        "i-": 6,
        "i ": -7,
    }

    return kappa_value.get(str_subshell, 0)


######################################################################


def doubleJ_to_J(input_doubleJ: int | str) -> str:
    doubleJ: int = int(input_doubleJ)
    if doubleJ % 2 == 0:
        return f"{int(doubleJ / 2)}"
    else:
        return f"{doubleJ}/2"


######################################################################


def chunk_string(s: str, n: int) -> list[str]:
    """将字符串分割成固定长度的块"""
    return [s[i : i + n] for i in range(0, len(s), n)]


######################################################################


def LS_shell_full_charged(shell_name: str, shell_charged_num: int) -> bool:
    full_charged = {"s": 2, "p": 6, "d": 10, "f": 14, "g": 18, "h": 22, "i": 26}
    return full_charged.get(shell_name, 0) == shell_charged_num


######################################################################
