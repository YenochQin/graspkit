# -*- encoding: utf-8 -*-

def str_subshell_2_kappa(str_subshell: str) -> int:
    r"""
    Convert a relativistic subshell label to its Dirac kappa value.

    Args:
        str_subshell: Subshell label with the GRASP relativistic suffix, such
            as ``"p-"`` or ``"p "``.

    Returns:
        Dirac kappa value. Unknown labels return 0.

    Notes:
        For ``j = l + 1/2``, ``kappa = -(l + 1)``. For ``j = l - 1/2``,
        ``kappa = +l``.
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
    """Convert doubled angular momentum to the conventional J string.

    Args:
        input_doubleJ: Integer-like value representing ``2J``.

    Returns:
        Integer or half-integer J value formatted as a string.
    """
    doubleJ: int = int(input_doubleJ)
    if doubleJ % 2 == 0:
        return f"{int(doubleJ / 2)}"
    else:
        return f"{doubleJ}/2"


######################################################################


def chunk_string(s: str, n: int) -> list[str]:
    """Split a string into fixed-width chunks.

    Args:
        s: Input string.
        n: Chunk width.

    Returns:
        List of chunks in original order. The final chunk may be shorter than
        ``n``.
    """
    return [s[i : i + n] for i in range(0, len(s), n)]


######################################################################


def LS_shell_full_charged(shell_name: str, shell_charged_num: int) -> bool:
    """Check whether an LS-coupled shell is fully occupied.

    Args:
        shell_name: Orbital shell label such as ``"s"``, ``"p"``, or ``"d"``.
        shell_charged_num: Number of electrons occupying the shell.

    Returns:
        True if the electron count equals the configured full occupation for
        the shell.
    """
    full_charged = {"s": 2, "p": 6, "d": 10, "f": 14, "g": 18, "h": 22, "i": 26}
    return full_charged.get(shell_name, 0) == shell_charged_num


######################################################################
