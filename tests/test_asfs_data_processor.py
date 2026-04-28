import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graspkit.grasp_data_extractor.asfs_data_processor import (  # noqa: E402
    format_configuration,
    format_energy_configurations,
)


def test_format_energy_configurations_keeps_split_output_by_default() -> None:
    energy_df = pl.DataFrame({"configuration_raw": ["3d(2)3F_4F", ""]})

    result = format_energy_configurations(energy_df)

    assert result["configuration"].to_list() == [r"$3d^{2}\,(^3\mathrm{F})\;$", ""]
    assert result["LSJ"].to_list() == [r"$^{4}\mathrm{F}$", ""]


def test_format_energy_configurations_can_merge_configuration_and_lsj() -> None:
    energy_df = pl.DataFrame({"configuration_raw": ["3d(2)3F_4F", "3s"]})

    result = format_energy_configurations(energy_df, merge_output=True)

    assert result["configuration"].to_list() == [
        r"$3d^{2}\,(^3\mathrm{F})\;^{4}\mathrm{F}$",
        r"$3s\;$",
    ]
    assert result["LSJ"].to_list() == [r"$^{4}\mathrm{F}$", ""]


def test_format_energy_configurations_wraps_merged_multishell_output_in_math_mode() -> None:
    energy_df = pl.DataFrame(
        {
            "configuration_raw": [
                "5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)8S0_8S.5d_9D"
            ]
        }
    )

    result = format_energy_configurations(energy_df, merge_output=True)

    assert result["configuration"].to_list() == [
        r"$4f^{7}\,(^8_0\mathrm{S})\;5d\;^{9}\mathrm{D}$"
    ]
    assert result["LSJ"].to_list() == [r"$^{9}\mathrm{D}$"]


def test_format_configuration_handles_empty_input() -> None:
    assert format_configuration("") == ("", "")


def test_format_configuration_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="temp_configuration must be a string"):
        format_configuration(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="temp_configuration must be a string"):
        format_configuration(123)
