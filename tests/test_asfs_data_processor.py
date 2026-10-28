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
    iterative_levels_collection,
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


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_iterative_levels_collection_preserves_unmatched_levels(tmp_path: Path) -> None:
    header = (
        "GRASP energy levels\n"
        "No Pos  J  Parity  EnergyTotal  EnergyLevel  splitting  configuration_raw\n"
        "header separator\nheader units\n"
    )
    first = tmp_path / "first.level"
    second = tmp_path / "second.level"
    first.write_text(
        header + "1 1 0 + -10.0 0.0 0.0 1s\n2 2 0 + -9.0 20.0 0.0 2s\n-----\n",
        encoding="utf-8",
    )
    second.write_text(
        header + "1 2 0 + -8.5 25.0 0.0 2s\n2 3 0 + -8.0 50.0 0.0 3s\n-----\n",
        encoding="utf-8",
    )

    merged = iterative_levels_collection([str(first), str(second)], ["A", "B"]).sort(
        "Pos"
    )

    assert merged.select("Pos", "J", "Parity").rows() == [
        ("1", "0", "+"), ("2", "0", "+"), ("3", "0", "+")
    ]
    assert merged["EnergyLevel_A"].to_list() == [0.0, 20.0, 0.0]
    assert merged["EnergyLevel_B"].to_list() == [0.0, 25.0, 50.0]
    assert merged["DeltaEA_to_B"].to_list() == [0.0, 5.0, 0.0]
    assert not any(column.endswith("_right") for column in merged.columns)


def test_format_configuration_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="temp_configuration must be a string"):
        format_configuration(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="temp_configuration must be a string"):
        format_configuration(123)
