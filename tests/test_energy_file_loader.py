import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graspkit.data_IO import EnergyFileLoader  # noqa: E402


def _write_level_file(tmp_path: Path) -> Path:
    level_file = tmp_path / "sample.level"
    _ = level_file.write_text(
        """GRASP energy levels
No Pos  J  Parity  EnergyTotal  EnergyLevel  splitting  configuration_raw
header separator
header units
1 1 0 + -11257.596876908 0.0 0.0 ground-configuration
-----
""",
        encoding="utf-8",
    )
    return level_file


def test_energy_file_loader_returns_total_energy(tmp_path: Path) -> None:
    loader = EnergyFileLoader(_write_level_file(tmp_path))

    assert loader.get_total_energy() == -11257.596876908


def test_energy_file_loader_formats_available_columns(tmp_path: Path) -> None:
    loader = EnergyFileLoader(_write_level_file(tmp_path))
    loader.df = pl.DataFrame({"Other": [1]})

    with pytest.raises(ValueError, match="Available columns: \\['Other'\\]"):
        _ = loader.get_total_energy()
