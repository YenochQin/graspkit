import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graspkit.data_IO import GJFactorLoader  # noqa: E402


def test_gj_factor_loader_parses_gj_file(tmp_path: Path) -> None:
    gj_file = tmp_path / "sample.gj"
    gj_file.write_text(
        """Nuclear spin                         0.000000000000000D+00 au
Nuclear magnetic dipole moment       0.000000000000000D+00 n.m.
Nuclear electric quadrupole moment   0.000000000000000D+00 barns


 Interaction constants:

 Level1  J Parity           g_J              delta g_J           total g_J

   1        1 +      1.4996344866D+00    1.1599984508D-03    1.5007944850D+00
   3        1 -      5.0155075076D-01   -1.1553889073D-03    5.0039536186D-01
""",
        encoding="utf-8",
    )

    loader = GJFactorLoader(gj_file)

    nuclear_parameters = loader.get_nuclear_parameters()
    assert nuclear_parameters.spin == 0.0
    assert nuclear_parameters.spin_unit == "au"
    assert nuclear_parameters.magnetic_dipole_moment == 0.0
    assert nuclear_parameters.magnetic_dipole_moment_unit == "n.m."
    assert nuclear_parameters.electric_quadrupole_moment == 0.0
    assert nuclear_parameters.electric_quadrupole_moment_unit == "barns"

    df = loader.load()
    assert df.columns == [
        "Level",
        "J",
        "Parity",
        "g_J",
        "delta_g_J",
        "total_g_J",
    ]
    assert df.shape == (2, 6)
    assert df["Level"].to_list() == [1, 3]
    assert df["J"].to_list() == ["1", "1"]
    assert df["Parity"].to_list() == ["+", "-"]
    assert df["g_J"].to_list() == [1.4996344866, 0.50155075076]
    assert df["delta_g_J"].to_list() == [0.0011599984508, -0.0011553889073]
    assert df["total_g_J"].to_list() == [1.500794485, 0.50039536186]
    assert loader.get_number_of_levels() == 2


def test_gj_factor_loader_parses_real_sample_file() -> None:
    gj_file = Path("/Users/yiqin/Downloads/e1_vv3as2.gj")
    if not gj_file.is_file():
        pytest.skip(f"Real sample file is not available: {gj_file}")

    loader = GJFactorLoader(gj_file)
    df = loader.get_gj_factors()

    assert df.shape == (29, 6)
    assert df.row(0, named=True) == {
        "Level": 1,
        "J": "1",
        "Parity": "+",
        "g_J": 1.4996344866,
        "delta_g_J": 0.0011599984508,
        "total_g_J": 1.500794485,
    }
