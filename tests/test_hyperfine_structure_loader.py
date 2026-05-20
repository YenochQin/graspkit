import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graspkit.data_IO import HyperfineStructureLoader  # noqa: E402


def test_hyperfine_structure_loader_parses_h_file(tmp_path: Path) -> None:
    hyperfine_file = tmp_path / "sample.h"
    hyperfine_file.write_text(
        """Nuclear spin                         0.000000000000000D+00 au
Nuclear magnetic dipole moment       0.000000000000000D+00 n.m.
Nuclear electric quadrupole moment   0.000000000000000D+00 barns


 Interaction constants:

 Level1  J Parity         A (MHz)             B (MHz)             g_J              delta g_J           total g_J

   1        1 +                   NaN   -0.0000000000D+00    1.4996344866D+00    1.1599984508D-03    1.5007944850D+00
   2        1 +                   NaN    0.0000000000D+00    1.4980570919D+00    1.1563123624D-03    1.4992134043D+00
""",
        encoding="utf-8",
    )

    loader = HyperfineStructureLoader(hyperfine_file)

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
        "A_MHz",
        "B_MHz",
        "g_J",
        "delta_g_J",
        "total_g_J",
    ]
    assert df.shape == (2, 8)
    assert math.isnan(df["A_MHz"][0])
    assert df["Level"].to_list() == [1, 2]
    assert df["J"].to_list() == ["1", "1"]
    assert df["total_g_J"][1] == 1.4992134043


def test_hyperfine_structure_loader_parses_ch_file(tmp_path: Path) -> None:
    hyperfine_file = tmp_path / "sample.ch"
    hyperfine_file.write_text(
        """Nuclear spin                         1.500000000000000D+00 au
Nuclear magnetic dipole moment      -3.373000000000000D-01 n.m.
Nuclear electric quadrupole moment   1.360000000000000D+00 barns


 Interaction constants:

 Level1  J Parity         A (MHz)             B (MHz)             g_J              delta g_J           total g_J

   1        2 +     -2.4023728407D+02    1.6169434620D+02    2.3224046852D+00    3.0708234925D-03    2.3254755087D+00
   2        3 -     -1.7727590392D+02   -6.1995456152D+02    1.9483724334D+00    2.2021543360D-03    1.9505745878D+00
""",
        encoding="utf-8",
    )

    loader = HyperfineStructureLoader(hyperfine_file)

    nuclear_parameters = loader.get_nuclear_parameters()
    assert nuclear_parameters.spin == 1.5
    assert nuclear_parameters.magnetic_dipole_moment == -0.3373
    assert nuclear_parameters.electric_quadrupole_moment == 1.36

    df = loader.get_interaction_constants()
    assert df.shape == (2, 8)
    assert df["A_MHz"].to_list() == [-240.23728407, -177.27590392]
    assert df["B_MHz"].to_list() == [161.6943462, -619.95456152]
    assert df["Parity"].to_list() == ["+", "-"]
    assert loader.get_number_of_levels() == 2
