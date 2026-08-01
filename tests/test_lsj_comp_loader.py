import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graspkit.data_IO.loaders.lsj_comp_loader import (  # noqa: E402
    CompositionUnit,
    LSJCompLoader,
    LevelComposition,
)


def test_lsj_comp_loader_builds_typed_levels_and_dataframe(
    tmp_path: Path,
) -> None:
    lsj_file = tmp_path / "sample.lsj.lbl"
    _ = lsj_file.write_text(
        """1 1 - -11257.596876908 97.449%
 0.95591698 0.91377727 first-configuration
 0.11744206 0.01379264 second-configuration
2 3/2 + -11255.123456789 88.000%
 0.90000000 0.81000000 third-configuration
""",
        encoding="utf-8",
    )

    loader = LSJCompLoader(lsj_file)

    assert loader.get_levels() == [
        LevelComposition(
            pos=1,
            j="1",
            parity="-",
            energy_total=-11257.596876908,
            composition_asf="97.449",
            compositions=[
                CompositionUnit(0.95591698, 0.91377727, "first-configuration"),
                CompositionUnit(0.11744206, 0.01379264, "second-configuration"),
            ],
        ),
        LevelComposition(
            pos=2,
            j="3/2",
            parity="+",
            energy_total=-11255.123456789,
            composition_asf="88.000",
            compositions=[
                CompositionUnit(0.9, 0.81, "third-configuration"),
            ],
        ),
    ]

    dataframe = loader.load()
    assert dataframe["Pos"].to_list() == [1, 2]
    assert dataframe["J"].to_list() == ["1", "3/2"]
    assert dataframe["EnergyTotal"].to_list() == [
        -11257.596876908,
        -11255.123456789,
    ]
