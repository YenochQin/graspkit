import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graspkit.data_IO import TransitionLoader  # noqa: E402


def test_lsj_transition_loader_preserves_level_energy_strings(tmp_path: Path) -> None:
    transition_file = tmp_path / "sample.ct.lsj"
    transition_file.write_text(
        """Transition between files:
header 1
header 2
header 3
header 4
  2-11257.5967049  5s(2).4d(10).5p(6).6s(2)
  4-11255.1234567  5s(2).4d(10).5p(6).6p
  12462.51 CM-1      8024.06 ANGS(VAC)      8023.23 ANGS(AIR)
 E1  S =  8.20207D-09   GF =  1.61683D-10   AKI =  9.08381D-04
          5.47381D-05         1.07902D-06          6.06225D+00
""",
        encoding="utf-8",
    )

    df = TransitionLoader(transition_file, load_ct_lsj=True).to_dataframe()

    assert df.schema["upper_energy"] == pl.String
    assert df.schema["lower_energy"] == pl.String
    assert df["upper_energy"].to_list() == ["-11255.1234567"]
    assert df["lower_energy"].to_list() == ["-11257.5967049"]

    for suffix in (
        ".ci.transition.lsj",
        ".ci.transition.phase.lsj",
        ".ci.transition.lsj.txt",
        ".ci.transition.phase.lsj.txt",
    ):
        report = tmp_path / ("sample" + suffix)
        report.write_bytes(transition_file.read_bytes())
        assert TransitionLoader(report).to_dataframe().equals(df)
    assert TransitionLoader(tmp_path, load_ct_lsj=True).get_loaded_files() == [
        str(tmp_path / "sample.ci.transition.lsj")
    ]


def test_scan_keeps_legacy_only_pairs_and_falls_back_to_main_report(tmp_path: Path) -> None:
    native = tmp_path / "first.second.mcdhf.transition"
    legacy = tmp_path / "third.fourth.t"
    previous_suite = tmp_path / "fifth.sixth.mcdhf.transition.txt"
    for path in (
        native,
        legacy,
        previous_suite,
        native.with_name(native.name + ".txt"),
        tmp_path / "first.second.t",
        tmp_path / "first.second.ci.transition",
    ):
        path.touch()
    loader = TransitionLoader(tmp_path, calculation_type="rmcdhf", load_ct_lsj=True)
    assert loader.get_loaded_files() == [
        str(path) for path in sorted((native, legacy, previous_suite))
    ]


def test_main_report_reads_same_state_rows(tmp_path: Path) -> None:
    report = tmp_path / "sample.sample.mcdhf.transition"
    report.write_text(
        " Transition in file:\n f = sample\n"
        "\n Magnetic 2**( 1)-pole transitions\n ===========================\n\n"
        " Upper           Lower\n"
        " Lev  J P        Lev  J P       E (Kays)       A (s-1)           gf            S\n"
        " f    1    3 +   f    1    4 +   1379.17 M   6.81481D-02   3.75987D-07   6.74154D+00\n"
    )
    rows = TransitionLoader(tmp_path, calculation_type="rmcdhf", load_ct_lsj=True).to_dataframe()
    assert rows.select("upper_file", "lower_file", "energy", "transition_type").row(0) == (
        "f", "f", 1379.17, "M1"
    )
