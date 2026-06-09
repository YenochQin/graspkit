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
