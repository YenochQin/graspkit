from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "graspkit"


def test_graspkit_declares_pep561_typing_marker() -> None:
    assert (PACKAGE_ROOT / "py.typed").is_file()
