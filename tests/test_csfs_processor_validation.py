from pathlib import Path

import numpy as np
import pytest

from graspkit.CSFs_processor.validation import (
    load_selection_idxs,
    validate_coupling_level,
    validate_header_lines,
    validate_headers_match,
    validate_output_path_disjoint,
    validate_selection_idxs,
)


def test_validate_selection_idxs_accepts_1d_int_array() -> None:
    result = validate_selection_idxs(np.array([2, 0, 1], dtype=np.int32))

    np.testing.assert_array_equal(result, np.array([2, 0, 1], dtype=np.int64))
    assert result.dtype == np.int64


def test_validate_selection_idxs_accepts_empty_array() -> None:
    result = validate_selection_idxs(np.array([], dtype=np.int64))

    assert result.shape == (0,)


@pytest.mark.parametrize(
    "bad_array",
    [
        np.array([[0, 1], [2, 3]], dtype=np.int64),
    ],
    ids=["2d"],
)
def test_validate_selection_idxs_rejects_non_1d(bad_array: np.ndarray) -> None:
    with pytest.raises(ValueError, match="一维数组"):
        validate_selection_idxs(bad_array)


def test_validate_selection_idxs_rejects_float_dtype() -> None:
    with pytest.raises(ValueError, match="整数 dtype"):
        validate_selection_idxs(np.array([1.9, 2.0]))


def test_validate_selection_idxs_rejects_bool_dtype() -> None:
    with pytest.raises(ValueError, match="整数 dtype"):
        validate_selection_idxs(np.array([True, False]))


def test_validate_selection_idxs_rejects_object_dtype() -> None:
    with pytest.raises(ValueError, match="整数 dtype"):
        validate_selection_idxs(np.array([1, "2"], dtype=object))


def test_validate_selection_idxs_rejects_string_dtype() -> None:
    with pytest.raises(ValueError, match="整数 dtype"):
        validate_selection_idxs(np.array(["1", "2"]))


def test_validate_selection_idxs_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="不能为负数"):
        validate_selection_idxs(np.array([0, -1, 2], dtype=np.int64))


def test_validate_selection_idxs_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="越界"):
        validate_selection_idxs(np.array([0, 5], dtype=np.int64), row_count=3)


def test_validate_selection_idxs_accepts_in_range_values() -> None:
    result = validate_selection_idxs(np.array([0, 2], dtype=np.int64), row_count=3)

    np.testing.assert_array_equal(result, np.array([0, 2], dtype=np.int64))


def test_validate_selection_idxs_rejects_duplicates_by_default() -> None:
    with pytest.raises(ValueError, match="重复值"):
        validate_selection_idxs(np.array([1, 1, 2], dtype=np.int64))


def test_validate_selection_idxs_allows_duplicates_when_requested() -> None:
    result = validate_selection_idxs(
        np.array([1, 1, 2], dtype=np.int64),
        allow_duplicates=True,
    )

    np.testing.assert_array_equal(result, np.array([1, 1, 2], dtype=np.int64))


def test_validate_selection_idxs_error_includes_source_label() -> None:
    with pytest.raises(ValueError, match="foo.npy"):
        validate_selection_idxs(
            np.array([-1], dtype=np.int64),
            source="foo.npy",
        )


def test_load_selection_idxs_reads_and_validates_npy_file(tmp_path: Path) -> None:
    idx_file = tmp_path / "idxs.npy"
    np.save(idx_file, np.array([2, 0, 1], dtype=np.int64))

    result = load_selection_idxs(idx_file, row_count=3)

    np.testing.assert_array_equal(result, np.array([2, 0, 1], dtype=np.int64))


def test_load_selection_idxs_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_selection_idxs(tmp_path / "missing.npy")


def test_load_selection_idxs_rejects_invalid_dtype(tmp_path: Path) -> None:
    idx_file = tmp_path / "idxs.npy"
    np.save(idx_file, np.array([1.5, 2.5]))

    with pytest.raises(ValueError, match="整数 dtype"):
        load_selection_idxs(idx_file)


def test_validate_header_lines_accepts_exactly_five_strings() -> None:
    header = ["a", "b", "c", "d", "e"]

    assert validate_header_lines(header) == header


@pytest.mark.parametrize(
    "bad_header",
    [
        ["a", "b", "c", "d"],
        ["a", "b", "c", "d", "e", "f"],
    ],
    ids=["too-few", "too-many"],
)
def test_validate_header_lines_rejects_wrong_line_count(
    bad_header: list[str],
) -> None:
    with pytest.raises(ValueError, match="5 行"):
        validate_header_lines(bad_header)


def test_validate_header_lines_rejects_non_string_entries() -> None:
    with pytest.raises(ValueError, match="字符串列表"):
        validate_header_lines(["a", "b", "c", "d", 5])  # type: ignore[list-item]


def test_validate_headers_match_accepts_identical_headers() -> None:
    header = ["a", "b", "c", "d", "e"]

    result = validate_headers_match([header, list(header), list(header)])

    assert result == header


def test_validate_headers_match_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="没有可比较"):
        validate_headers_match([])


def test_validate_headers_match_rejects_any_difference() -> None:
    first = ["a", "b", "c", "d", "e"]
    second = ["a", "b", "c", "d", "DIFFERENT"]

    with pytest.raises(ValueError, match="header 不一致"):
        validate_headers_match([first, second])


def test_validate_headers_match_error_includes_labels() -> None:
    first = ["a", "b", "c", "d", "e"]
    second = ["a", "b", "c", "d", "DIFFERENT"]

    with pytest.raises(ValueError, match="j0"):
        validate_headers_match([first, second], labels=["j0", "j2"])


def test_validate_output_path_disjoint_accepts_distinct_paths(tmp_path: Path) -> None:
    validate_output_path_disjoint(tmp_path / "out.c", [tmp_path / "in.c", None])


def test_validate_output_path_disjoint_rejects_matching_path(tmp_path: Path) -> None:
    shared = tmp_path / "shared.c"

    with pytest.raises(ValueError, match="冲突"):
        validate_output_path_disjoint(shared, [shared])


def test_validate_output_path_disjoint_ignores_none_entries(tmp_path: Path) -> None:
    validate_output_path_disjoint(tmp_path / "out.c", [None, None])


def test_validate_coupling_level_accepts_none() -> None:
    assert validate_coupling_level(None) is None


def test_validate_coupling_level_accepts_positive_int() -> None:
    assert validate_coupling_level(2) == 2


@pytest.mark.parametrize("bad_level", [0, -1])
def test_validate_coupling_level_rejects_non_positive(bad_level: int) -> None:
    with pytest.raises(ValueError, match="正整数"):
        validate_coupling_level(bad_level)
