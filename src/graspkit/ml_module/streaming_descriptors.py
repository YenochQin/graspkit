from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import polars as pl


@dataclass(frozen=True)
class DescriptorShape:
    n_features: int
    channels: int
    sequence_length: int


def validate_cnn_descriptor_shape(
    n_features: int,
    channels: int = 3,
) -> DescriptorShape:
    if channels <= 0:
        raise ValueError("channels must be positive")
    if n_features % channels != 0:
        raise ValueError("n_features must be divisible by channels")

    sequence_length = n_features // channels
    if sequence_length < 3:
        raise ValueError("sequence_length must be at least 3")

    return DescriptorShape(
        n_features=n_features,
        channels=channels,
        sequence_length=sequence_length,
    )


def iter_indexed_descriptor_batches(
    descriptors: pl.LazyFrame,
    indices: NDArray[np.int64],
    batch_size: int,
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.float32]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    row_index_column = "__graspkit_descriptor_row_index"
    request_order_column = "__graspkit_request_order"
    found_column = "__graspkit_descriptor_found"
    feature_columns = descriptors.collect_schema().names()
    indexed_descriptors = descriptors.with_row_index(row_index_column).with_columns(
        pl.lit(True).alias(found_column)
    )

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        request = pl.DataFrame(
            {
                request_order_column: np.arange(len(batch_indices), dtype=np.int64),
                row_index_column: batch_indices,
            }
        ).lazy()

        batch_frame = (
            request.join(
                indexed_descriptors.filter(
                    pl.col(row_index_column).is_in(batch_indices.tolist())
                ),
                on=row_index_column,
                how="left",
            )
            .sort(request_order_column)
            .collect()
        )

        missing_indices = [
            int(index)
            for index in dict.fromkeys(
                batch_frame.filter(pl.col(found_column).is_null())
                .get_column(row_index_column)
                .to_list()
            )
        ]
        if missing_indices:
            raise IndexError(
                f"Requested descriptor row indices were not found: {missing_indices}"
            )

        batch_features = batch_frame.select(pl.col(feature_columns).cast(pl.Float32))

        yield batch_indices, batch_features.to_numpy().astype(np.float32, copy=False)


def _iter_unselected_index_chunks(
    total_csfs_count: int,
    batch_size: int,
    selected_idxs: set[int],
) -> Iterator[NDArray[np.int64]]:
    for start in range(0, total_csfs_count, batch_size):
        stop = min(start + batch_size, total_csfs_count)
        chunk = np.fromiter(
            (idx for idx in range(start, stop) if idx not in selected_idxs),
            dtype=np.int64,
        )
        if chunk.size > 0:
            yield chunk


def validate_csf_desc_coverage_streaming(
    final_sampled_idxs: NDArray[np.int64],
    raw_csfs_descriptors: pl.LazyFrame,
    total_csfs_count: int,
    batch_size: int = 100_000,
) -> NDArray[np.int64]:
    if final_sampled_idxs.size == 0:
        raise RuntimeError("final_sampled_idxs 为空，输入错误")

    selected_idxs = np.unique(final_sampled_idxs.astype(np.int64, copy=False))
    selected_idx_set = {int(idx) for idx in selected_idxs}
    n_features = len(raw_csfs_descriptors.collect_schema().names())
    electron_idxs = np.arange(0, n_features, 3, dtype=np.int64)
    covered_orbitals = np.zeros(len(electron_idxs), dtype=np.bool_)

    for _, selected_descriptors in iter_indexed_descriptor_batches(
        raw_csfs_descriptors,
        selected_idxs,
        batch_size,
    ):
        covered_orbitals |= np.any(selected_descriptors[:, electron_idxs] > 0, axis=0)

    if covered_orbitals.all():
        return final_sampled_idxs

    additional_idxs: list[int] = []
    for candidate_idxs in _iter_unselected_index_chunks(
        total_csfs_count,
        batch_size,
        selected_idx_set,
    ):
        _, candidate_descriptors = next(
            iter_indexed_descriptor_batches(
                raw_csfs_descriptors,
                candidate_idxs,
                batch_size,
            )
        )
        candidate_electrons = candidate_descriptors[:, electron_idxs]

        for row_offset, candidate_idx in enumerate(candidate_idxs):
            newly_covered = (candidate_electrons[row_offset] > 0) & ~covered_orbitals
            if not newly_covered.any():
                continue

            additional_idxs.append(int(candidate_idx))
            selected_idx_set.add(int(candidate_idx))
            covered_orbitals |= newly_covered
            if covered_orbitals.all():
                return np.unique(
                    np.concatenate(
                        [
                            selected_idxs,
                            np.array(additional_idxs, dtype=np.int64),
                        ]
                    )
                )

    if not additional_idxs:
        return selected_idxs

    return np.unique(
        np.concatenate(
            [
                selected_idxs,
                np.array(additional_idxs, dtype=np.int64),
            ]
        )
    )
