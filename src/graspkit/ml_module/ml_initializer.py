# -*- encoding: utf-8 -*-
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import polars as pl
import rtoml
from tabulate import tabulate

from ..data_IO import (
    CalPath,
    EnergyFileLoader,
    MLCalConfig,
    MixCoefLoader,
    csfs_idxs_ci_loader,
    csfs_idxs_ci_storage,
    save_descriptors,
    scan_descriptors_polars,
)
from ..grasp_data_extractor.rmix_data_processor import ci_squared
from ..utils import (
    MixCoefficientData,
    get_environment_config,
)
from .streaming_descriptors import iter_indexed_descriptor_batches


def setup_directories(root_path: Path) -> None:
    """Create standard output directories under a calculation root.

    Args:
        root_path: Root directory where ``models``, ``roc_curves``, and
            ``results`` should be created.
    """

    directories = ["models", "roc_curves", "results"]

    for dir_name in directories:
        (root_path / dir_name).mkdir(parents=True, exist_ok=True)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure environment-aware logging for ML workflows.

    Args:
        log_dir: Directory where the training log file should be written.

    Returns:
        Logger for this module after global logging has been configured.
    """
    env_config = get_environment_config()
    log_config = env_config.get_logging_config()

    # 创建日志目录
    log_dir.mkdir(exist_ok=True)

    # 配置日志级别
    log_level = getattr(logging, log_config["level"])

    # 创建处理器列表
    handlers: list[logging.Handler] = []
    handlers.append(logging.FileHandler(log_dir / "ml_training.log", encoding="utf-8"))

    # 在调试模式下添加控制台输出
    if not env_config.is_production_mode:
        handlers.append(logging.StreamHandler())

    # 配置日志
    logging.basicConfig(
        level=log_level,
        format=log_config["format"],
        datefmt="%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,  # 强制重新配置
    )

    logger = logging.getLogger(__name__)

    # 输出环境信息
    env_info = env_config.get_environment_info()
    logger.info(
        f"环境配置 - SLURM: {env_info['is_slurm']}, 调试模式: {env_info['is_debug']}, 生产模式: {env_info['is_production']}"
    )

    if env_info["slurm_job_id"]:
        logger.info(f"SLURM作业ID: {env_info['slurm_job_id']}")

    return logger


def training_data_loader(
    paths_cfg: CalPath,
    cal_method: str,
    logger: logging.Logger
) -> tuple[pl.LazyFrame, int, pl.DataFrame, MixCoefficientData, np.ndarray, int]:
    """加载当前计算轮次所需的数据文件。

    Args:
        paths_cfg: config的cal_path子类，即config.cal_path
        cal_method: 计算方法 ("rmcdhf" 或 "rci")
        logger: 日志记录器

    Returns:
        ``(raw_csfs_descriptors, total_csfs_count, energy_level_data,
        rmix_file_data, caled_csfs_idxs_array, cal_csfs_count)``.
    """

    # 加载初始 CSFs 描述符文件
    # 使用 rcsfs 生成的 parquet 文件
    raw_desc_file_path = paths_cfg.full_CSFs_set_desc_path.with_suffix(".parquet")

    raw_csfs_descriptors, parquet_meta_data = scan_descriptors_polars(
        raw_desc_file_path
    )
    logger.info(f"加载 parquet raw_CSFs 描述符: {raw_desc_file_path}")
    raw_csfs_desc_count = parquet_meta_data["n_rows"]

    raw_csfs_header_file = paths_cfg.full_CSFs_set_header_path
    csfs_header = rtoml.load(raw_csfs_header_file)
    raw_csfs_num = csfs_header.get("conversion_stats", {}).get("csf_count", 0)

    if raw_csfs_num == 0:
        error_msg = f"raw_csfs_header_file: {str(raw_csfs_header_file)} 中没有'csf_count'数据请重新运行initial_csfs.py"
        logger.error(error_msg)
        raise ValueError(error_msg)
    elif raw_csfs_desc_count != raw_csfs_num:
        error_msg = f"raw_csfs_header_file: {str(raw_csfs_header_file)} 中'csf_count'数值与描述符文件长度不一致"
        logger.error(error_msg)
        raise ValueError(error_msg)
    else:
        total_csfs_count = raw_csfs_desc_count

    # 加载能级文件
    energy_level_file_path = (
        paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}.level"
    )
    energy_level_file_load = EnergyFileLoader(energy_level_file_path)
    energy_level_data = energy_level_file_load.load()
    logger.info(f"加载能级数据: {energy_level_file_path}")

    # 加载rmix文件
    # 根据计算轮次确定文件后缀
    if cal_method == "rmcdhf":
        rmix_file_path = paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}.m"
    elif cal_method == "rci":
        rmix_file_path = paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}.cm"
    else:
        raise ValueError(f"不支持的计算方法: {cal_method}")

    rmix_file_load = MixCoefLoader(rmix_file_path)
    rmix_file_data = rmix_file_load.load()
    logger.info(f"加载 mix coefficient 文件数据: {rmix_file_path}")

    first_mix_block = rmix_file_data.blocks[0]
    csfs_count_from_rmix = first_mix_block.csf_count

    # 加载本轮选择的CSFs的索引文件
    caled_csfs_idxs_file_path = (
        paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}_sampled_idxs.npy"
    )
    caled_csfs_idxs_array = np.load(caled_csfs_idxs_file_path)
    logger.info(f"加载本轮选择的 CSFs 的索引文件: {caled_csfs_idxs_file_path}")

    if caled_csfs_idxs_array.shape[0] != csfs_count_from_rmix:
        logger.error(
            f"rmix_file_data.blocks[0].csf_count={csfs_count_from_rmix}, {caled_csfs_idxs_array.shape[0]=}"
        )
        raise ValueError("本轮计算的CSFs数量数据不一致，请检查数据文件")
    cal_csfs_count = csfs_count_from_rmix

    return (
        raw_csfs_descriptors,
        total_csfs_count,
        energy_level_data,
        rmix_file_data,
        caled_csfs_idxs_array,
        cal_csfs_count,
    )


def check_configuration_coupling(
    paths_cfg: CalPath,
    energy_level_data: pl.DataFrame,
    rmix_file_data: MixCoefficientData,
    spectral_term: list[str],
    cal_loop_num: int,
    logger: logging.Logger,
) -> tuple[bool, pl.DataFrame | None, NDArray[np.float64] | None]:
    """检查组态耦合是否正确

    优化版本：一次遍历完成计数和位置记录，时间复杂度从 O(n*m) 降至 O(n)
    """
    cal_configuration_list: list[str] = energy_level_data["configuration_raw"].to_list()

    # 优化1: 一次遍历同时构建计数和位置映射
    term_positions: dict[str, list[int]] = {}
    actual_counts: Counter[str] = Counter()

    for idx, term in enumerate(cal_configuration_list):
        if term in spectral_term:
            actual_counts[term] += 1
            term_positions.setdefault(term, []).append(idx)

    # 优化2: 使用Counter构建期望计数
    expected_counts = Counter(spectral_term)

    # 检查并收集位置
    spectral_term_positions: list[int] = []
    errors: list[str] = []

    for term in expected_counts:
        expected = expected_counts[term]
        actual = actual_counts.get(term, 0)
        positions = term_positions.get(term, [])

        if actual == expected:
            spectral_term_positions.extend(positions)
            if expected == 1:
                logger.info(f"光谱项 '{term}' 在位置 {positions[0]} 找到")
            else:
                logger.info(
                    f"光谱项 '{term}' 在位置 {positions} 找到（期望{expected}次，实际{actual}次）"
                )
        elif actual == 0:
            errors.append(f"光谱项 '{term}' 未找到")
        else:
            errors.append(f"光谱项 '{term}' 出现{actual}次，期望{expected}次")

    # 优化3: 提前返回模式，减少嵌套
    if errors:
        for err in errors:
            logger.error(err)
        error_msg = f"cal_loop {cal_loop_num} 组态耦合错误"
        logger.error(error_msg)
        return False, None, None

    # 成功路径
    # 注意：排序后 selected_energy_data 的行顺序为能量文件中的升序位置，
    # 而非 spectral_term 列表的顺序。下游代码不得用行索引 i 与
    # spectral_term[i] 对应，必须通过 configuration_raw 列做字典匹配。
    spectral_term_positions.sort()
    logger.info(
        f"cal_loop {cal_loop_num} 组态耦合正确，位置索引: {spectral_term_positions}"
    )

    selected_energy_data = energy_level_data[spectral_term_positions]
    correct_levels_ci = rmix_file_data.blocks[0].mix_coefficients[spectral_term_positions]
    correct_levels_csv_path = (
        paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}_correct_levels.csv"
    )
    selected_energy_data.write_csv(correct_levels_csv_path)
    logger.info(f"正确的能级数据已保存到: {correct_levels_csv_path}")

    return True, selected_energy_data, correct_levels_ci


def _load_loop_energy_csv(config: MLCalConfig, loop_num: int) -> pl.DataFrame | None:
    """Load ``correct_levels.csv`` for a previous calculation loop.

    Args:
        config: ML calculation configuration.
        loop_num: Calculation loop number to load.

    Returns:
        DataFrame if the file exists, otherwise None.
    """
    csv_path = (
        config.cal_settings.root_path
        / f"{config.target.conf}_{loop_num}"
        / f"{config.target.conf}_{loop_num}_correct_levels.csv"
    )
    if not csv_path.exists():
        return None
    return pl.read_csv(csv_path)


def _get_reference_discrepancies(
    df: pl.DataFrame,
    spectral_term: list[str],
    reference_energy_levels: list[float],
) -> list[float] | None:
    """从 correct_levels DataFrame 计算各谱项相对于参考值的能级间距偏差

    Args:
        df: correct_levels.csv 的 DataFrame
        spectral_term: 谱项列表
        reference_energy_levels: 参考能级值（cm⁻¹），与 spectral_term 一一对应

    Returns:
        各谱项的偏差列表（cm⁻¹），数据不完整时返回 None
    """
    term_energy: dict[str, float] = {}
    for row in df.iter_rows(named=True):
        term = row["configuration_raw"]
        if term in spectral_term:
            term_energy[term] = float(row["EnergyLevel"])

    if len(term_energy) < len(spectral_term):
        return None

    ref_base = min(reference_energy_levels)
    base_idx = reference_energy_levels.index(ref_base)
    calc_base = term_energy[spectral_term[base_idx]]

    return [
        abs((term_energy[term] - calc_base) - (reference_energy_levels[i] - ref_base))
        for i, term in enumerate(spectral_term)
    ]


def compute_reference_gap_errors(
    selected_energy_data: pl.DataFrame,
    spectral_term: list[str],
    reference_energy_levels: list[float],
) -> pl.DataFrame:
    """Compute relative-energy errors against reference levels.

    Args:
        selected_energy_data: DataFrame containing selected level energies.
        spectral_term: Target spectral terms in the expected output order.
        reference_energy_levels: Reference energies in ``cm^-1`` aligned with
            ``spectral_term``.

    Returns:
        DataFrame with reference-relative energy, calculated-relative energy,
        and absolute gap error for each target term.

    Raises:
        ValueError: If any target spectral term is missing from the energy data.
    """
    term_energy: dict[str, float] = {}
    for row in selected_energy_data.iter_rows(named=True):
        term = row["configuration_raw"]
        if term in spectral_term:
            term_energy[term] = float(row["EnergyLevel"])

    missing = [term for term in spectral_term if term not in term_energy]
    if missing:
        raise ValueError(f"以下谱项在能级数据中找不到: {missing}")

    ref_base = min(reference_energy_levels)
    base_idx = reference_energy_levels.index(ref_base)
    calc_base = term_energy[spectral_term[base_idx]]

    rows: list[dict[str, str | float]] = []
    for idx, term in enumerate(spectral_term):
        ref_rel = float(reference_energy_levels[idx] - ref_base)
        calc_rel = float(term_energy[term] - calc_base)
        rows.append(
            {
                "configuration_raw": term,
                "reference_relative_energy": ref_rel,
                "calculated_relative_energy": calc_rel,
                "gap_error": abs(calc_rel - ref_rel),
            }
        )
    return pl.DataFrame(rows)


def compute_pairwise_gap_error_matrix(
    selected_energy_data: pl.DataFrame,
    spectral_term: list[str],
    reference_energy_levels: list[float],
) -> NDArray[np.float64]:
    """Compute pairwise relative-energy gap errors between target terms.

    Args:
        selected_energy_data: DataFrame containing selected level energies.
        spectral_term: Target spectral terms in the expected output order.
        reference_energy_levels: Reference energies in ``cm^-1`` aligned with
            ``spectral_term``.

    Returns:
        Square matrix whose ``(i, j)`` entry is the absolute error between the
        calculated and reference energy gaps for terms ``i`` and ``j``.
    """
    reference_gap_errors = compute_reference_gap_errors(
        selected_energy_data,
        spectral_term,
        reference_energy_levels,
    )
    calc_rel = np.asarray(
        reference_gap_errors["calculated_relative_energy"], dtype=np.float64
    )
    ref_rel = np.asarray(
        reference_gap_errors["reference_relative_energy"], dtype=np.float64
    )

    calc_gap = calc_rel[:, np.newaxis] - calc_rel[np.newaxis, :]
    ref_gap = ref_rel[:, np.newaxis] - ref_rel[np.newaxis, :]
    return np.abs(calc_gap - ref_gap)


def score_correction_candidates_from_ci(
    candidate_level_scores: NDArray[np.float64],
    pairwise_gap_error_matrix: NDArray[np.float64],
    pair_weighting: str = "error_magnitude",
    top_pair_count: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Score candidate CSFs by their ability to correct reference-gap errors.

    Args:
        candidate_level_scores: Per-candidate, per-level model scores.
        pairwise_gap_error_matrix: Pairwise relative-energy gap error matrix.
        pair_weighting: Strategy for weighting level-pair errors.
        top_pair_count: Optional number of highest-error level pairs to use.

    Returns:
        Tuple of correction scores and dominant level-pair indices for each
        candidate.
    """
    n_candidates, n_levels = candidate_level_scores.shape
    correction_scores = np.zeros(n_candidates, dtype=np.float64)
    dominant_pairs = np.full((n_candidates, 2), -1, dtype=np.int64)
    dominant_pair_scores = np.full(n_candidates, -np.inf, dtype=np.float64)

    candidate_level_scores = np.asarray(candidate_level_scores, dtype=np.float64)
    pairwise_gap_error_matrix = np.asarray(pairwise_gap_error_matrix, dtype=np.float64)

    pair_entries: list[tuple[float, int, int]] = []
    for left_idx in range(n_levels):
        for right_idx in range(left_idx + 1, n_levels):
            gap_error = float(pairwise_gap_error_matrix[left_idx, right_idx])
            pair_entries.append((gap_error, left_idx, right_idx))

    pair_entries.sort(reverse=True)
    if top_pair_count is not None:
        pair_entries = pair_entries[:top_pair_count]

    for gap_error, left_idx, right_idx in pair_entries:
        if pair_weighting == "uniform":
            pair_weight = 1.0 if gap_error > 0 else 0.0
        elif pair_weighting == "error_squared":
            pair_weight = gap_error**2
        else:
            pair_weight = gap_error

        if pair_weight <= 0:
            continue

        pair_contribution = pair_weight * np.abs(
            candidate_level_scores[:, left_idx] - candidate_level_scores[:, right_idx]
        )
        correction_scores += pair_contribution

        better_pair_mask = pair_contribution > dominant_pair_scores
        dominant_pair_scores[better_pair_mask] = pair_contribution[better_pair_mask]
        dominant_pairs[better_pair_mask, 0] = left_idx
        dominant_pairs[better_pair_mask, 1] = right_idx

    return correction_scores, dominant_pairs


def combine_importance_and_reference_scores(
    importance_scores: NDArray[np.float64],
    reference_correction_scores: NDArray[np.float64],
    importance_weight: float,
    correction_weight: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Normalize and combine model-importance and reference-correction scores.

    Args:
        importance_scores: Base model importance score for each candidate.
        reference_correction_scores: Reference-energy correction score for
            each candidate.
        importance_weight: Weight assigned to normalized model importance.
        correction_weight: Weight assigned to normalized correction scores.

    Returns:
        Tuple of final combined scores, normalized importance scores, and
        normalized reference-correction scores.

    Raises:
        ValueError: If the sum of the two weights is not positive.
    """

    def _normalize_scores(scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize a score vector to the range ``[0, 1]``.

        Args:
            scores: Raw score vector.

        Returns:
            Normalized score vector with stable handling for empty or constant
            inputs.
        """
        if scores.size == 0:
            return scores.astype(np.float64)
        min_value = float(np.min(scores))
        max_value = float(np.max(scores))
        if np.isclose(min_value, max_value):
            if max_value <= 0:
                return np.zeros_like(scores, dtype=np.float64)
            return np.ones_like(scores, dtype=np.float64)
        return (scores - min_value) / (max_value - min_value)

    normalized_importance = _normalize_scores(np.asarray(importance_scores, dtype=np.float64))
    normalized_reference = _normalize_scores(
        np.asarray(reference_correction_scores, dtype=np.float64)
    )

    total_weight = importance_weight + correction_weight
    if total_weight <= 0:
        raise ValueError("importance_weight 和 correction_weight 的和必须大于 0")

    final_scores = (
        importance_weight * normalized_importance
        + correction_weight * normalized_reference
    ) / total_weight
    return final_scores, normalized_importance, normalized_reference


def _build_diff_ci_positive_mask(
    accumulated_ci_squared: NDArray[np.float64],
    config: MLCalConfig,
    selected_energy_data: pl.DataFrame | None,
) -> NDArray[np.bool_]:
    """Build extra positive labels from differential CI ranking criteria."""
    if (
        selected_energy_data is None
        or config.cal_settings.diff_ci_cutoff <= 0
        or len(config.cal_settings.reference_energy_levels) <= 1
    ):
        return np.zeros_like(accumulated_ci_squared, dtype=bool)

    diff_ci_cutoff = config.cal_settings.diff_ci_cutoff
    spectral_terms = config.cal_settings.spectral_term
    ref_energy_list = config.cal_settings.reference_energy_levels
    ref_zero = min(ref_energy_list)
    ref_zero_idx = ref_energy_list.index(ref_zero)

    term_to_ci_row: dict[str, int] = {}
    term_to_calc_energy: dict[str, float] = {}
    for ci_row_idx, energy_row in enumerate(selected_energy_data.iter_rows(named=True)):
        term_name = energy_row["configuration_raw"]
        if term_name in spectral_terms:
            term_to_ci_row[term_name] = ci_row_idx
            term_to_calc_energy[term_name] = float(energy_row["EnergyLevel"])

    if len(term_to_ci_row) != len(spectral_terms):
        return np.zeros_like(accumulated_ci_squared, dtype=bool)

    calc_zero_energy = term_to_calc_energy[spectral_terms[ref_zero_idx]]
    diff_mask = np.zeros_like(accumulated_ci_squared, dtype=bool)

    for outer_idx in range(len(spectral_terms)):
        for inner_idx in range(outer_idx + 1, len(spectral_terms)):
            outer_term = spectral_terms[outer_idx]
            inner_term = spectral_terms[inner_idx]
            outer_ref_rel = ref_energy_list[outer_idx] - ref_zero
            inner_ref_rel = ref_energy_list[inner_idx] - ref_zero

            if outer_ref_rel >= inner_ref_rel:
                high_term, low_term = outer_term, inner_term
                ref_gap_high_minus_low = outer_ref_rel - inner_ref_rel
            else:
                high_term, low_term = inner_term, outer_term
                ref_gap_high_minus_low = inner_ref_rel - outer_ref_rel

            if ref_gap_high_minus_low == 0.0:
                continue

            high_ci_row = term_to_ci_row[high_term]
            low_ci_row = term_to_ci_row[low_term]
            high_calc_rel = term_to_calc_energy[high_term] - calc_zero_energy
            low_calc_rel = term_to_calc_energy[low_term] - calc_zero_energy
            calc_gap_high_minus_low = high_calc_rel - low_calc_rel

            if calc_gap_high_minus_low > ref_gap_high_minus_low:
                ci_diff_for_high = (
                    accumulated_ci_squared[high_ci_row]
                    - accumulated_ci_squared[low_ci_row]
                )
                diff_mask[high_ci_row] |= ci_diff_for_high >= diff_ci_cutoff
            elif calc_gap_high_minus_low < ref_gap_high_minus_low:
                ci_diff_for_low = (
                    accumulated_ci_squared[low_ci_row]
                    - accumulated_ci_squared[high_ci_row]
                )
                diff_mask[low_ci_row] |= ci_diff_for_low >= diff_ci_cutoff

    return diff_mask


def _build_important_csfs_mask(
    accumulated_ci_squared: NDArray[np.float64],
    cutoff_value: float,
    config: MLCalConfig | None = None,
    selected_energy_data: pl.DataFrame | None = None,
) -> NDArray[np.bool_]:
    important_csfs_mask = (accumulated_ci_squared >= cutoff_value).T
    if config is None:
        return important_csfs_mask
    return important_csfs_mask | _build_diff_ci_positive_mask(
        accumulated_ci_squared,
        config,
        selected_energy_data,
    ).T


def check_reference_energy_agreement(
    selected_energy_data: pl.DataFrame,
    spectral_term: list[str],
    reference_energy_levels: list[float],
    reference_energy_threshold: float,
    logger: logging.Logger,
) -> bool:
    """监控并检查计算能级间距与参考值（如NIST）之间的偏差

    以 reference_energy_levels 中最小值对应的谱项为参考零点，比较计算值与参考值
    的相对能级差，输出对比表格；当 reference_energy_threshold > 0 时返回是否满足阈值。

    Args:
        selected_energy_data: 包含目标谱项的 Polars DataFrame（来自 check_configuration_coupling）
        spectral_term: 谱项列表
        reference_energy_levels: 参考能级值（cm⁻¹），与 spectral_term 一一对应
        reference_energy_threshold: 允许的最大偏差（cm⁻¹），0 表示仅监控不阻断
        logger: 日志记录器

    Returns:
        bool: True=满足阈值或纯监控模式, False=超出阈值（用于收敛门控）
    """
    try:
        reference_gap_errors = compute_reference_gap_errors(
            selected_energy_data,
            spectral_term,
            reference_energy_levels,
        )
    except ValueError as exc:
        logger.error("%s，判定为计算错误", exc)
        return False

    table_rows = [
        [
            str(row["configuration_raw"]),
            f"{float(row['reference_relative_energy']):.2f}",
            f"{float(row['calculated_relative_energy']):.2f}",
            f"{float(row['gap_error']):.2f}",
        ]
        for row in reference_gap_errors.iter_rows(named=True)
    ]
    discrepancies = [
        float(value) for value in reference_gap_errors["gap_error"].to_list()
    ]

    table = tabulate(
        table_rows,
        headers=["谱项", "参考相对值 (cm⁻¹)", "计算相对值 (cm⁻¹)", "偏差 (cm⁻¹)"],
        tablefmt="simple",
    )
    logger.info(f"能级间距对比:\n{table}")
    logger.info(
        "当前轮参考能级阈值: %.1f cm⁻¹",
        reference_energy_threshold,
    )

    if reference_energy_threshold > 0:
        within_threshold = all(d < reference_energy_threshold for d in discrepancies)
        logger.info(
            f"  参考能级间距阈值检查: {'满足' if within_threshold else '超出'}"
            f" (阈值: {reference_energy_threshold:.1f} cm⁻¹)"
        )
        return within_threshold
    return True


def check_energy_convergence(
    config: MLCalConfig,
    logger: logging.Logger,
    current_energy_data: pl.DataFrame,
    increase_tolerance: float = 1e-4,
) -> bool:
    """
    检查能量收敛性：验证当前轮能量符合变分原理（单调不增）

    在 MCDHF/RCI 变分计算中，随着组态空间扩大，束缚态总能量应单调下降
    （变分原理保证）。若当前轮某能级较上一轮显著升高，则表明计算出现异常
    （如线性相关、收敛到错误态或数值不稳定），需要回退重算。

    能量大幅下降属于正常收敛过程，不触发回退。

    Args:
        config: 配置对象
        logger: 日志记录器
        current_energy_data: 当前轮的能量数据DataFrame
        increase_tolerance: 允许的能量上升上限（Hartree），超过此值视为异常，
                            默认为 1e-4 Hartree（0.1 mHartree）

    Returns:
        bool: True表示继续计算（能量正常下降或稳定），False表示需要回退到上一轮重算
    """

    try:
        previous_loop = config.cal_settings.cal_loop_num - 1
        previous_energy_data = _load_loop_energy_csv(config, previous_loop)
        if previous_energy_data is None:
            logger.warning(
                f"未找到上一轮能量数据: "
                f"{config.target.conf}_{previous_loop}_correct_levels.csv"
            )
            return True

        # 检查数据一致性
        if len(current_energy_data) != len(previous_energy_data):
            logger.warning(
                f"当前轮与能量数据数量不一致: 当前{len(current_energy_data)} vs 上轮{len(previous_energy_data)}"
            )
            return True

        # 获取能量列
        current_energies = np.asarray(
            current_energy_data["EnergyTotal"], dtype=np.float64
        )
        previous_energies = np.asarray(
            previous_energy_data["EnergyTotal"], dtype=np.float64
        )

        # 计算带符号的能量变化（正值=上升，负值=下降）
        energy_changes = current_energies - previous_energies
        max_increase = np.max(energy_changes)
        max_decrease = np.min(energy_changes)

        logger.info("能量变化统计 (当前轮 - 上一轮):")
        logger.info(f"  最大上升量: {max_increase:+.7f} Hartree")
        logger.info(f"  最大下降量: {max_decrease:+.7f} Hartree")

        # 核心检查：能量不应显著上升（违反变分原理）
        if max_increase > increase_tolerance:
            max_increase_idx = int(np.argmax(energy_changes))
            logger.warning(
                f"检测到能量异常上升（违反变分原理）: "
                f"第 {max_increase_idx + 1} 个能级上升 {max_increase:.7f} Hartree"
                f"（容差: {increase_tolerance:.1e} Hartree）"
            )
            logger.warning(f"  当前轮能量: {current_energies[max_increase_idx]:.7f} Hartree")
            logger.warning(f"  上一轮能量: {previous_energies[max_increase_idx]:.7f} Hartree")
            return False

        logger.info("能量变化方向正常（单调下降或数值稳定），继续计算")

        # === 趋势回退：参考能级间距单调递增检查 ===
        if (
            config.cal_settings.reference_energy_levels
            and config.cal_settings.reference_energy_threshold > 0
            and config.cal_settings.cal_loop_num >= 3
        ):
            current_loop = config.cal_settings.cal_loop_num
            spectral_term = config.cal_settings.spectral_term
            reference_energy_levels = config.cal_settings.reference_energy_levels

            # 加载前两轮 CSV；当前轮直接使用传入的 current_energy_data，避免重复 I/O
            recent_discrepancies: list[list[float]] = []
            all_loaded = True
            for ln in range(current_loop - 2, current_loop):
                frame = _load_loop_energy_csv(config, ln)
                if frame is None:
                    all_loaded = False
                    break
                disc = _get_reference_discrepancies(frame, spectral_term, reference_energy_levels)
                if disc is None:
                    all_loaded = False
                    break
                recent_discrepancies.append(disc)
            # 当前轮使用传入的 current_energy_data
            if all_loaded:
                curr_disc = _get_reference_discrepancies(
                    current_energy_data, spectral_term, reference_energy_levels
                )
                if curr_disc is None:
                    all_loaded = False
                else:
                    recent_discrepancies.append(curr_disc)

            if all_loaded and len(recent_discrepancies) == 3:
                n_terms = len(spectral_term)
                monotonically_increasing = all(
                    recent_discrepancies[0][k] < recent_discrepancies[1][k] < recent_discrepancies[2][k]
                    for k in range(n_terms)
                )
                if monotonically_increasing:
                    logger.warning(
                        "检测到能级间距偏差连续3轮单调递增（持续远离参考值），触发回退"
                    )
                    return False
                else:
                    logger.info("能级间距偏差无单调递增趋势，继续计算")

        return True

    except (FileNotFoundError, pl.exceptions.ColumnNotFoundError) as e:
        logger.exception("能量收敛检查过程中发生错误: %s", e)
        return True  # 无法判断时，默认继续（不触发回退）


def evaluate_calculation_convergence(
    config: MLCalConfig,
    logger: logging.Logger,
    cal_loop_csfs_count: int,
    current_energy_data: pl.DataFrame | None = None,
) -> bool:
    """
    检查GRASP计算的收敛性

    使用最近三次计算结果的能级数据和组态数量数据，通过计算：
    1. 三轮共有能级的跨轮次标准差均值（能级收敛）
    2. 组态数量的相对标准差（CSF数量稳定性）
    来判定收敛。两个条件同时满足才停止计算。

    Args:
        config: 配置对象
        logger: 日志记录器
        cal_loop_csfs_count: 当前轮的CSFs数量

    Returns:
        bool: True 表示继续计算，False 表示已收敛停止计算
    """
    try:
        current_loop = config.cal_settings.cal_loop_num

        # === 1. 加载最近三轮能级数据，取列 configuration_raw 和 EnergyTotal ===
        energy_frames: list[pl.DataFrame] = []
        for i, loop_num in enumerate(range(current_loop - 2, current_loop + 1)):
            frame = _load_loop_energy_csv(config, loop_num)
            if frame is None:
                logger.warning(f"未找到第{loop_num}轮能级数据")
                return True
            energy_frames.append(
                frame
                .select(["configuration_raw", "EnergyTotal"])
                .rename({"EnergyTotal": f"E{i}"})
            )
            logger.info(f"读取第{loop_num}轮能级数据")

        # === 2. 三轮内连接取共有能级，向量化计算跨轮标准差均值 ===
        merged = (
            energy_frames[0]
            .join(energy_frames[1], on="configuration_raw", how="inner")
            .join(energy_frames[2], on="configuration_raw", how="inner")
        )
        if merged.height < energy_frames[0].height:
            logger.warning(
                f"部分 configuration 在某轮数据中缺失，"
                f"仅对 {merged.height}/{energy_frames[0].height} 个共有能级计算收敛性"
            )
        energies: NDArray[np.float64] = np.asarray(merged.select(["E0", "E1", "E2"]).to_numpy(), dtype=np.float64)
        per_level_std: NDArray[np.float64] = np.std(energies, axis=1)
        avg_energy_std: float = np.mean(per_level_std).item()

        # === 3. 加载三轮 CSF 数量（前两轮来自文件，当前轮来自参数）===
        iteration_df = pl.read_csv(config.cal_path.iteration_results)
        csfs_num: list[int] = []
        for loop_num in range(current_loop - 2, current_loop):
            row = iteration_df.filter(pl.col("cal_loop_num") == loop_num)
            if row.height == 0:
                logger.warning(f"未在 iteration_results.csv 中找到第{loop_num}轮数据")
                return True
            count = int(row["current_calculation_count"][0])
            csfs_num.append(count)
            logger.info(f"读取第{loop_num}轮组态数量: {count}")
        csfs_num.append(cal_loop_csfs_count)
        logger.info(f"读取第{current_loop}轮组态数量: {cal_loop_csfs_count}")

        # === 4. 计算 CSF 数量相对标准差 ===
        csfs_mean = float(np.mean(csfs_num))
        csfs_std = float(np.std(csfs_num))
        csfs_relative_std = csfs_std / csfs_mean if csfs_mean > 0 else csfs_std

        # === 5. 读取阈值并判断收敛 ===
        energy_threshold = config.cal_settings.energy_std_threshold
        csfs_threshold = config.cal_settings.csfs_num_relative_std_threshold

        energy_converged = avg_energy_std < energy_threshold
        csfs_converged = csfs_relative_std < csfs_threshold

        logger.info("收敛性统计:")
        logger.info(f"  最近3轮组态数量: {csfs_num}")
        logger.info(
            f"  组态数量相对标准差: {csfs_relative_std:.4f}"
            f" (阈值: {csfs_threshold:.4f}) → {'收敛' if csfs_converged else '未收敛'}"
        )
        logger.info(
            f"  能级平均标准差:     {avg_energy_std:.5e}"
            f" (阈值: {energy_threshold:.5e}) → {'收敛' if energy_converged else '未收敛'}"
        )

        # === 6. 参考能级间距收敛判据（可选）===
        ref_energy_converged = True
        if (
            config.cal_settings.reference_energy_levels
            and config.cal_settings.reference_energy_threshold > 0
            and current_energy_data is not None
        ):
            logger.info(
                "参考能级阈值策略: base=%.1f, current=%.1f, min=%.1f, start_loop=%s, decay=%.3f",
                config.cal_settings.reference_energy_threshold_base,
                config.cal_settings.reference_energy_threshold,
                config.cal_settings.reference_energy_threshold_min,
                config.cal_settings.reference_energy_threshold_tighten_start_loop,
                config.cal_settings.reference_energy_threshold_decay,
            )
            ref_energy_converged = check_reference_energy_agreement(
                current_energy_data,
                config.cal_settings.spectral_term,
                config.cal_settings.reference_energy_levels,
                config.cal_settings.reference_energy_threshold,
                logger,
            )
            logger.info(
                f"  参考能级间距收敛: {'满足' if ref_energy_converged else '未满足'}"
                f" (阈值: {config.cal_settings.reference_energy_threshold:.1f} cm⁻¹)"
            )

        if energy_converged and csfs_converged and ref_energy_converged:
            logger.info("能级、组态数量、参考能级间距均已收敛，停止计算")
            return False

        if not energy_converged:
            logger.info("能级未完全收敛，继续计算")
        if not csfs_converged:
            logger.info("组态数量未稳定收敛，继续计算")
        return True

    except (FileNotFoundError, pl.exceptions.ColumnNotFoundError) as e:
        logger.exception("收敛性检查过程中发生错误: %s", e)
        return True


def merge_historical_ci_data(
    previous_idxs: NDArray[np.int64],
    previous_ci_squared: NDArray[np.float64],
    current_idxs: NDArray[np.int64],
    current_ci_squared: NDArray[np.float64], 
    logger: logging.Logger
) -> tuple[np.ndarray, np.ndarray]:
    """将历史轮次与当前轮次的 CI 系数平方数据合并为累积数据集。

    合并规则：

    - 索引取并集，覆盖写入：先写历史数据，再用当前数据覆盖，
      交集 CSF 的 CI 系数以当前轮次为准（反映最新自洽场收敛结果）。
    - 并集索引经 np.union1d 排序，保证输出有序。

    Args:
        previous_idxs: 前序轮次累积的 CSF 索引，一维，shape (n_prev,)
        previous_ci_squared: 前序轮次对应的 CI 系数平方，二维，shape (n_levels, n_prev)
        current_idxs: 当前轮次参与计算的 CSF 索引，一维，shape (n_curr,)
        current_ci_squared: 当前轮次对应的 CI 系数平方，二维，shape (n_levels, n_curr)
        logger: 日志记录器

    Returns:
        ``(merged_idxs, merged_ci_squared)``，其中 ``merged_idxs`` 为排序后的
        CSF 索引，``merged_ci_squared`` 为形状 ``(n_levels, n_merged)`` 的
        CI 系数平方数组。
    """

    # 取索引并集，np.union1d 保证结果升序且无重复
    merged_idxs = np.union1d(previous_idxs, current_idxs)

    # 用 searchsorted 在有序 merged_idxs 中定位历史/当前 CSF 各自的落点
    prev_pos = np.searchsorted(merged_idxs, previous_idxs)
    curr_pos = np.searchsorted(merged_idxs, current_idxs)

    # 以转置视图 (n_merged, n_levels) 按行赋值，避免非连续切片
    # 先填历史数据，再用当前数据逐行覆盖——交集位置自动取当前轮次值
    merged_ci_t = np.empty(
        (len(merged_idxs), previous_ci_squared.shape[0]),
        dtype=previous_ci_squared.dtype,
    )
    merged_ci_t[prev_pos] = previous_ci_squared.T
    merged_ci_t[curr_pos] = current_ci_squared.T
    merged_ci_squared = merged_ci_t.T  # 转回 (n_levels, n_merged)

    # |交集| = |历史| + |当前| - |并集|，无需额外构建 set
    n_repeated = len(previous_idxs) + len(current_idxs) - len(merged_idxs)
    n_new = len(current_idxs) - n_repeated

    logger.info("历史数据合并统计:")
    logger.info(f"历史数据CSFs数量: {len(previous_idxs)}")
    logger.info(f"当前数据CSFs数量: {len(current_idxs)}")
    logger.info(f"合并后CSFs数量: {len(merged_idxs)}")
    logger.info(f"新增CSFs数量: {n_new}")
    logger.info(f"重复CSFs数量: {n_repeated}")

    return merged_idxs, merged_ci_squared


def ci_idx_data_processor(
    correct_levels_ci: NDArray[np.float64],
    caled_csfs_idxs_array: NDArray[np.int64],
    config: MLCalConfig,
    logger: logging.Logger,
) -> np.ndarray:
    """
    处理CI系数数据，并与历史数据合并后保存

    Args:
        correct_levels_ci: 正确能级位置的CI系数
        caled_csfs_idxs_array: 当前计算的CSF索引数组
        config: 配置对象
        logger: 日志记录器

    Returns:
        np.ndarray: CI系数平方数组
    """

    # 保存正确能级位置的 CI 系数平方对应 CSF 总池索引（用于历史数据累积）
    correct_levels_ci_squared = ci_squared(np.atleast_2d(correct_levels_ci))

    accumulated_idxs_ci_path = config.cal_path.accumulated_idxs_ci_path
    accumulated_idxs: NDArray[np.int64]
    accumulated_ci_squared: NDArray[np.float64]
    # 如果是第二轮及之后，读取历史数据并合并
    if config.cal_settings.cal_loop_num > 1:
        if accumulated_idxs_ci_path.exists():
            logger.info("读取历史CI系数数据并进行合并")

            previous_idxs, previous_ci_squared = csfs_idxs_ci_loader(accumulated_idxs_ci_path)

            accumulated_idxs, accumulated_ci_squared = merge_historical_ci_data(
                previous_idxs,
                previous_ci_squared,
                caled_csfs_idxs_array,
                correct_levels_ci_squared,
                logger
            )
        else:
            raise FileNotFoundError(
                f"历史CI数据文件不存在: {accumulated_idxs_ci_path}，"
                f"第 {config.cal_settings.cal_loop_num} 轮计算依赖前序轮次保存的累积数据"
            )
            
    else:
        # 第一轮直接使用当前数据
        accumulated_idxs = caled_csfs_idxs_array
        accumulated_ci_squared = correct_levels_ci_squared

    # 保存合并后的累积数据到 accumulated_idxs_ci_path（供下次计算使用）

    csfs_idxs_ci_storage(accumulated_idxs_ci_path, accumulated_idxs, accumulated_ci_squared)
    logger.info(
        f"保存累积CI系数数据: {accumulated_idxs_ci_path}(包含{len(accumulated_idxs)}个CSFs)"
    )

    return correct_levels_ci_squared


def build_labeled_training_array_from_lazy_descriptors(
    raw_csfs_descriptors: pl.LazyFrame,
    accumulated_idxs: NDArray[np.int64],
    accumulated_ci_squared: NDArray[np.float64],
    cutoff_value: float,
    batch_size: int = 100_000,
    *,
    config: MLCalConfig | None = None,
    selected_energy_data: pl.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Build labeled training rows from a lazy descriptor source.

    Only rows referenced by ``accumulated_idxs`` are materialized. Missing or
    out-of-range index validation is delegated to ``iter_indexed_descriptor_batches``.
    """
    important_csfs_mask = _build_important_csfs_mask(
        accumulated_ci_squared,
        cutoff_value,
        config=config,
        selected_energy_data=selected_energy_data,
    )
    if logger is not None and config is not None:
        diff_count = int(
            np.sum(
                _build_diff_ci_positive_mask(
                    accumulated_ci_squared,
                    config,
                    selected_energy_data,
                )
            )
        )
        if diff_count > 0:
            logger.info(
                f"差动CI加权新增正样本: {diff_count} "
                f"(diff_ci_cutoff={config.cal_settings.diff_ci_cutoff})"
            )

    if len(accumulated_idxs) == 0:
        feature_count = len(raw_csfs_descriptors.collect_schema().names())
        label_count = accumulated_ci_squared.shape[0]
        return np.empty((0, feature_count + label_count), dtype=np.float32)

    labeled_batches: list[np.ndarray] = []
    row_offset = 0
    for _, descriptor_batch in iter_indexed_descriptor_batches(
        raw_csfs_descriptors,
        indices=accumulated_idxs,
        batch_size=batch_size,
    ):
        next_offset = row_offset + descriptor_batch.shape[0]
        label_batch = important_csfs_mask[row_offset:next_offset].astype(
            descriptor_batch.dtype,
            copy=False,
        )
        labeled_batches.append(np.column_stack([descriptor_batch, label_batch]))
        row_offset = next_offset

    return np.vstack(labeled_batches)


def generate_train_csfs_descriptors(
    config: MLCalConfig,
    raw_csfs_descriptors: NDArray[np.float64],
    logger: logging.Logger,
    selected_energy_data: pl.DataFrame | None = None,
) -> np.ndarray:
    """
    生成用于机器学习训练的CSFs描述符数据

    优化说明：
    - 数据合并逻辑已移至 ci_idx_data_processor 函数
    - 本函数直接读取合并后的累积数据，避免冗余读写操作
    - previous_idxs_ci_path 包含所有历史轮次的数据并集

    Args:
        config: 配置对象
        raw_csfs_descriptors: 原始CSFs描述符数组
        logger: 日志记录器
        selected_energy_data: 当前轮的目标谱项能级数据，用于差动CI加权正样本生成（可选）

    Returns:
        np.ndarray: 包含描述符和标签的训练数据
    """

    # 直接读取合并后的累积CI系数数据
    # 该数据已由 ci_idx_data_processor 函数预先合并并保存
    logger.info("加载累积的CSF索引和CI系数数据（已包含所有历史轮次）")
    accumulated_idxs_ci_path = config.cal_path.accumulated_idxs_ci_path

    if config.cal_settings.cal_loop_num > 1 and not accumulated_idxs_ci_path.exists():
        raise FileNotFoundError(f"累积CI系数文件不存在: {accumulated_idxs_ci_path}")

    accumulated_idxs, accumulated_ci_squared = csfs_idxs_ci_loader(accumulated_idxs_ci_path)

    logger.info(f"训练数据统计")
    logger.info(f"CSF总数（累积）: {len(accumulated_idxs)}")
    logger.info(f"当前轮次: {config.cal_settings.cal_loop_num}")

    # 初始化变量
    cutoff_value = config.cal_settings.cutoff_value

    sampled_csfs_descriptors = raw_csfs_descriptors[accumulated_idxs]
    # 转置以匹配描述符的行维度: (n_current_csfs, n_correct_levels)
    important_csfs_mask = _build_important_csfs_mask(
        accumulated_ci_squared,
        cutoff_value,
        config=config,
        selected_energy_data=selected_energy_data,
    )

    # === 差动 CI 加权正样本（可选）===
    # 物理依据：差动关联（ci²在不同谱项间差异大的CSF）主要影响能级间距。
    # 当计算间距偏离参考值时，优先选择能拉近间距方向的 CSF 作为额外正样本。
    if (
        selected_energy_data is not None
        and config.cal_settings.diff_ci_cutoff > 0
        and len(config.cal_settings.reference_energy_levels) > 1
    ):
        diff_mask = _build_diff_ci_positive_mask(
            accumulated_ci_squared,
            config,
            selected_energy_data,
        )
        diff_count = int(np.sum(diff_mask))
        logger.info(
            f"差动CI加权新增正样本: {diff_count} "
            f"(diff_ci_cutoff={config.cal_settings.diff_ci_cutoff})"
        )

    positive_count = int(np.sum(important_csfs_mask))
    total_elements = important_csfs_mask.size

    logger.info(f"生成完整训练数据: {sampled_csfs_descriptors.shape[0]} 个CSF")
    logger.info(
        f"正样本数量: {positive_count} (占比: {positive_count / total_elements:.4f})"
    )

    # 返回完整的训练数据（类似旧版ann3_proba.py的处理方式）
    caled_csfs_descriptors = np.column_stack(
        [sampled_csfs_descriptors, important_csfs_mask]
    )

    # 保存描述符文件
    descriptor_path = config.cal_path.cal_loop_path / f"{config.cal_path.loop_file_name}_full"
    save_descriptors(caled_csfs_descriptors, descriptor_path, "npy")
    logger.info(f"保存完整历史数据并集描述符文件: {descriptor_path}.npy")

    logger.info("CSFs描述符标签生成完成")
    logger.info(
        f"正样本数量: {positive_count} (在正确能级位置混合系数 ≥ {cutoff_value})"
    )
    logger.info(f"负样本总数量: {total_elements - positive_count}")
    logger.info(f"正样本比例: {positive_count / total_elements:.4f}")

    return caled_csfs_descriptors


def generate_regression_descriptors_from_config(
    config: MLCalConfig,
    raw_csfs_descriptors: NDArray[np.float64],
    logger: logging.Logger,
) -> np.ndarray | None:
    """
    从累积 CI 系数数据生成回归训练描述符（log₁₀(CI²) 连续标签）。

    当 config.cal_settings.use_regression_model = True 时由主流程调用。
    首轮样本不足时返回 None，由调用方回退到分类路径。

    Args:
        config: ML 计算配置对象
        raw_csfs_descriptors: 原始 CSF 描述符，shape: (total_csfs, n_features)
        logger: 日志记录器

    Returns:
        回归训练数据矩阵，shape: (n_accumulated, n_features + n_levels)，
        或 None（样本不足时）
    """
    accumulated_idxs_ci_path = config.cal_path.accumulated_idxs_ci_path

    if config.cal_settings.cal_loop_num > 1 and not accumulated_idxs_ci_path.exists():
        raise FileNotFoundError(
            f"累积CI系数文件不存在: {accumulated_idxs_ci_path}"
        )

    accumulated_idxs, accumulated_ci_squared = csfs_idxs_ci_loader(
        accumulated_idxs_ci_path
    )

    # 首轮保护：样本不足时回退到分类路径
    if len(accumulated_idxs) < 1000:
        logger.warning(
            "回归路径样本不足（%d < 1000），回退到分类路径",
            len(accumulated_idxs),
        )
        return None

    # 诊断日志
    n_levels = accumulated_ci_squared.shape[0]
    for level_idx in range(n_levels):
        level_ci = accumulated_ci_squared[level_idx]
        nonzero_count = int(np.count_nonzero(level_ci))
        if nonzero_count > 0:
            nonzero_vals = level_ci[level_ci > 0]
            log_min = float(np.log10(nonzero_vals.min()))
            log_max = float(np.log10(nonzero_vals.max()))
            # top-5 覆盖率：前 5 个最大 CI² 值占该能级总 CI² 的比例
            sorted_ci = np.sort(level_ci)[::-1]
            total_ci = float(np.sum(level_ci))
            top5_coverage = float(np.sum(sorted_ci[:5]) / total_ci) if total_ci > 0 else 0.0
            logger.info(
                "能级 %d: 非零CSF=%d, log₁₀(CI²) 范围=[%.2f, %.2f], top-5 覆盖率=%.4f",
                level_idx, nonzero_count, log_min, log_max, top5_coverage,
            )
        else:
            logger.warning("能级 %d: 无非零 CI² 值", level_idx)

    # 生成回归训练描述符
    from .ml_regression_trainer import generate_regression_train_descriptors

    min_clip = config.cal_settings.regression_min_clip
    result = generate_regression_train_descriptors(
        raw_csfs_descriptors, accumulated_idxs, accumulated_ci_squared, min_clip
    )

    # 保存描述符文件
    descriptor_path = (
        config.cal_path.cal_loop_path
        / f"{config.cal_path.loop_file_name}_regression_full"
    )
    save_descriptors(result, descriptor_path, "npy")
    logger.info(f"保存回归训练描述符文件: {descriptor_path}.npy")

    return result


def get_stay_descriptors(
    raw_csfs_descriptors: NDArray[np.float64],
    sampled_csfs_idxs_array: NDArray[np.int64]
) -> NDArray[np.float64]:
    """
    找出不在sampled_csfs_idxs_array索引中的描述符

    Args:
        raw_csfs_descriptors: 原始CSFs描述符数组
        sampled_csfs_idxs_array: 已选择的CSFs索引

    Returns:
        np.ndarray: 不在sampled_csfs_idxs_array中的描述符数组
    """
    # 验证字典并安全获取所有已选择的索引
    if sampled_csfs_idxs_array.size == 0:
        raise ValueError("sampled_csfs_idxs_array为空，无法获取选中的CSFs索引")

    mask = np.ones(raw_csfs_descriptors.shape[0], dtype=bool)

    # 2. 把要剔除的行的位置设为 False
    mask[sampled_csfs_idxs_array] = False

    # 3. 数组切片：当你对二维数组使用一维布尔掩码时，NumPy 默认就是筛选“行”
    # 这样会自动保留 mask 为 True 的整行数据
    return raw_csfs_descriptors[mask]
