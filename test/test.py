# %%
import logging
from dataclasses import dataclass
from typing import Optional

log_config = {
    "level": "DEBUG",
    "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    "show_progress_logs": True,
    "highlight_stages": True,
}
logging.basicConfig(
    level=log_config["level"],
    format=log_config["format"],
    datefmt="%m-%d %H:%M:%S",
    force=True,  # 强制重新配置
)
logger = logging.getLogger(__name__)


# %%
@dataclass
class MLDataCounts:
    total_csfs_count: int
    cal_csfs_count: int

    important_csfs_count: Optional[int] = None
    important_retention_rate: Optional[float] = None
    ml_sampled_count: Optional[int] = None
    ml_retention_rate: Optional[float] = None
    ml_new_count: Optional[int] = None
    ml_predicted_count: Optional[int] = None
    final_sampled_count: Optional[int] = None
    final_retention_rate: Optional[float] = None


# %%
train_data_counts = MLDataCounts(
    total_csfs_count=5000,
    cal_csfs_count=2000,
    important_csfs_count=500,
    ml_sampled_count=1500,
    final_sampled_count=1300,
)
# %%
stats_config = [
    ("important_csfs_count", "重要 CSFs ", "important_retention_rate"),
    ("ml_sampled_count", "ML新增 CSFs ", "ml_retention_rate"),
    ("final_sampled_count", "最终选择 CSFs ", "final_retention_rate"),
]
total = train_data_counts.total_csfs_count

for field, label, rate_key in stats_config:
    count = getattr(train_data_counts, field, None)
    if count is not None:
        rate = count / total
        setattr(train_data_counts, rate_key, rate)
        logger.info(f"- {label}数量: {count} (占原始: {rate:.4%})")
train_data_counts
