# -*- encoding: utf-8 -*-
"""
@Id: environment_config.py
@date: 2025/01/22
@author: YenochQin (秦毅)
@description: 统一的环境检测和配置模块
"""

import os
import sys
from typing import NotRequired, TextIO, TypedDict


class EnvironmentInfo(TypedDict):
    is_slurm: bool
    is_debug: bool
    is_interactive: bool
    is_production: bool
    cpu_count: int
    slurm_job_id: str | None
    slurm_procid: str | None
    slurm_localid: str | None
    slurm_task_pid: str | None


class ProgressConfig(TypedDict):
    disable: bool
    leave: bool
    dynamic_ncols: bool
    file: NotRequired[TextIO]
    colour: NotRequired[str]


class EnvironmentConfig:
    """环境配置管理器，用于检测运行环境和设置相应的配置"""

    def __init__(self) -> None:
        self._is_slurm = self._detect_slurm_environment()
        self._is_interactive = self._detect_interactive()
        self._is_debug = self._detect_debug_mode()
        self._cpu_count = os.cpu_count() or 4

    def _detect_slurm_environment(self) -> bool:
        """检测是否在SLURM环境中运行"""
        slurm_indicators: list[str] = [
            "SLURM_JOB_ID",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_TASK_PID",
        ]
        return any(env_var in os.environ for env_var in slurm_indicators)

    def _detect_debug_mode(self) -> bool:
        """检测是否处于调试模式"""
        # 检查环境变量
        if os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"):
            return True
        if os.environ.get("PYTHON_DEBUG", "").lower() in ("1", "true", "yes"):
            return True

        # 检查命令行参数
        if "--debug" in sys.argv:
            return True

        return False

    def _detect_interactive(self) -> bool:
        """检测是否在交互式环境（REPL/Jupyter）中运行"""
        return hasattr(sys, "ps1") or bool(sys.flags.interactive)

    @property
    def is_interactive(self) -> bool:
        """是否在交互式环境（REPL/Jupyter）中运行"""
        return self._is_interactive

    @property
    def is_slurm_environment(self) -> bool:
        """是否在SLURM环境中运行"""
        return self._is_slurm

    @property
    def is_debug_mode(self) -> bool:
        """是否处于调试模式"""
        return self._is_debug

    @property
    def is_production_mode(self) -> bool:
        """是否处于生产模式（SLURM环境且非调试模式）"""
        return self._is_slurm and not self._is_debug

    @property
    def cpu_count(self) -> int:
        """系统CPU核心数"""
        return self._cpu_count

    def get_environment_info(self) -> EnvironmentInfo:
        """获取环境信息摘要"""
        return {
            "is_slurm": self.is_slurm_environment,
            "is_debug": self.is_debug_mode,
            "is_interactive": self.is_interactive,
            "is_production": self.is_production_mode,
            "cpu_count": self.cpu_count,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_procid": os.environ.get("SLURM_PROCID"),
            "slurm_localid": os.environ.get("SLURM_LOCALID"),
            "slurm_task_pid": os.environ.get("SLURM_TASK_PID"),
        }

    def get_progress_config(self) -> ProgressConfig:
        """获取进度条配置"""
        if self.is_production_mode:
            # 生产模式：关闭进度条
            return {
                "disable": True,
                "leave": False,
                "dynamic_ncols": False,
            }
        else:
            # 调试模式：启用进度条
            return {
                "disable": False,
                "leave": True,
                "dynamic_ncols": True,
                "file": sys.stderr,
                "colour": "green",
            }

    def get_logging_config(self) -> dict[str, str]:
        """获取标准日志配置（兼容 logging.basicConfig）"""
        if self.is_production_mode:
            return {
                "level": "INFO",
                "format": "%(asctime)s [%(levelname)s] %(module)s:%(lineno)d - %(message)s",
            }
        else:
            return {
                "level": "DEBUG",
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
            }

    def get_display_config(self) -> dict[str, bool]:
        """获取应用层日志显示选项（非标准 logging 键）"""
        if self.is_production_mode:
            return {
                "show_progress_logs": False,
                "highlight_stages": True,
            }
        else:
            return {
                "show_progress_logs": True,
                "highlight_stages": True,
            }


# 全局环境配置实例
_env_config: EnvironmentConfig | None = None

def get_environment_config() -> EnvironmentConfig:
    """获取全局环境配置实例"""
    global _env_config
    if _env_config is None:
        _env_config = EnvironmentConfig()
    return _env_config


def is_slurm_environment() -> bool:
    """快捷函数：检查是否在SLURM环境"""
    return get_environment_config().is_slurm_environment


def is_debug_mode() -> bool:
    """快捷函数：检查是否在调试模式"""
    return get_environment_config().is_debug_mode


def is_production_mode() -> bool:
    """快捷函数：检查是否在生产模式"""
    return get_environment_config().is_production_mode
