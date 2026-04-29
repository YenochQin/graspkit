# -*- encoding: utf-8 -*-
import os
import sys
from typing import NotRequired, TextIO, TypedDict


class EnvironmentInfo(TypedDict):
    """Detected execution-environment metadata.

    Attributes:
        is_slurm: Whether SLURM variables were detected.
        is_debug: Whether debug mode is enabled.
        is_interactive: Whether Python is running interactively.
        is_production: Whether the process is a non-debug SLURM run.
        cpu_count: Detected CPU core count.
        slurm_job_id: SLURM job id, if present.
        slurm_procid: SLURM process id, if present.
        slurm_localid: SLURM local id, if present.
        slurm_task_pid: SLURM task pid, if present.
    """

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
    """Keyword options passed to progress-bar helpers.

    Attributes:
        disable: Whether progress bars should be disabled.
        leave: Whether progress bars should remain after completion.
        dynamic_ncols: Whether progress bars should auto-size columns.
        file: Optional stream used for progress output.
        colour: Optional display color supported by tqdm.
    """

    disable: bool
    leave: bool
    dynamic_ncols: bool
    file: NotRequired[TextIO]
    colour: NotRequired[str]


class EnvironmentConfig:
    """Detect runtime context and provide environment-aware defaults.

    The class centralizes SLURM, debug, interactivity, logging, and progress
    configuration so calculation scripts can use consistent behavior across
    local and batch environments.
    """

    def __init__(self) -> None:
        """Initialize cached environment flags.

        The detection work is performed once so repeated property access does
        not repeatedly inspect environment variables or interpreter flags.
        """
        self._is_slurm = self._detect_slurm_environment()
        self._is_interactive = self._detect_interactive()
        self._is_debug = self._detect_debug_mode()
        self._cpu_count = os.cpu_count() or 4

    def _detect_slurm_environment(self) -> bool:
        """Detect whether the current process is running under SLURM.

        Returns:
            True if common SLURM environment variables are present.
        """
        slurm_indicators: list[str] = [
            "SLURM_JOB_ID",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_TASK_PID",
        ]
        return any(env_var in os.environ for env_var in slurm_indicators)

    def _detect_debug_mode(self) -> bool:
        """Detect whether debug mode was requested.

        Returns:
            True when debug-related environment variables or CLI flags are set.
        """
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
        """Detect whether Python is running interactively.

        Returns:
            True in a REPL-like or interactive Python session.
        """
        return hasattr(sys, "ps1") or bool(sys.flags.interactive)

    @property
    def is_interactive(self) -> bool:
        """Whether Python is running interactively.

        Returns:
            True in a REPL-like or interactive Python session.
        """
        return self._is_interactive

    @property
    def is_slurm_environment(self) -> bool:
        """Whether SLURM environment variables were detected.

        Returns:
            True when common SLURM variables are present.
        """
        return self._is_slurm

    @property
    def is_debug_mode(self) -> bool:
        """Whether debug mode is enabled.

        Returns:
            True when debug environment variables or CLI flags are set.
        """
        return self._is_debug

    @property
    def is_production_mode(self) -> bool:
        """Whether the process is in non-debug SLURM production mode.

        Returns:
            True for SLURM jobs that are not running in debug mode.
        """
        return self._is_slurm and not self._is_debug

    @property
    def cpu_count(self) -> int:
        """Detected CPU core count.

        Returns:
            Number of available CPU cores, with a conservative fallback.
        """
        return self._cpu_count

    def get_environment_info(self) -> EnvironmentInfo:
        """Return a dictionary summary of detected environment flags.

        Returns:
            EnvironmentInfo dictionary including SLURM identifiers when
            present.
        """
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
        """Return progress-bar defaults for the current environment.

        Returns:
            ProgressConfig dictionary suitable for tqdm-style progress bars.
        """
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
        """Return logging.basicConfig-compatible logging options.

        Returns:
            Dictionary containing logging level and format strings.
        """
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
        """Return non-standard display flags used by application logging.

        Returns:
            Dictionary of display flags consumed by higher-level logging code.
        """
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
    """Return the process-wide EnvironmentConfig singleton.

    Returns:
        Cached EnvironmentConfig instance, creating it on first use.
    """
    global _env_config
    if _env_config is None:
        _env_config = EnvironmentConfig()
    return _env_config


def is_slurm_environment() -> bool:
    """Check whether the process is running under SLURM.

    Returns:
        True when SLURM variables are present.
    """
    return get_environment_config().is_slurm_environment


def is_debug_mode() -> bool:
    """Check whether debug mode is enabled.

    Returns:
        True when debug mode is enabled.
    """
    return get_environment_config().is_debug_mode


def is_production_mode() -> bool:
    """Check whether the process is in non-debug SLURM mode.

    Returns:
        True for non-debug SLURM execution.
    """
    return get_environment_config().is_production_mode
