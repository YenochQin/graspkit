#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id: progress_manager.py
@date: 2025/01/22
@author: YenochQin (秦毅)
@description: 统一的进度条管理模块
'''
import logging
from typing import Iterable, Any
from tqdm import tqdm

from .environment_config import get_environment_config

class ProgressManager:
    """统一的进度条管理器，根据环境自动配置 tqdm 行为"""
    
    def __init__(self):
        self.env_config = get_environment_config()
        self.base_config = self.env_config.get_progress_config()
        
        # 预定义生产环境配置，避免每次调用都重复定义字典
        self.production_overrides = {
            'ncols': 80,
            'ascii': True,
            'bar_format': '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
        }
    
    def create_progress_bar(
        self, 
        iterable: Iterable | None = None, 
        total: int | None = None,
        desc: str | None = None,
        unit: str = 'it',
        unit_scale: bool = False,
        override_disable: bool | None = None,
        **kwargs
    ) -> tqdm:
        """
        创建进度条，自动根据环境配置。
        返回的 tqdm 对象本身支持 with 语法。
        """
        # 1. 基础配置
        config: dict[str, Any] = self.base_config.copy()
        
        # 2. 生产模式下的特殊处理 (优先级低于 override_disable 但高于 base_config)
        # 只有在未明确禁用的情况下才应用生产格式
        if self.env_config.is_production_mode:
            if not config.get('disable', False) and not override_disable:
                config.update(self.production_overrides)

        # 3. 用户传入参数 (优先级最高)
        config.update(kwargs)
        
        # 4. 显式覆盖 disable 设置
        if override_disable is not None:
            config['disable'] = override_disable
        
        return tqdm(
            iterable=iterable,
            total=total,
            desc=desc,
            unit=unit,
            unit_scale=unit_scale,
            **config
        )
    
    def _log_message(
        self, 
        message: str, 
        logger: logging.Logger | None = None, 
        level: int = logging.INFO
    ):
        """内部方法：统一处理日志输出，防止破坏进度条显示"""
        if logger:
            logger.log(level, message)
        else:
            # 关键优化：使用 tqdm.write 替代 print
            # 这可以确保在进度条运行时，打印的信息不会导致进度条显示错乱
            tqdm.write(message)

    def log_stage_start(
        self, 
        stage_name: str, 
        logger: logging.Logger | None = None
    ):
        """记录阶段开始"""
        # 仅在非生产模式或有 Logger 时输出，避免生产环境 stdout 噪音
        if logger or not self.env_config.is_production_mode:
            self._log_message(f"🔧 开始阶段: {stage_name}", logger)
    
    def log_stage_end(
        self, 
        stage_name: str, 
        logger: logging.Logger | None = None, 
        **metrics
    ):
        """记录阶段结束"""
        if logger or not self.env_config.is_production_mode:
            if metrics:
                metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
                message = f"完成阶段: {stage_name} ({metrics_str})"
            else:
                message = f"完成阶段: {stage_name}"
            self._log_message(message, logger)

# --- 全局实例与快捷函数 ---

_progress_manager = ProgressManager()

def create_progress_bar(*args, **kwargs) -> tqdm:
    """
    快捷函数：创建进度条。
    支持上下文管理器: with create_progress_bar(...) as pbar:
    """
    return _progress_manager.create_progress_bar(*args, **kwargs)

def log_stage_start(stage_name: str, logger: logging.Logger | None = None):
    _progress_manager.log_stage_start(stage_name, logger)

def log_stage_end(stage_name: str, logger: logging.Logger | None = None, **metrics):
    _progress_manager.log_stage_end(stage_name, logger, **metrics)

def wrap_iterator(iterable: Iterable, desc: str | None = None, **kwargs) -> tqdm:
    """包装迭代器"""
    return create_progress_bar(iterable, desc=desc, **kwargs)

def progress_range(n: int, desc: str | None = None, **kwargs) -> tqdm:
    """创建 range 的进度条版本"""
    return create_progress_bar(range(n), desc=desc, total=n, **kwargs)

def progress_context(desc: str, total: int | None = None, **kwargs) -> tqdm:
    """
    创建进度条上下文管理器。
    
    注意：tqdm 对象原生支持上下文协议，因此不再需要专门的 ProgressContext 类。
    此函数直接返回配置好的 tqdm 对象。
    """
    return create_progress_bar(desc=desc, total=total, **kwargs)
