"""
配置模块

集中管理所有可调参数，确保实验的单一变量原则。
所有模块从 config 读取参数，不自行硬编码。
"""

from .settings import Config, get_config

__all__ = ["Config", "get_config"]
