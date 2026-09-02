"""工具包：定义 agent 可用的全部本地工具并统一注册。

导入本包时，各子模块会把各自的工具注册进 base 的注册表；
外部通过 get_tool_schemas() / execute_tool() 使用，无需关心实现细节。
"""

from . import files, git, search, shell  # noqa: F401  触发注册
from .base import ToolResult, execute_tool, get_tool_schemas

__all__ = ["ToolResult", "get_tool_schemas", "execute_tool"]
