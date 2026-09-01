"""工具基础设施：工具结果类型、工具定义与注册表。

一个「工具」由两部分组成：
  1. schema   —— 提供给模型 function calling 的 JSON Schema（描述用途与参数）；
  2. executor —— 本地执行的 Python 函数（接收参数 dict，返回 ToolResult）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolResult:
    """工具执行结果。output 会作为 tool 消息回传给模型。"""

    output: str
    is_error: bool = False


@dataclass
class Tool:
    name: str
    schema: dict
    executor: Callable[[dict], ToolResult]


_REGISTRY: dict[str, Tool] = {}


def make_tool(
    name: str,
    description: str,
    properties: dict,
    required: list[str],
    executor: Callable[[dict], ToolResult],
) -> Tool:
    """便捷构造：拼出 OpenAI function calling 的完整 schema。"""
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
    return Tool(name=name, schema=schema, executor=executor)


def register(tool: Tool) -> None:
    _REGISTRY[tool.name] = tool


def get_tool_schemas() -> list[dict]:
    """返回全部工具的 schema（用于发给模型）。"""
    return [t.schema for t in _REGISTRY.values()]


def execute_tool(name: str, args: dict) -> ToolResult:
    """按名字执行工具。任何异常都会被捕获并作为错误结果回传，不使循环崩溃。"""
    if name not in _REGISTRY:
        return ToolResult(output=f"未知工具：{name}", is_error=True)
    try:
        return _REGISTRY[name].executor(args)
    except Exception as e:
        return ToolResult(output=f"工具执行异常：{e}", is_error=True)
