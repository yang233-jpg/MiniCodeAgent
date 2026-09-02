"""Git 工具：git_status / git_diff，均为只读，用于让 agent 在改动后检查「改了什么」。

刻意只做这两个最常用的检查工具，不把 Git 做成完整的版本控制系统。
"""

from __future__ import annotations

import subprocess

from .base import ToolResult, make_tool, register


def _run_git(argv: list[str]) -> ToolResult:
    """执行一条只读 git 命令，统一捕获错误与输出。"""
    try:
        proc = subprocess.run(["git", *argv], capture_output=True, timeout=30)
    except FileNotFoundError:
        return ToolResult(output="未找到 git 命令，请确认已安装并加入 PATH", is_error=True)
    except subprocess.TimeoutExpired:
        return ToolResult(output="git 命令超时（>30s）", is_error=True)
    except Exception as e:
        return ToolResult(output=f"git 执行异常：{e}", is_error=True)

    out = proc.stdout.decode("utf-8", errors="replace").strip()
    err = proc.stderr.decode("utf-8", errors="replace").strip()

    # git 失败（如不在仓库内）时把 stderr 作为错误回传，交给模型判断
    if proc.returncode != 0:
        return ToolResult(output=err or out or f"git 退出码 {proc.returncode}", is_error=True)

    return ToolResult(output=out if out else "（无改动）")


def _git_status(args: dict) -> ToolResult:
    return _run_git(["status", "--short"])


def _git_diff(args: dict) -> ToolResult:
    path = str(args.get("path", "") or "").strip()
    argv = ["diff"]
    if path:
        argv += ["--", path]
    return _run_git(argv)


register(make_tool(
    name="git_status",
    description="查看当前 Git 工作区状态（--short 格式：新增/修改/删除的文件）。",
    properties={},
    required=[],
    executor=_git_status,
))

register(make_tool(
    name="git_diff",
    description="查看当前工作区相对于最近一次提交的代码改动（git diff）。可指定 path 只看某路径。",
    properties={
        "path": {"type": "string", "description": "可选，只看某个文件/目录的改动"},
    },
    required=[],
    executor=_git_diff,
))
