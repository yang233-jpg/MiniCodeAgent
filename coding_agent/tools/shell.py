"""命令执行工具：run_command。"""

from __future__ import annotations

import subprocess

from .base import ToolResult, make_tool, register


def _decode(data: bytes) -> str:
    """按 utf-8 → gbk → latin-1 顺序尝试解码命令输出，尽量还原中文。"""
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _run_command(args: dict) -> ToolResult:
    command = args.get("command", "")
    timeout = int(args.get("timeout", 60))

    try:
        proc = subprocess.run(command, shell=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ToolResult(output=f"命令超时（>{timeout}s），已终止", is_error=True)
    except Exception as e:
        return ToolResult(output=f"命令执行失败：{e}", is_error=True)

    out = _decode(proc.stdout).strip()
    err = _decode(proc.stderr).strip()

    parts = [f"退出码 {proc.returncode}"]
    if out:
        parts.append(out)
    if err:
        parts.append(f"[stderr]\n{err}")
    return ToolResult(output="\n".join(parts))


register(make_tool(
    name="run_command",
    description="在本地 shell 中执行命令并返回输出与退出码。用于运行程序、跑测试、安装依赖等。",
    properties={
        "command": {"type": "string", "description": "要执行的命令"},
        "timeout": {"type": "integer", "description": "超时秒数，默认 60"},
    },
    required=["command"],
    executor=_run_command,
))
