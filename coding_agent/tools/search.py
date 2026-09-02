"""搜索类工具：glob 文件匹配 / grep 内容检索。"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from .base import ToolResult, make_tool, register

# 搜索时跳过的目录（避免把 .git/.venv 等噪音塞给模型）
_SKIP_DIRS = {".git", ".venv", "__pycache__", ".claude", ".idea", "node_modules", ".pytest_cache"}


def _skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


def _glob(args: dict) -> ToolResult:
    pattern = args.get("pattern", "*")
    matches = [str(p) for p in Path(".").glob(pattern) if not _skip(p)]
    return ToolResult(output="\n".join(sorted(matches)) if matches else "（无匹配）")


def _grep(args: dict) -> ToolResult:
    pattern = args.get("pattern", "")
    root = Path(args.get("path", "."))
    file_glob = args.get("glob")  # 可选，只搜文件名匹配该 glob 的文件

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return ToolResult(output=f"正则表达式无效：{e}", is_error=True)

    if not root.is_dir():
        return ToolResult(output=f"目录不存在：{root}", is_error=True)

    hits: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or _skip(p):
            continue
        if p.stat().st_size > 1_000_000:  # 跳过 >1MB 的大文件
            continue
        if file_glob and not fnmatch.fnmatch(p.name, file_glob):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                hits.append(f"{p}:{i}:{line.rstrip()}")
                if len(hits) >= 200:
                    break
        if len(hits) >= 200:
            break

    return ToolResult(output="\n".join(hits) if hits else "（无匹配）")


register(make_tool(
    name="glob",
    description="按 glob 模式查找文件，返回匹配的相对路径列表。",
    properties={
        "pattern": {"type": "string", "description": "glob 模式，如 **/*.py"},
    },
    required=["pattern"],
    executor=_glob,
))

register(make_tool(
    name="grep",
    description="在文件内容中搜索正则表达式，返回 文件:行号:内容 的匹配列表。",
    properties={
        "pattern": {"type": "string", "description": "正则表达式"},
        "path": {"type": "string", "description": "搜索根目录，默认当前目录"},
        "glob": {"type": "string", "description": "可选，只搜索文件名匹配该 glob 的文件"},
    },
    required=["pattern"],
    executor=_grep,
))
