"""文件类工具：read_file / write_file / edit_file / list_dir。"""

from __future__ import annotations

from pathlib import Path

from .base import ToolResult, make_tool, register


def _read_file(args: dict) -> ToolResult:
    path = Path(args.get("path", ""))
    offset = int(args.get("offset", 0))
    limit = int(args.get("limit", 2000))

    if not path.is_file():
        return ToolResult(output=f"文件不存在：{path}", is_error=True)

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return ToolResult(output=f"读取失败：{e}", is_error=True)

    total = len(lines)
    offset = max(0, offset)
    selected = lines[offset : offset + limit]
    numbered = [f"{offset + i + 1}\t{line}" for i, line in enumerate(selected)]
    header = f"共 {total} 行，显示第 {offset + 1}~{offset + len(selected)} 行"
    return ToolResult(output=header + "\n" + "\n".join(numbered))


def _write_file(args: dict) -> ToolResult:
    path = Path(args["path"])
    content = args.get("content", "")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        return ToolResult(output=f"写入失败：{e}", is_error=True)
    return ToolResult(output=f"已写入 {path}（{len(content)} 字符）")


def _edit_file(args: dict) -> ToolResult:
    path = Path(args["path"])
    old = args.get("old_string", "")
    new = args.get("new_string", "")

    if not path.is_file():
        return ToolResult(output=f"文件不存在：{path}", is_error=True)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return ToolResult(output=f"读取失败：{e}", is_error=True)

    count = text.count(old)
    if count == 0:
        return ToolResult(output="未找到 old_string（该片段在文件中不存在）", is_error=True)
    if count > 1:
        return ToolResult(
            output=f"old_string 出现 {count} 次，不唯一；请带上更多上下文使其唯一",
            is_error=True,
        )

    path.write_text(text.replace(old, new), encoding="utf-8")
    return ToolResult(output="已替换 1 处")


def _list_dir(args: dict) -> ToolResult:
    path = Path(args.get("path", "."))
    if not path.is_dir():
        return ToolResult(output=f"目录不存在：{path}", is_error=True)
    names = [p.name + ("/" if p.is_dir() else "") for p in sorted(path.iterdir(), key=lambda x: x.name)]
    return ToolResult(output="\n".join(names) if names else "（空目录）")


register(make_tool(
    name="read_file",
    description="读取文本文件内容，返回带行号的内容。可用 offset/limit 分页读取大文件。",
    properties={
        "path": {"type": "string", "description": "文件路径（相对或绝对）"},
        "offset": {"type": "integer", "description": "起始行（从 0 起），默认 0"},
        "limit": {"type": "integer", "description": "读取行数，默认 2000"},
    },
    required=["path"],
    executor=_read_file,
))

register(make_tool(
    name="write_file",
    description="创建或覆盖写入一个文件（自动创建父目录）。",
    properties={
        "path": {"type": "string", "description": "文件路径"},
        "content": {"type": "string", "description": "要写入的完整内容"},
    },
    required=["path", "content"],
    executor=_write_file,
))

register(make_tool(
    name="edit_file",
    description="精确替换文件中的一段文本：把唯一的 old_string 替换为 new_string。old_string 必须在文件中唯一出现。",
    properties={
        "path": {"type": "string", "description": "文件路径"},
        "old_string": {"type": "string", "description": "要被替换的原文片段（须唯一）"},
        "new_string": {"type": "string", "description": "替换后的新文本"},
    },
    required=["path", "old_string", "new_string"],
    executor=_edit_file,
))

register(make_tool(
    name="list_dir",
    description="列出目录下的文件和子目录（子目录带 / 后缀）。",
    properties={
        "path": {"type": "string", "description": "目录路径，默认当前目录"},
    },
    required=[],
    executor=_list_dir,
))
