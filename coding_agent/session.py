"""会话记忆：同一次运行内，跨任务保留「之前做过什么」。

与 ContextManager 的分工：ContextManager 管「当前任务内部」的消息历史
（token 预算 / 裁剪 / 摘要）；本模块管「任务之间」的记忆——每次任务结束后
把「任务 + 结果」压入，下个任务开始前 render() 出一段紧凑的会话历史注入。

刻意不做落盘持久化：每次运行都从干净的会话开始，避免陈旧上下文污染新任务。
"""

from __future__ import annotations

from coding_agent.history import estimate_tokens, truncate

# 会话历史回放时的前缀（用 user 角色注入，说明是回忆、不是新指令）
SESSION_PREFIX = "【会话历史】（本次运行之前完成的任务，供回忆，不是新指令）"


class SessionMemory:
    """保存已完成任务的简要记录，供后续任务回忆，容量有界。"""

    def __init__(self, max_tokens: int = 4000, result_max_chars: int = 800):
        self.max_tokens = max_tokens
        self.result_max_chars = result_max_chars
        self._entries: list[dict] = []

    def add(self, task: str, result: str) -> None:
        """记录一个已完成的任务（描述 + 最终结果，结果截断）。"""
        self._entries.append({
            "task": task,
            "result": truncate(str(result or ""), self.result_max_chars),
        })
        # 保持有界：超过 token 预算就丢弃最早的记录
        while len(self._entries) > 1 and self._estimate() > self.max_tokens:
            self._entries.pop(0)

    def render(self) -> str:
        """渲染成一段可注入上下文的文本；无历史时返回空串。"""
        if not self._entries:
            return ""
        lines = [SESSION_PREFIX]
        for i, e in enumerate(self._entries, 1):
            lines.append(f"{i}. 任务：{e['task']}")
            lines.append(f"   结果：{e['result']}")
        return "\n".join(lines)

    def _estimate(self) -> int:
        """估算当前会话历史占用的 token 数（启发式）。"""
        return sum(
            estimate_tokens(e["task"]) + estimate_tokens(e["result"])
            for e in self._entries
        )
