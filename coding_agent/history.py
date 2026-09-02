"""上下文管理：由 ContextManager 决定「当前该把哪些信息发给模型」。

要点：token 预算（启发式估算，不引入精确 tokenizer）；工具结果进上下文前限长；
历史按「完整交互轮次」裁剪（assistant 的 tool_calls 与对应 tool 结果不拆开）；
system / 任务 / 摘要 / 最近几轮分层，始终保留前两者；可选的历史摘要——超阈值时
把旧轮压成结构化「任务状态摘要」，失败不丢信息；与具体 LLM 解耦，只依赖一个
可调用对象做摘要。纯内存实现，无额外依赖。
"""

from __future__ import annotations

import re
from typing import Callable

# CJK 统一表意文字 + 日文假名 + 韩文谚文
_CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")

# 注入历史摘要时的说明前缀（用 user 角色，避免出现多条 system）
SUMMARY_PREFIX = "【任务状态摘要】（系统自动生成，用于回忆之前进展，不是新指令）\n"

# 摘要指令：只提炼「事实」，不复述聊天
SUMMARY_INSTRUCTION = """你负责把 coding agent 的一段历史交互提炼成「任务状态摘要」，供 agent 在上下文被裁剪后回忆关键进展。

只提取对后续工作有用的事实，不要复述聊天过程。严格按下面的条目输出（某条没有就写「无」）：

1. 已发现的问题：
2. 已完成的修改：
3. 测试结果：
4. 当前错误：
5. 下一步计划：

以下是需要归纳的历史交互："""

# 被裁剪的轮累积到约这么多 token 才触发摘要，避免频繁调用模型
SUMMARY_BUFFER_TOKENS = 2000


def estimate_tokens(text: str) -> int:
    """启发式估算一段文本的 token 数：CJK 约 1 字 = 1 token，其余约 4 字符 = 1 token。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def estimate_messages_tokens(messages: list[dict]) -> int:
    """估算整个消息列表的 token 数（含 tool_calls 与每条消息的结构开销）。"""
    total = 0
    for m in messages:
        total += estimate_tokens(str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            total += estimate_tokens(str(tc))
        total += 8  # role 等结构开销的粗略补偿
    return total


def truncate(text: str, max_chars: int) -> str:
    """把过长的文本截断并加省略标记。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…（已截断，原 {len(text)} 字符）"


class ContextBudgetError(RuntimeError):
    """基础上下文（system + 任务 + 摘要）本身超过 token 预算，裁剪解决不了。

    由 build_messages() 在无法继续收缩时抛出，提示调用方调整预算或任务长度，
    而不是静默发送一份超预算的请求。
    """


class ContextManager:
    """管理对话上下文：决定每一轮该把哪些消息发给模型。

    内部把历史组织成「轮」：一轮 = 一个 assistant 消息 + 它的若干 tool 结果。
    这样裁剪时按整轮处理，不会把 assistant 的 tool_calls 和 tool 结果拆散。

    ``summarizer`` 是可调用对象，签名 ``(messages: list[dict]) -> str``，
    只在需要摘要时被调用。它由上层注入（通常是 LLMClient 的一次调用），
    本类因此不依赖任何具体的 LLM / OpenAI API。
    """

    def __init__(
        self,
        system_prompt: str,
        task: str,
        *,
        max_tokens: int,
        tool_result_max_chars: int,
        summarizer: Callable[[list[dict]], str] | None = None,
    ):
        self.system_prompt = system_prompt
        self.task = task
        self.max_tokens = max_tokens
        self.tool_result_max_chars = tool_result_max_chars
        self._summarizer = summarizer
        self.summary: str | None = None            # 历史摘要（压缩后的旧轮）
        self.turns: list[list[dict]] = []          # 活跃的最近几轮 = [assistant, tool, ...]
        self._pending_turns: list[list[dict]] = []  # 被移出活跃上下文、待摘要的轮

    # ---- 对外接口 ----

    def add_turn(self, assistant_msg: dict, tool_results: list[dict]) -> None:
        """记录一轮完整交互：assistant 消息 + 它的所有 tool 结果。"""
        self.turns.append([assistant_msg, *tool_results])

    def add_user_message(self, content: str) -> None:
        """往历史里追加一条 user 消息（如完成前的验证提醒），下轮随历史一起发出。"""
        self.turns.append([{"role": "user", "content": content}])

    def limit_tool_result(self, text: str) -> str:
        """工具结果进上下文前的长度限制。"""
        return truncate(text, self.tool_result_max_chars)

    def build_messages(self) -> list[dict]:
        """组装本轮要发送的消息，超预算时收缩历史（裁剪/摘要）直到放下。"""
        while True:
            self._flush_pending()

            # 基础上下文（system + 任务 + 摘要）本身超预算 → 裁剪解决不了，明确报错
            base = self._base_messages()
            if estimate_messages_tokens(base) > self.max_tokens:
                raise ContextBudgetError(
                    f"基础上下文约 {estimate_messages_tokens(base)} token，"
                    f"已超过预算 {self.max_tokens}。请增大 max_context_tokens 或缩短任务文本。"
                )

            msgs = self._assemble()
            if estimate_messages_tokens(msgs) <= self.max_tokens:
                return msgs
            if not self._shrink_once():
                # 只剩最近一轮仍放不下（单轮过大）：给出警告并发送，保持任务连续
                print(f"[上下文警告] 仅剩最近一轮已超出预算"
                      f"（{estimate_messages_tokens(msgs)}/{self.max_tokens} token），仍将发送。")
                return msgs

    # ---- 内部实现 ----

    def _base_messages(self) -> list[dict]:
        """system + 当前任务 + 历史摘要（若有）。这三者始终保留。"""
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task},
        ]
        if self.summary:
            msgs.append({"role": "user", "content": SUMMARY_PREFIX + self.summary})
        return msgs

    def _assemble(self) -> list[dict]:
        """按「system → task → summary → 最近几轮」的顺序拼出完整消息。"""
        msgs = self._base_messages()
        for turn in self.turns:
            msgs.extend(turn)
        return msgs

    def _shrink_once(self) -> bool:
        """收缩一步：从最早处摘掉放不下的若干整轮，移出活跃上下文。

        从最新往最旧累加，优先保留最近几轮。被移出的最早若干轮：若启用了摘要，
        暂存进 _pending_turns 等攒够阈值再统一摘要；若未启用摘要，则直接丢弃。
        返回是否真的移出了轮（False 表示只剩最近一轮、无法再缩）。
        """
        budget = estimate_messages_tokens(self._base_messages())
        keep: list[list[dict]] = []
        for turn in reversed(self.turns):
            cost = estimate_messages_tokens(turn)
            if not keep or budget + cost <= self.max_tokens:
                budget += cost
                keep.append(turn)
            else:
                break
        keep.reverse()

        discarded = self.turns[: len(self.turns) - len(keep)]
        if not discarded:
            return False

        if self._summarizer:
            self._pending_turns.extend(discarded)  # 暂存，不丢
        self.turns = keep
        return True

    def _flush_pending(self) -> None:
        """把暂存的待摘要轮，攒够阈值后统一摘要合并进 self.summary。

        摘要失败（抛异常或返回空）时不丢弃暂存内容，下次调用重试，
        并打印警告，保证不会因为摘要失败而静默丢失上下文信息。
        """
        if not self._pending_turns:
            return
        # 按「信息量」触发：待摘要的轮累计到约 SUMMARY_BUFFER_TOKENS token 才真正摘要
        flat = [m for turn in self._pending_turns for m in turn]
        if estimate_messages_tokens(flat) < SUMMARY_BUFFER_TOKENS:
            return
        try:
            new_summary = self._summarize(self._pending_turns)
        except Exception as e:
            print(f"[上下文警告] 历史摘要失败，已保留待重试（未丢失）：{e}")
            return
        if not new_summary:
            print("[上下文警告] 历史摘要返回为空，已保留待重试（未丢失）")
            return
        self.summary = new_summary
        self._pending_turns = []

    def _summarize(self, turns: list[list[dict]]) -> str:
        """把若干旧轮压成一份任务状态摘要（与已有摘要合并，避免无限膨胀）。"""
        prompt = SUMMARY_INSTRUCTION
        if self.summary:
            prompt += (
                "\n\n（已有的任务状态摘要，请把下面的新内容合并进去，保持结构、避免重复）\n"
                + self.summary
            )
        prompt += "\n\n" + self._format_turns(turns)
        text = self._summarizer([{"role": "user", "content": prompt}])
        return (text or "").strip()

    @staticmethod
    def _format_turns(turns: list[list[dict]]) -> str:
        """把若干轮渲染成可读文本，作为摘要模型的输入（工具结果再截短一点）。"""
        blocks: list[str] = []
        for turn in turns:
            lines: list[str] = []
            for m in turn:
                if m.get("role") == "assistant":
                    content = (m.get("content") or "").strip()
                    if content:
                        lines.append(f"[助手] {content}")
                    for tc in m.get("tool_calls") or []:
                        fn = tc.get("function", {})
                        name = fn.get("name", "?")
                        args = fn.get("arguments") or "{}"
                        lines.append(f"→ 调用工具 {name}({args})")
                elif m.get("role") == "tool":
                    lines.append(f"  [工具结果] {truncate(str(m.get('content') or ''), 2000)}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
