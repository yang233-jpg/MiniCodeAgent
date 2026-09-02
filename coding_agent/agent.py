"""Agent 主循环：编排 LLM 与工具，自主完成编程任务。

流程：ContextManager 决定本轮发什么 → 调模型 → 解析输出（文本或 tool_calls）
→ 本地执行工具 → 结果连同 assistant 消息作为「一轮」交还 → 循环，直到模型
停止调用工具或达轮数上限。上下文管理（裁剪/摘要/工具结果限长）统一在
ContextManager，本文件只负责把「模型调用」和「工具执行」串起来。
"""

from __future__ import annotations

import json
from typing import Callable

from coding_agent.config import Config
from coding_agent.history import ContextBudgetError, ContextManager
from coding_agent.llm import LLMClient
from coding_agent.session import SessionMemory
from coding_agent.tools import execute_tool, get_tool_schemas

SYSTEM_PROMPT = """你是一个编程智能体（coding agent），能在本地读写文件、执行命令来自主完成编程任务。

## 工作方式
1. 先理解任务，用工具查看项目结构（list_dir / glob）和相关代码（read_file / grep）；
2. 规划改动再动手：用 write_file 创建文件、edit_file 精确修改；
3. 修改后用 run_command 运行程序或测试来验证结果；
4. 最后用纯文本给出简洁总结（说明做了什么、结果如何），不要再调用工具。

## 工具使用规则
- 动手前先看清现有代码，不要凭空猜测；
- 优先最小改动，不破坏已有功能；
- 工具返回错误时，分析原因并调整，不要用同样方式反复重试；
- **修改代码后必须用 run_command 运行程序或测试来验证**；验证失败就分析报错、修改、再测试，直到通过；
- 一次只调用必要的工具。

## 完成标准
- 只有在改动经过运行/测试验证通过后，才允许总结结束；没验证就直接结束是不合格的。
- 改动后可用 git_status / git_diff 检查自己改了什么，确认符合预期再收尾。

## 安全
- 只修改与任务相关的文件，不删除用户的重要文件；
- 执行可能有副作用的命令前，先想清楚。
"""

# 死循环保护：同一工具 + 同一参数连续重复调用超过该次数即终止
REPEAT_CALL_LIMIT = 3

# 终端回显的长度上限（只影响显示，不影响发给模型的内容）
TOOL_RESULT_PREVIEW_CHARS = 80
TOOL_ARGS_PREVIEW_CHARS = 120


def _preview(text: str, max_chars: int = TOOL_RESULT_PREVIEW_CHARS) -> str:
    """把工具结果压成一行简短预览（换行转空格），用于终端轨迹显示。"""
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    return flat[:max_chars] + " …"


class Agent:
    """把 LLM 客户端、本地工具与 ContextManager 串起来的编排器。"""

    def __init__(
        self,
        config: Config,
        client: LLMClient | None = None,
        before_tool: Callable[[str, dict], bool] | None = None,
        session: SessionMemory | None = None,
    ):
        self.config = config
        self.client = client or LLMClient(config)
        self.before_tool = before_tool  # 工具执行前的确认钩子（返回 False 跳过）
        self.session = session  # 跨任务会话记忆（None 表示无记忆）

    def run(self, task: str) -> str:
        """执行一个任务，返回模型的最终文本（过程已流式打印）。"""
        # 跨任务记忆：把之前任务的历史注入到发给模型的上下文，但不改 task 本身，
        # 这样 session.add(task, result) 存的是干净任务，历史不会越嵌越深
        context_task = task
        if self.session is not None:
            history = self.session.render()
            if history:
                context_task = history + "\n\n（当前任务）\n" + task
        cm = ContextManager(
            system_prompt=SYSTEM_PROMPT,
            task=context_task,
            max_tokens=self.config.max_context_tokens,
            tool_result_max_chars=self.config.tool_result_max_chars,
            summarizer=self._summarize if self.config.enable_summarization else None,
        )
        schemas = get_tool_schemas()

        total_tool_calls = 0                # 本次任务累计的工具调用次数
        last_call_key: str | None = None    # 上一次工具调用签名（用于死循环检测）
        repeat_streak = 0

        # 完成闸门状态：防止「改了文件却从不验证就直接说完成」的假完成
        mutation_unverified = False  # 是否有「改过文件但还没运行验证」的改动
        verify_nudged = False        # 针对当前未验证改动，是否已提醒过一次

        for turn in range(1, self.config.max_turns + 1):
            try:
                messages = cm.build_messages()
                msg = self.client.chat(messages, tools=schemas, stream=True)
            except ContextBudgetError as e:
                return f"（上下文预算错误：{e}）"
            except Exception as e:
                return f"（调用模型失败：{e}）"

            # 没有工具调用 → 模型认为任务完成
            if not msg.get("tool_calls"):
                # 完成闸门：改过文件却一直没运行验证，先提醒一次，避免「假完成」
                if mutation_unverified and not verify_nudged:
                    print()
                    print("[提示] 检测到文件改动后尚未运行验证，先提醒模型验证")
                    # 记下模型这次的「完成」说法（去掉无用的 tool_calls 字段），保持对话连贯
                    cm.add_turn({"role": "assistant", "content": msg.get("content")}, [])
                    cm.add_user_message(
                        "（系统提醒）你刚才修改了文件，但还没有运行验证。"
                        "请先用 run_command 运行程序或测试，确认通过后再总结结束；"
                        "若本次改动确实无需运行验证，请简要说明原因。"
                    )
                    verify_nudged = True
                    continue
                print()
                return msg.get("content") or "（模型未给出文本）"

            print()
            tool_results: list[dict] = []
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                raw = tc["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    args = {}
                    print(f"[警告] 工具参数非法 JSON，已忽略：{raw!r}")

                print(f"→ 调用工具：{name}({_preview(json.dumps(args, ensure_ascii=False), TOOL_ARGS_PREVIEW_CHARS)})")

                total_tool_calls += 1
                if total_tool_calls > self.config.max_tool_calls:
                    print(f"[停止] 已达到最大工具调用次数 {self.config.max_tool_calls}")
                    return f"（已达到最大工具调用次数 {self.config.max_tool_calls}，任务可能未完成）"

                # 死循环检测
                call_key = f"{name}\x00{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                repeat_streak = repeat_streak + 1 if call_key == last_call_key else 1
                last_call_key = call_key
                if repeat_streak >= REPEAT_CALL_LIMIT:
                    print(f"[停止] 连续 {REPEAT_CALL_LIMIT} 次重复调用 {name}，疑似死循环，已终止")
                    return "（检测到重复调用，已终止以避免死循环，请人工介入）"

                if self.before_tool and not self.before_tool(name, args):
                    print(f"[跳过] 已取消该工具调用")
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": name,
                        "content": "用户取消了该操作。请调整方案或询问用户。",
                    })
                    continue
                result = execute_tool(name, args)
                # 维护「改动 vs 验证」状态，供完成闸门判断（失败不算改动/验证）
                if name in {"write_file", "edit_file"} and not result.is_error:
                    mutation_unverified = True
                    verify_nudged = False
                elif name == "run_command" and not result.is_error:
                    mutation_unverified = False
                    verify_nudged = False
                # 回显工具结果预览（错误单独标出）
                if result.is_error:
                    print(f"  ↳ 错误：{_preview(result.output)}")
                else:
                    print(f"  ↳ {_preview(result.output)}")
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": name,
                    "content": cm.limit_tool_result(result.output),
                })

            # 本轮交互写入 ContextManager
            cm.add_turn(msg, tool_results)

        return f"（已达到最大轮数上限 {self.config.max_turns}，任务可能未完成）"

    def _summarize(self, messages: list[dict]) -> str:
        """把历史交互交给模型，生成任务状态摘要（非流式，不打印到终端）。"""
        resp = self.client.chat(messages, tools=None, stream=False)
        return resp.get("content") or ""
