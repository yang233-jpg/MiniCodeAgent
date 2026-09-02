"""Agent 主循环：编排 LLM 与工具，自主完成编程任务。

循环流程：ContextManager 决定本轮要发的消息 → 调模型 → 解析输出
（文本 或 tool_calls）→ 本地执行工具 → 结果连同 assistant 消息一起
作为「一轮」交还给 ContextManager → 循环，直到模型停止调用工具或达轮数上限。

上下文怎么管理（裁剪 / 摘要 / 工具结果长度限制）不在本文件里，统一由
ContextManager 负责；本文件只负责把「模型调用」和「工具执行」串起来。
"""

from __future__ import annotations

import json
from typing import Callable

from coding_agent.config import Config
from coding_agent.history import ContextBudgetError, ContextManager
from coding_agent.llm import LLMClient
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
- 一次只调用必要的工具。

## 安全
- 只修改与任务相关的文件，不删除用户的重要文件；
- 执行可能有副作用的命令前，先想清楚。
"""


class Agent:
    """把 LLM 客户端、本地工具与 ContextManager 串起来的编排器。"""

    def __init__(
        self,
        config: Config,
        client: LLMClient | None = None,
        before_tool: Callable[[str, dict], bool] | None = None,
    ):
        self.config = config
        self.client = client or LLMClient(config)
        self.before_tool = before_tool  # 每个工具执行前的确认钩子（返回 False 则跳过）

    def run(self, task: str) -> str:
        """执行一个任务，返回模型的最终文本（过程已流式打印）。"""
        cm = ContextManager(
            system_prompt=SYSTEM_PROMPT,
            task=task,
            max_tokens=self.config.max_context_tokens,
            tool_result_max_chars=self.config.tool_result_max_chars,
            summarizer=self._summarize if self.config.enable_summarization else None,
        )
        schemas = get_tool_schemas()

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
                print()
                return msg.get("content") or "（模型未给出文本）"

            print()  # 与上方流式文本分隔
            tool_results: list[dict] = []
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                raw = tc["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    args = {}
                    print(f"[警告] 工具参数非法 JSON，已忽略：{raw!r}")

                print(f"→ 调用工具：{name}({json.dumps(args, ensure_ascii=False)})")
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
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": name,
                    "content": cm.limit_tool_result(result.output),
                })

            # 把「assistant + 工具结果」作为一轮完整交互交给 ContextManager
            cm.add_turn(msg, tool_results)

        return f"（已达到最大轮数上限 {self.config.max_turns}，任务可能未完成）"

    def _summarize(self, messages: list[dict]) -> str:
        """把历史交互交给模型，生成任务状态摘要（非流式，不打印到终端）。"""
        resp = self.client.chat(messages, tools=None, stream=False)
        return resp.get("content") or ""
