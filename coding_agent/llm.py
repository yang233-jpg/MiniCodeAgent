"""LLM 客户端：封装 OpenAI 兼容接口的调用（支持流式）。

只负责一件事：把消息（和可选的工具定义）发给模型，返回完整的
assistant 消息（含文本与 tool_calls）。对话历史、工具执行、循环控制
等 agent 逻辑不在这里，由上层（agent.py）负责。
"""

from __future__ import annotations

from openai import OpenAI

from coding_agent.config import Config


class LLMClient:
    """基于 openai 官方客户端库的薄封装（厂商客户端，非 agent 框架）。"""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = True,
    ) -> dict:
        """调用模型，返回 assistant 消息 dict。

        返回结构（与 OpenAI Messages 兼容）：
            {"role": "assistant", "content": str|None, "tool_calls": list|None}

        当 stream=True 时，文本会边接收边打印到终端（实时反馈），
        同时把分散在各分片里的 tool_calls 增量拼接成完整结构。
        """
        kwargs = dict(
            model=self.config.model,
            messages=messages,
            timeout=self.config.timeout,
        )
        if tools:
            kwargs["tools"] = tools

        if not stream:
            resp = self.client.chat.completions.create(**kwargs)
            return _message_to_dict(resp.choices[0].message)

        # ---- 流式路径 ----
        content_parts: list[str] = []
        tool_calls: dict[int, dict] = {}  # 按 index 归并 tool_call 分片

        stream_resp = self.client.chat.completions.create(stream=True, **kwargs)
        for chunk in stream_resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                content_parts.append(delta.content)
                print(delta.content, end="", flush=True)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls[idx]["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls[idx]["function"]["arguments"] += tc.function.arguments

        content = "".join(content_parts).strip() or None
        calls = [tool_calls[i] for i in sorted(tool_calls)] or None
        return {"role": "assistant", "content": content, "tool_calls": calls}


def _message_to_dict(message) -> dict:
    """把 openai 的 message 对象转成普通 dict（非流式路径用）。"""
    msg: dict = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return msg
