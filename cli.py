"""命令行入口。

第 1 步先用一个最小 REPL 验证「能聊天」；后续步骤会扩展成
带工具的单任务模式 + 完整交互。

用法：
  python cli.py        # 进入对话（输入 exit 退出）
"""

from __future__ import annotations

from coding_agent.config import Config
from coding_agent.llm import LLMClient


def main() -> None:
    try:
        cfg = Config.load()
    except RuntimeError as e:
        print(f"[配置错误] {e}")
        return

    client = LLMClient(cfg)
    history: list[dict] = [
        {"role": "system", "content": "你是一个乐于助人的编程助手。"}
    ]

    print(f"已连接模型：{cfg.model}（base_url={cfg.base_url}）")
    print("输入 exit / quit / 退出 结束对话。\n")

    while True:
        try:
            user = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user.lower() in {"exit", "quit", "退出"}:
            break
        if not user:
            continue

        history.append({"role": "user", "content": user})
        print("助手 > ", end="", flush=True)
        try:
            msg = client.chat(history)
        except Exception as e:  # 网络/鉴权等错误，给出可读提示
            print(f"\n[调用失败] {e}")
            print("请检查 .env 里的 DEEPSEEK_API_KEY 是否正确、网络是否可用。")
            history.pop()  # 撤销这条没得到回复的 user 消息
            continue
        history.append(msg)
        print("\n")


if __name__ == "__main__":
    main()
