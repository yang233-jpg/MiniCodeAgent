"""命令行入口：让 coding agent 真正跑起来。

两种模式：
1. 单任务模式：python cli.py "任务描述" —— 跑完一个任务直接退出（自动执行，不确认）；
2. 交互模式：python cli.py —— 逐条输入任务；执行 run_command 前询问确认
   （y 执行 / a 之后全部执行 / 回车跳过），-y / --yes 全程跳过确认。

用法示例：
  python cli.py "在当前目录写一个 hello.py 打印 Hello，然后运行它"
  python cli.py
  python cli.py -y
"""

from __future__ import annotations

import argparse
from typing import Callable

from coding_agent.agent import Agent
from coding_agent.config import Config
from coding_agent.session import SessionMemory


def _make_confirm() -> Callable[[str, dict], bool]:
    """交互模式下，只对 run_command 做确认，其余工具直接放行。

    回答 y 执行本次、a 表示本次起全部执行（后续不再询问）、回车跳过。
    """

    trust_all = False

    def confirm(name: str, args: dict) -> bool:
        nonlocal trust_all
        if name != "run_command":
            return True
        if trust_all:
            return True
        command = str(args.get("command", ""))
        ans = input(
            f"\n[确认] 要执行命令：\n  {command}\n"
            "  输入 y 执行 / a 之后全部执行 / 直接回车跳过："
        ).strip().lower()
        if ans in {"a", "all", "全部", "y全部", "ya"}:
            trust_all = True
            print("  （已信任：本会话后续命令不再询问）")
            return True
        return ans in {"y", "yes", "是", "确定"}

    return confirm


def _run_task(cfg: Config, task: str, confirm: Callable[[str, dict], bool] | None,
              session: SessionMemory | None = None) -> None:
    print(f"任务：{task}\n" + "=" * 50)
    agent = Agent(cfg, before_tool=confirm, session=session)
    result = agent.run(task)
    # run() 会把模型正文流式打印、工具调用逐条打印；
    # 但错误/终止信息是以「（...）」字符串返回的（未打印），这里补打。
    if result and result.startswith("（"):
        print(result)
    print("=" * 50 + "\n")
    # 任务结束后把「任务 + 结果」记入会话记忆，供后续任务回忆
    if session is not None:
        session.add(task, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 coding agent 命令行入口")
    parser.add_argument("task", nargs="*", help="单任务描述；留空进入交互模式")
    parser.add_argument("-y", "--yes", action="store_true", help="交互模式下不确认命令，直接执行")
    args = parser.parse_args()

    try:
        cfg = Config.load()
    except RuntimeError as e:
        print(f"[配置错误] {e}")
        return

    task_text = " ".join(args.task).strip()
    if task_text:
        # 单任务模式：自动执行，不确认
        _run_task(cfg, task_text, confirm=None)
        return

    # 交互模式
    print(f"已连接模型：{cfg.model}（base_url={cfg.base_url}）")
    print("逐条输入任务描述；输入 exit / quit / 退出 结束。\n")
    confirm = None if args.yes else _make_confirm()
    session = SessionMemory()  # 本次运行内跨任务共享的记忆
    while True:
        try:
            user = input("任务 > ").strip()
        except EOFError:
            print("\n[提示] 终端没有可读取的输入（stdin 不是交互终端），程序退出。")
            print("若是 VS Code 里第一次运行：请等提示符前出现 (.venv) 后再执行 `python cli.py`。")
            break
        except KeyboardInterrupt:
            break
        if user.lower() in {"exit", "quit", "退出"}:
            break
        if not user:
            continue
        _run_task(cfg, user, confirm=confirm, session=session)


if __name__ == "__main__":
    main()