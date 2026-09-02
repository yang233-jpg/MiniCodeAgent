# MiniCodeAgent

一个**自研的编程智能体（coding agent）**——通过 OpenAI 兼容接口调用大模型，在本地读写文件、执行命令，自主完成编程任务。可以理解为一个精简版的 Claude Code / Codex。

本项目的核心逻辑**全部自行实现**，不依赖 LangChain / LlamaIndex / OpenAI Agents SDK 等任何 agent 框架或 SDK；仅使用厂商客户端库 `openai` 与 OpenAI 兼容网关完成最底层的模型调用。

## 特性

- **流式输出 + 原生 function calling**：模型文本实时打印，`tool_calls` 分片按 index 归并成完整结构
- **自研上下文管理**：token 预算、超预算时裁剪或摘要旧历史、单个工具结果限长
- **跨任务会话记忆**：同一次运行内，后续任务能回忆之前做过的事（容量有界、不落盘）
- **完成闸门**：改过文件却未运行验证时提醒一次，防止「假完成」
- **执行控制**：最大轮数、最大工具调用次数、连续重复调用死循环检测
- **交互式 CLI**：执行命令前确认（`y` 执行 / `a` 之后全部执行 / 回车跳过），`-y` 全程跳过
- **错误处理**：工具异常捕获、上下文预算错误、模型调用失败，均不使循环崩溃

## 安装

> 需要 Python 3.10+（代码使用了 `X | Y` 联合类型语法）。

```bash
git clone https://github.com/yang233-jpg/MiniCodeAgent.git
cd MiniCodeAgent
python -m venv .venv
# Windows：.venv\Scripts\activate     macOS/Linux：source .venv/bin/activate
pip install -r requirements.txt
```

依赖只有一个：

```
openai>=1.40,<2
```

## 配置

复制 `.env.example` 为 `.env` 并填入真实值（`.env` 已被 `.gitignore` 忽略，不会进仓库）：

```bash
DEEPSEEK_API_KEY=sk-xxxxx            # 必填
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

`DEEPSEEK_BASE_URL` 指向任意 OpenAI 兼容网关，换成其他厂商只需改这一处。可选配置（均有默认值，见 `.env.example`）：

| 变量 | 含义 | 默认 |
| --- | --- | --- |
| `AGENT_MAX_TURNS` | 单任务最大迭代轮数 | 30 |
| `AGENT_MAX_TOOL_CALLS` | 单任务最大工具调用次数 | 80 |
| `AGENT_TIMEOUT` | 单次模型调用超时（秒） | 120 |
| `AGENT_MAX_CONTEXT_TOKENS` | 上下文 token 预算 | 40000 |
| `AGENT_TOOL_RESULT_MAX_CHARS` | 单个工具结果进上下文前的字符上限 | 8000 |
| `AGENT_ENABLE_SUMMARIZATION` | 超预算时摘要旧历史（0 则直接丢弃） | 1 |

## 使用

**单任务模式**（跑完即退出，自动执行不确认）：

```bash
python cli.py "在当前目录写一个 hello.py 打印 Hello，然后运行它"
```

**交互模式**（逐条输入任务，执行 `run_command` 前询问确认）：

```bash
python cli.py
python cli.py -y    # 全程不确认，直接执行
```

## 项目结构

```
MiniCodeAgent/
├── cli.py                  # 命令行入口（单任务 / 交互两种模式）
├── requirements.txt        # 依赖（仅 openai）
├── .env.example            # 环境变量示例（真实 .env 不入库）
└── coding_agent/
    ├── __init__.py         # 包文档 + __version__
    ├── agent.py            # 主循环：编排 LLM 与工具，循环终止 + 完成闸门
    ├── llm.py              # LLM 客户端：流式 + tool_calls 拼接
    ├── config.py           # 配置加载（环境变量 / .env）
    ├── history.py          # 上下文管理 ContextManager（裁剪 / 摘要 / 限长）
    ├── session.py          # 跨任务会话记忆 SessionMemory
    └── tools/
        ├── __init__.py     # 导入各子模块，触发工具注册
        ├── base.py         # 工具基础设施：注册表 + schema + executor
        ├── files.py        # read_file / write_file / edit_file / list_dir
        ├── search.py       # glob / grep
        ├── shell.py        # run_command
        └── git.py          # git_status / git_diff（只读）
```

## 核心设计（自研部分）

作业要求自写的五项核心能力，对应实现如下：

| 能力 | 实现位置 | 说明 |
| --- | --- | --- |
| 对话历史与上下文管理 | `history.py` | token 预算、超预算按轮裁剪或交模型摘要、工具结果限长 |
| 工具定义与本地执行 | `tools/` | 注册表 + JSON Schema 提供给模型，executor 本地执行 |
| 模型输出解析 | `llm.py` | 流式文本与 `tool_calls` 分片按 index 归并 |
| 循环终止条件 | `agent.py` | 无工具调用即结束 + 完成闸门 + 轮数/调用次数/重复检测 |
| 错误处理 | 各处 | 工具异常、上下文预算、模型调用失败统一回传不崩溃 |

## 安全说明

- API key 只从环境变量或未入库的 `.env` 读取，**绝不硬编码进代码**
- `run_command` 默认在交互模式下需人工确认；单任务模式请自行评估任务安全性
- `git_status` / `git_diff` 为只读工具，agent 不能直接提交、推送


