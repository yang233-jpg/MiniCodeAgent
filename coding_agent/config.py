"""配置加载：从环境变量或项目根目录的 .env 文件读取配置。

凭据（API key）一律来自环境变量或未入库的 .env 文件，绝不硬编码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 项目根目录 = 本文件所在包的上一级（即仓库根目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析器：把 KEY=VALUE 行读入 os.environ。

    只处理空行、# 注释、KEY=VALUE 三类行；已存在的环境变量优先，.env 不覆盖它。
    自写实现，不依赖 python-dotenv。
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Config:
    """运行所需的全部配置项。frozen 表示创建后不可改，避免中途被篡改。"""

    api_key: str
    base_url: str
    model: str
    max_turns: int
    max_tool_calls: int
    timeout: float
    max_context_tokens: int
    tool_result_max_chars: int
    enable_summarization: bool

    @classmethod
    def load(cls) -> "Config":
        _load_dotenv(PROJECT_ROOT / ".env")

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "未找到 DEEPSEEK_API_KEY。\n"
                "请在项目根目录创建 .env 文件（内容参考 .env.example），"
                "或设置环境变量 DEEPSEEK_API_KEY。"
            )

        return cls(
            api_key=api_key,
            # base_url 指向 OpenAI 兼容网关；换成其他厂商只改这里
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            max_turns=int(os.environ.get("AGENT_MAX_TURNS", "30")),
            # 单次任务允许的最大工具调用次数，防止模型反复调用工具失控
            max_tool_calls=int(os.environ.get("AGENT_MAX_TOOL_CALLS", "80")),
            timeout=float(os.environ.get("AGENT_TIMEOUT", "120")),
            # deepseek-chat 上下文窗口约 64K，预留输出空间后取 40K 作为裁剪预算
            max_context_tokens=int(os.environ.get("AGENT_MAX_CONTEXT_TOKENS", "40000")),
            # 单个工具结果进上下文前的字符上限
            tool_result_max_chars=int(os.environ.get("AGENT_TOOL_RESULT_MAX_CHARS", "8000")),
            # 上下文超预算时，是否用摘要压缩旧历史（0/false 则直接丢弃）
            enable_summarization=os.environ.get("AGENT_ENABLE_SUMMARIZATION", "1")
            .strip()
            .lower()
            not in ("0", "false", "no", "off"),
        )
