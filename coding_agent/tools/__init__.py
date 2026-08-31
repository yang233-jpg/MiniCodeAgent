"""工具包：定义 agent 可用的全部本地工具并统一注册。

每个工具声明自己的 JSON Schema（供 function calling 使用），并提供
本地执行函数。外部通过 get_tool_schemas() / execute_tool() 访问，
不直接依赖具体实现。
"""
