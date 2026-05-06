from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_lab.core.tools import Tool, ToolRegistry


def main() -> None:
    # Registry 是 runtime 未来查找和调用工具的统一入口。
    registry = ToolRegistry()

    # 一个可靠 tool 至少包含 name、description、input_schema 和 callable。
    calculator = Tool(
        name="adder",
        description="Add two integers.",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        fn=lambda args: args["a"] + args["b"],
    )

    print("[TOOL_REGISTRY] register adder")
    registry.register(calculator)

    # list_tools 模拟 runtime 把可用工具说明交给 LLM。
    print(f"[TOOL_REGISTRY] list_tools {registry.list_tools()}")

    # 正常调用返回 ok=True 和 output。
    print('[TOOL_CALL] adder {"a": 2, "b": 3}')
    result = registry.call("adder", {"a": 2, "b": 3})
    print(
        f"[TOOL_RESULT] ok={result.ok} output={result.output} error={result.error}"
    )

    # 错误参数不会进入 fn，而是返回结构化 error。
    print('[TOOL_CALL] adder {"a": "2", "b": 3}')
    invalid_result = registry.call("adder", {"a": "2", "b": 3})
    print(
        "[TOOL_RESULT] "
        f"ok={invalid_result.ok} "
        f"output={invalid_result.output} "
        f"error={invalid_result.error}"
    )


if __name__ == "__main__":
    main()
