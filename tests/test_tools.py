from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_lab.core.tools import Tool, ToolRegistry


def test_registry_calls_registered_tool() -> None:
    # 正常路径：注册工具后，通过 registry 按名字调用。
    registry = ToolRegistry()
    registry.register(
        Tool(
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
    )

    result = registry.call("adder", {"a": 2, "b": 3})

    assert result.ok is True
    assert result.output == 5
    assert result.error is None


def test_registry_returns_structured_error_for_unknown_tool() -> None:
    # 未知工具不应该抛未捕获异常，而应该返回 ToolResult。
    registry = ToolRegistry()

    result = registry.call("missing", {})

    assert result.ok is False
    assert result.output is None
    assert result.error == "Unknown tool: missing"


def test_tool_call_catches_internal_exception() -> None:
    # 工具自己的异常要被捕获，否则后续 agent loop 会直接崩掉。
    def broken_tool(args: dict) -> None:
        raise RuntimeError("boom")

    tool = Tool(
        name="broken",
        description="Always fails.",
        input_schema={"type": "object"},
        fn=broken_tool,
    )

    result = tool.call({})

    assert result.ok is False
    assert result.output is None
    assert result.error == "boom"


def test_registry_list_tools_returns_schema_information() -> None:
    # list_tools 给 LLM/runtime 看的是 schema 信息，不暴露 Python callable。
    registry = ToolRegistry()
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    registry.register(
        Tool(
            name="echo",
            description="Return the input text.",
            input_schema=schema,
            fn=lambda args: args["text"],
        )
    )

    assert registry.list_tools() == [
        {
            "name": "echo",
            "description": "Return the input text.",
            "input_schema": schema,
        }
    ]


def test_tool_call_validates_required_arguments() -> None:
    # required 校验发生在真正执行 fn 之前。
    tool = Tool(
        name="echo",
        description="Return the input text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=lambda args: args["text"],
    )

    result = tool.call({})

    assert result.ok is False
    assert result.output is None
    assert result.error == "Missing required argument: text"


def test_tool_call_validates_argument_types() -> None:
    # 类型校验能把明显错误的 LLM 参数挡在工具边界外。
    tool = Tool(
        name="echo",
        description="Return the input text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=lambda args: args["text"],
    )

    result = tool.call({"text": 123})

    assert result.ok is False
    assert result.output is None
    assert result.error == "Invalid type for argument text: expected string"
