from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolResult:
    # Runtime 只看这个统一结果结构，不需要猜测函数返回的是成功还是失败。
    ok: bool
    output: Any | None = None
    error: str | None = None


@dataclass(frozen=True)
class Tool:
    # Tool 是给 agent/runtime 使用的能力契约，不只是一个 Python 函数。
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[[dict[str, Any]], Any]

    def call(self, args: dict[str, Any]) -> ToolResult:
        # LLM 未来会生成结构化 args；这里先挡住非 dict 输入。
        if not isinstance(args, dict):
            return ToolResult(ok=False, error="Tool arguments must be a dict.")

        # 在执行真实函数前校验输入，避免错误进入工具内部才暴露。
        schema_error = _validate_object_schema(args, self.input_schema)
        if schema_error is not None:
            return ToolResult(ok=False, error=schema_error)

        try:
            return ToolResult(ok=True, output=self.fn(args))
        except Exception as exc:
            # 工具内部异常也转成 observation，避免打断整个 agent loop。
            return ToolResult(ok=False, error=str(exc))

    def schema_info(self) -> dict[str, Any]:
        # list_tools 暴露给 LLM/runtime 的是工具说明，不是函数对象本身。
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self) -> None:
        # 用 name 做索引，模拟 LLM 输出 tool_name 后 runtime 的查找过程。
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        # 重名工具会让 runtime 无法确定实际调用哪个能力，所以直接拒绝。
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        return [tool.schema_info() for tool in self._tools.values()]

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        # 未知工具属于模型/调用方错误，但仍然返回结构化结果。
        tool = self.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"Unknown tool: {name}")
        return tool.call(args)


def _validate_object_schema(
    args: dict[str, Any], schema: dict[str, Any]
) -> str | None:
    # 1.1 只实现最小 schema 子集：object + required + properties.type。
    if schema.get("type") not in (None, "object"):
        return "Tool input_schema must describe an object."

    required = schema.get("required", [])
    for field in required:
        if field not in args:
            return f"Missing required argument: {field}"

    properties = schema.get("properties", {})
    for field, value in args.items():
        field_schema = properties.get(field)
        if field_schema is None:
            continue

        expected_type = field_schema.get("type")
        if expected_type is not None and not _matches_json_type(value, expected_type):
            return f"Invalid type for argument {field}: expected {expected_type}"

    return None


def _matches_json_type(value: Any, expected_type: str) -> bool:
    # Python 里 bool 是 int 的子类，这里要显式排除，才符合 JSON 类型直觉。
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(
            value, bool
        )
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    return True
