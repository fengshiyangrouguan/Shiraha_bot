from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.common.logger import get_logger

logger = get_logger("tool_registry")


@dataclass
class ToolDescriptor:
    name: str
    description: str
    scopes: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


@dataclass
class RegisteredTool:
    descriptor: ToolDescriptor
    executor: Callable[..., Awaitable[Any]]


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, RegisteredTool] = {}

    def register_tool(
        self,
        descriptor: ToolDescriptor,
        executor: Callable[..., Awaitable[Any]],
    ) -> None:
        if descriptor.name in self._tools:
            logger.warning(f"工具 '{descriptor.name}' 被重复注册，后一个定义将覆盖前一个。")
        self._tools[descriptor.name] = RegisteredTool(
            descriptor=descriptor,
            executor=executor,
        )

    def get_tool(self, name: str) -> Optional[RegisteredTool]:
        return self._tools.get(name)

    def clear(self) -> None:
        self._tools.clear()

    def list_descriptors(self, scopes: List[str] | str | None = None) -> List[ToolDescriptor]:
        if scopes is None:
            return [registered.descriptor for registered in self._tools.values()]

        if isinstance(scopes, str):
            scopes = [scopes]

        requested_scopes = set(scopes)
        return [
            registered.descriptor
            for registered in self._tools.values()
            if requested_scopes & set(registered.descriptor.scopes)
        ]

    def get_tool_schemas(self, scopes: List[str] | str) -> List[Dict[str, Any]]:
        return [descriptor.to_openai_schema() for descriptor in self.list_descriptors(scopes)]

    async def execute_tool(self, name: str, **kwargs) -> Any:
        registered = self.get_tool(name)
        if not registered:
            error_msg = f"错误：尝试调用一个未注册的工具 '{name}'。"
            logger.error(error_msg)
            return {"error": error_msg}

        try:
            return await registered.executor(**kwargs)
        except Exception as exc:
            logger.error(f"调用工具 '{name}' 时发生错误: {exc}", exc_info=True)
            return {"error": f"错误调用工具 '{name}': {exc}"}
