# src/features/manager.py
import inspect
import json
from typing import Callable, Any, Dict, List, Coroutine

from src.llm_api.dto import ToolCall

# 类型提示
CallableTool = Callable[..., Coroutine[Any, Any, Any]]

class FeatureManager:
    """
    特性与工具的管理者。
    它使用装饰器模式来自动发现和注册工具，并负责执行它们。
    这是一个单例。
    """
    _instance = None
    # Schema 字典: {scope: {tool_name: schema}}
    # 实现字典：{tool_name: callable_function}
    _tool_schemas: Dict[str, Dict[str, Dict]]
    _tool_implementations: Dict[str, CallableTool]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FeatureManager, cls).__new__(cls)
            cls._instance._tool_schemas = {}   
            cls._instance._tool_implementations = {}
        return cls._instance

    def tool(self, scope: str = "main"):
        """
        工具注册装饰器。
        """
        def decorator(func: CallableTool) -> CallableTool:
            tool_name = func.__name__
            if tool_name in self._tool_implementations:
                print(f"警告：工具 '{tool_name}' 被重复定义。")
            
            sig = inspect.signature(func)
            docstring = inspect.getdoc(func)
            
            description = docstring.strip().split('\n')[0] if docstring else "无描述"

            parameters = {"type": "object", "properties": {}}
            required_params = []

            type_mapping = { str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object" }

            for name, param in sig.parameters.items():
                if name in ('self', 'cls'): continue
                
                param_type = type_mapping.get(param.annotation, "string")
                parameters["properties"][name] = {"type": param_type}
                
                if param.default is inspect.Parameter.empty:
                    required_params.append(name)

            if required_params:
                parameters["required"] = required_params

            schema = {
                "type": "function",
                "function": { "name": tool_name, "description": description, "parameters": parameters }
            }

            if scope not in self._tool_schemas:
                self._tool_schemas[scope] = {}
            self._tool_schemas[scope][tool_name] = schema
            self._tool_implementations[tool_name] = func
            
            print(f"工具 '{tool_name}' 已成功注册到作用域 '{scope}'。")
            return func
        return decorator

    def get_tool_schemas(self, scope: str) -> List[Dict]:
        """获取指定作用域下的所有工具 Schema 列表，用于构建 Prompt。"""
        return list(self._tool_schemas.get(scope, {}).values())

    async def execute_tool(self, tool_call: ToolCall) -> Any:
        """根据 ToolCall 对象执行相应的工具。"""
        tool_name = tool_call.func_name
        if tool_name not in self._tool_implementations:
            return f"错误：未找到名为 '{tool_name}' 的工具实现。"
        
        func = self._tool_implementations[tool_name]
        args = tool_call.args or {}

        try:
            # FeatureManager 假定所有工具都是异步函数
            return await func(**args)
        except Exception as e:
            return f"执行工具 '{tool_name}' 时出错: {e}"