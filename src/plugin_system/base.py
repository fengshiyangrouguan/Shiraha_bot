# E:\project\Shiraha_bot\shirahabot\plugin_system\base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type

class BaseTool(ABC):
    """
    所有工具的基类。
    每个工具都应该有名称、描述和参数定义，以便规划器（LLM）理解其功能。
    """
    name: str = "base_tool"
    description: str = "这是一个基础工具模板。"
    parameters: List[Dict[str, Any]] = []

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行工具的核心逻辑。
        """
        raise NotImplementedError

    @classmethod
    def get_definition(cls) -> Dict[str, Any]:
        """
        返回一个符合LLM Tool-Calling API格式的字典。
        """
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param['name']: {
                            'type': param['type'],
                            'description': param['description']
                        } for param in cls.parameters
                    },
                    "required": [param['name'] for param in cls.parameters if param.get('required', False)],
                },
            },
        }

class BasePlugin(ABC):
    """
    所有插件的基类。
    插件是工具的集合。
    """
    @abstractmethod
    def get_tools(self) -> List[Type[BaseTool]]:
        """
        返回该插件提供的所有工具类。
        """
        raise NotImplementedError
