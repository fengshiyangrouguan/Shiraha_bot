# src/cortices/tools_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Coroutine, Callable

class BaseTool(ABC):
    """
    工具的抽象基类。
    每个工具都是一个包含元数据和执行逻辑的可调用对象。
    """

    @property
    @abstractmethod
    def scope(self) -> str:
        """
        返回工具的作用域，例如 'main' 或 'qq_chat'。
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        返回工具的元数据字典，用于生成 OpenAI-compatible 的工具 Schema。
        
        应包含:
        - name: str
        - description: str
        - parameters: Dict (JSON Schema)
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        执行工具的核心逻辑。
        
        使用 **kwargs 接收所有由 LLM 提供的参数。
        """
        pass

    def get_schema(self) -> Dict[str, Any]:
        """
        根据元数据生成完整的 OpenAI-compatible 工具 Schema。
        """
        meta = self.metadata
        return {
            "type": "function",
            "function": {
                "name": meta["name"],
                "description": meta["description"],
                "parameters": {
                    "type": "object",
                    "properties":meta["parameters"],
                    "required":meta["required_parameters"]
                }
            }
        }
