# src/agent/planners/base_planner.py
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.system.di.container import container
from src.llm_api.factory import LLMRequestFactory
from src.llm_api.dto import LLMMessageBuilder

class BasePlanner(ABC):
    """
    所有规划器的抽象基类。
    它封装了与 LLM 交互的通用逻辑，子类只需专注于构建自己的 Prompt。
    """
    def __init__(self, task_name: str):
        """
        初始化规划器，并从 DI 容器中获取其所需的 LLMRequest 实例。
        
        Args:
            task_name (str): 与 config.toml 中 [model_task_config] 对应的任务名称。
        """
        llm_factory = container.resolve(LLMRequestFactory)
        self.llm_request = llm_factory.get_request(task_name)
        print(f"Planner ({self.__class__.__name__}): 已初始化，使用任务配置 '{task_name}'。")

    @abstractmethod
    def _build_prompt(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """
        构建特定于此规划器的 LLM 提示。
        每个子类都必须实现此方法，以定义自己的思考方式。
        """
        raise NotImplementedError

    async def plan(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        执行一次完整的规划流程：构建提示 -> 调用 LLM -> 解析结果。
        
        Returns:
            Optional[Dict[str, Any]]: 一个包含“thought”和“action”的字典，如果失败则返回 None。
        """
        prompt_messages = self._build_prompt(*args, **kwargs)
        
        try:
            # 在这里，我们只期望 LLM 返回一个不带工具调用的纯文本 JSON 响应
            content, model_name = await self.llm_request.execute(
                prompt=json.dumps(prompt_messages) # 注意：我们将整个消息列表序列化为字符串作为 prompt
            )
            
            if not content:
                print(f"Planner ({self.__class__.__name__}): LLM ({model_name}) 未返回任何内容。")
                return None
            
            print(f"Planner ({self.__class__.__name__}): LLM ({model_name}) 规划结果: {content}")
            
            try:
                # 解析 LLM 返回的 JSON 字符串
                parsed_plan = json.loads(content)
                return parsed_plan
            except json.JSONDecodeError:
                print(f"Planner ({self.__class__.__name__}): 无法解析 LLM 返回的 JSON 响应。")
                return None # 解析失败

        except Exception as e:
            print(f"Planner ({self.__class__.__name__}): 规划时 LLM 请求失败: {e}")
            return None # 请求失败
