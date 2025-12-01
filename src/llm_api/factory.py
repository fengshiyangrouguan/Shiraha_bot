# src/llm_api/factory.py
from typing import Dict
from .request import LLMRequest

class LLMRequestFactory:
    """
    创建并缓存不同任务的 LLMRequest 实例。
    工厂本身不再是单例，由 main_system 负责确保只创建一次。
    """
    def __init__(self):
        # 实例自己的缓存
        self._cache: Dict[str, LLMRequest] = {}

    def get_request(self, task_name: str) -> LLMRequest:
        """
        根据任务名称获取一个 LLMRequest 实例。
        如果已创建，则从缓存中返回。
        """
        if task_name not in self._cache:
            print(f"LLMRequestFactory: 首次为任务 '{task_name}' 创建 LLMRequest 实例。")
            self._cache[task_name] = LLMRequest(task_name=task_name)
        
        return self._cache[task_name]


