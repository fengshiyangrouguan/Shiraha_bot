# src/llm_api/factory.py
from typing import Dict
from .request import LLMRequest

class LLMRequestFactory:
    """
    负责创建和缓存不同任务的 LLMRequest 实例。
    这是一个单例工厂。
    """
    _instance = None
    _cache: Dict[str, LLMRequest] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMRequestFactory, cls).__new__(cls)
        return cls._instance

    def get_request(self, task_name: str) -> LLMRequest:
        """
        根据任务名称获取一个 LLMRequest 实例。
        如果已创建，则从缓存中返回。
        """
        if task_name not in self._cache:
            print(f"LLMRequestFactory: 首次为任务 '{task_name}' 创建 LLMRequest 实例。")
            self._cache[task_name] = LLMRequest(task_name=task_name)
        return self._cache[task_name]

