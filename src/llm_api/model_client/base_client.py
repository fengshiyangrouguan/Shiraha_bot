# src/llm_api/model_client/base_client.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Type

# 导入我们新的、统一的配置和服务
from src.common.config.schemas.llm_api_config import APIProviderConfig, ModelConfig

# 异常类现在从专属文件中导入，这里不再定义
from ..exceptions import *

class BaseClient(ABC):
    """所有API客户端的基类，定义了轻量化接口。"""
    def __init__(self, provider_config: APIProviderConfig):
        self.provider_config = provider_config

    @abstractmethod
    async def get_response(self, model_config: ModelConfig, prompt: str, **kwargs) -> str:
        """
        发送请求并获取LLM的回复。
        返回一个字符串结果。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_response_with_image(self, model_config: ModelConfig, messages: list, **kwargs) -> str:
        """
        发送包含图像的请求并获取LLM的回复。
        返回一个字符串结果。
        """
        raise NotImplementedError

class ClientRegistry:
    """客户端注册表，用于管理不同类型的客户端。"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClientRegistry, cls).__new__(cls)
            cls._registry: Dict[str, Type[BaseClient]] = {}
            cls._instances: Dict[str, BaseClient] = {}
        return cls._instance

    def register(self, client_type: str):
        def decorator(cls: Type[BaseClient]) -> Type[BaseClient]:
            self._registry[client_type] = cls
            return cls
        return decorator

    def get_client(self, provider_config: APIProviderConfig) -> BaseClient:
        instance_key = provider_config.name
        if instance_key not in self._instances:
            client_class = self._registry.get(provider_config.client_type)
            if not client_class:
                raise ValueError(f"未在客户端注册表中找到类型为 '{provider_config.client_type}' 的客户端。")
            self._instances[instance_key] = client_class(provider_config=provider_config)
        return self._instances[instance_key]

client_registry = ClientRegistry()