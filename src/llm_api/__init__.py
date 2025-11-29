# src/llm_api/__init__.py

# 只从该包中暴露 LLMRequestFactory，作为与外部模块交互的唯一、统一的入口点。
# 这种方式隐藏了所有内部实现细节（如 client, request, registry 等）。
from .factory import LLMRequestFactory

# 同时暴露异常类，方便上层捕获
from .model_client.base_client import LLMException, APIError, NetworkConnectionError, EmptyResponseError

__all__ = [
    "LLMRequestFactory",
    "LLMException",
    "APIError",
    "NetworkConnectionError",
    "EmptyResponseError"
]
