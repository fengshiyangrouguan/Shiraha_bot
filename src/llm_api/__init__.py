# src/llm_api/__init__.py

# 只从该包中暴露 LLMRequestFactory，作为与外部模块交互的唯一、统一的入口点。
# 这种方式隐藏了所有内部实现细节（如 client, request, registry 等）。
from .factory import LLMRequestFactory

# 同时暴露异常类，方便上层捕获
from .exceptions import NetworkConnectionError,ReqAbortException,RespNotOkException,RespParseException,EmptyResponseException,ModelAttemptFailed

__all__ = [
    "LLMRequestFactory",
    "NetworkConnectionError",
    "ReqAbortException",
    "RespNotOkException",
    "RespParseException",
    "EmptyResponseException",
    "ModelAttemptFailed",
]
