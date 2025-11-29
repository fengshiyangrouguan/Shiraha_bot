# src/llm_api/model_client/openai_client.py
from typing import Tuple

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# 导入我们统一的异常和基类
from .base_client import BaseClient, client_registry
from ..exceptions import RespNotOkException, NetworkConnectionError, EmptyResponseException

from src.common.config.schemas.llm_api_config import APIProviderConfig, ModelConfig

def _is_retryable_api_error(e: BaseException) -> bool:
    """判断一个API错误是否是可重试的（如服务器端错误、速率限制）"""
    if isinstance(e, RespNotOkException):
        # 5xx 服务器错误 和 429 速率限制 是可重试的
        return e.status_code >= 500 or e.status_code == 429
    return False

@client_registry.register("openai")
class OpenAIClient(BaseClient):
    """
    一个通用的、与OpenAI API格式兼容的客户端的轻量化实现。
    """
    def __init__(self, provider_config: APIProviderConfig):
        super().__init__(provider_config)
        self.client = AsyncOpenAI(
            api_key=self.provider_config.api_key, 
            base_url=self.provider_config.base_url,
            timeout=self.provider_config.timeout
        )

    async def get_response(self, model_config: ModelConfig, prompt: str, **kwargs) -> str:
        """
        核心调用方法，包含重试逻辑。
        """
        # tenactiy 装饰器负责处理可重试的临时性错误
        retry_decorator = retry(
            wait=wait_random_exponential(min=1, max=self.provider_config.retry_interval),
            stop=stop_after_attempt(self.provider_config.max_retry + 1),
            retry=retry_if_exception_type((NetworkConnectionError, RateLimitError))
        )
        
        decorated_func = retry_decorator(self._internal_get_response)
        return await decorated_func(model_config, prompt, **kwargs)

    async def _internal_get_response(self, model_config: ModelConfig, prompt: str, **kwargs) -> str:
        """
        实际执行请求的内部方法。
        """
        try:
            messages = [{"role": "user", "content": prompt}]

            params = {
                "model": model_config.model_identifier,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 4096),
                **model_config.extra_params,
                **kwargs.get("extra_body", {})
            }
            
            response = await self.client.chat.completions.create(**params, stream=False)

            if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
                raise EmptyResponseException() # 抛出我们自己的标准异常

            return response.choices[0].message.content.strip()

        except APIStatusError as e:
            # 将 openai 的特定异常转换为我们自己的标准异常
            raise RespNotOkException(status_code=e.status_code, message=e.message) from e
        except APIConnectionError as e:
            raise NetworkConnectionError() from e
        except Exception as e:
            # 对于其他未知错误，暂时直接抛出，或者也可以封装成一个通用异常
            raise e