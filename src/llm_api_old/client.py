# src/llm_api/client.py

import asyncio
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Type

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# 从新的config模块导入数据类
from ..config import APIProviderConfigConfig, ModelConfig

# --- 1. 标准化数据模型 (与之前相同，保持在此处以便模块自包含) ---

class ToolCall:
    """工具调用请求的数据类"""
    def __init__(self, call_id: str, func_name: str, args: Optional[Dict[str, Any]]):
        self.call_id = call_id
        self.func_name = func_name
        self.args = args

    def __repr__(self):
        return f"ToolCall(id={self.call_id}, func='{self.func_name}', args={self.args})"

class APIResponse:
    """标准化的API响应对象"""
    def __init__(self, 
                 content: Optional[str] = None, 
                 reasoning: Optional[str] = None, # <--- 新增字段
                 tool_calls: Optional[List[ToolCall]] = None, 
                 usage: Optional[Dict] = None):
        self.content = content
        self.reasoning = reasoning # <--- 新增字段
        self.tool_calls = tool_calls
        self.usage = usage
        
    def __repr__(self):
        return (f"APIResponse(content='{self.content}', reasoning='{self.reasoning}', "
                f"tool_calls={self.tool_calls}, usage={self.usage})")

# --- 异常类 (与之前相同) ---
class LLMException(Exception):
    """LLM相关操作的基类异常"""
    pass

class NetworkConnectionError(LLMException):
    """网络连接错误"""
    pass

class APIError(LLMException):
    """API返回非2xx状态码的错误"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")

class EmptyResponseError(LLMException):
    """API返回了空内容"""
    pass

# --- 2. 客户端基类与注册表 (升级版) ---

class BaseClient(ABC):
    """所有API客户端的基类，定义了通用接口。"""
    
    def __init__(self, provider_config: APIProviderConfigConfig):
        self.provider_config = provider_config

    @abstractmethod
    async def get_response(
        self, 
        model_config: ModelConfig, 
        messages: List[Dict], 
        tools: Optional[List[Dict]] = None, 
        **kwargs
    ) -> APIResponse:
        """发送请求并获取LLM的回复。"""
        raise NotImplementedError

class ClientRegistry:
    """客户端注册表，用于管理不同类型的客户端。"""
    def __init__(self):
        self._registry: Dict[str, Type[BaseClient]] = {}
        self._instances: Dict[str, BaseClient] = {}

    def register(self, client_type: str):
        def decorator(cls: Type[BaseClient]) -> Type[BaseClient]:
            self._registry[client_type] = cls
            return cls
        return decorator

    def get_client(self, provider_config: APIProviderConfigConfig) -> BaseClient:
        """
        根据 Provider 配置获取或创建一个客户端实例 (单例模式)。
        实例的唯一键是 Provider 的名称。
        """
        instance_key = provider_config.name
        if instance_key not in self._instances:
            client_class = self._registry.get(provider_config.client_type)
            if not client_class:
                raise ValueError(f"未在客户端注册表中找到类型为 '{provider_config.client_type}' 的客户端。")
            self._instances[instance_key] = client_class(provider_config=provider_config)
        return self._instances[instance_key]

client_registry = ClientRegistry()


# --- 3. 具体客户端实现 (升级版) ---

@client_registry.register("openai")
class OpenAIClient(BaseClient):
    """
    一个通用的、与OpenAI API格式兼容的客户端。
    现在它从 APIProviderConfigConfig 对象中获取配置。
    """
    def __init__(self, provider_config: APIProviderConfigConfig):
        super().__init__(provider_config)
        self.client = AsyncOpenAI(
            api_key=self.provider_config.api_key, 
            base_url=self.provider_config.base_url,
            timeout=self.provider_config.timeout
        )

    async def get_response(
        self, 
        model_config: ModelConfig, 
        messages: List[Dict], 
        tools: Optional[List[Dict]] = None, 
        **kwargs
    ) -> APIResponse:
        """
        使用 aiohttp 异步发送请求。
        集成了从配置中读取的重试和错误处理机制。
        """
        # 使用 tenacity 实现重试逻辑，配置来自 provider_config
        retry_decorator = retry(
            wait=wait_random_exponential(min=1, max=self.provider_config.retry_interval),
            stop=stop_after_attempt(self.provider_config.max_retry + 1), # +1 是因为第一次尝试也算在内
            retry=retry_if_exception_type((APIConnectionError, RateLimitError))
        )
        
        # 动态应用装饰器
        decorated_func = retry_decorator(self._internal_get_response)
        return await decorated_func(model_config, messages, tools, **kwargs)


    def _parse_cot_response(self, text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        解析包含 CoT (Chain-of-Thought) 的响应文本。
        分离 <think>...</think> 标签内的思考过程和外部的最终答案。
        """
        if not text:
            return None, None

        # 使用正则表达式查找并分离<think>标签
        # re.DOTALL 标志让 . 可以匹配包括换行符在内包括换行符的任意字符
        match = re.match(r"<think>(.*?)</think>", text, re.DOTALL)
        
        if match:
            reasoning = match.group(1).strip()
            # 从原始文本中移除 <think>...</think> 部分，剩下的就是最终内容
            content = text[match.end():].strip()
            return content, reasoning
        else:
            # 如果没有找到 <think> 标签，则全部视为最终内容
            return text, None

    async def _internal_get_response(
        self, 
        model_config: ModelConfig, 
        messages: List[Dict], 
        tools: Optional[List[Dict]] = None, 
        **kwargs
    ) -> APIResponse:
        """实际执行请求的内部方法"""
        try:
            # 准备请求参数，合并来自多方的参数
            params = {
                "model": model_config.model_identifier,
                "messages": self._convert_messages(messages),
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 4096),
                **model_config.extra_params, # 合并模型自带的额外参数
                **kwargs.get("extra_body", {}) # 合并调用时临时传入的额外参数
            }
            if tools:
                params["tools"] = self._convert_tools(tools)

            # 发送请求 (目前暂未实现 force_stream_mode)
            response = await self.client.chat.completions.create(**params, stream=False)

            # 解析响应
            choice = response.choices[0]
            message = choice.message
            
            # 先获取原始 content
            raw_content = message.content
            
            # 调用新的解析方法来分离 CoT 内容
            content, reasoning = self._parse_cot_response(raw_content)
            
            tool_calls = None
            if message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {"raw_arguments": tc.function.arguments}
                    tool_calls.append(
                        ToolCall(call_id=tc.id, func_name=tc.function.name, args=args)
                    )
            
            if not content and not tool_calls:
                raise EmptyResponseError("API返回了空内容和空工具调用。")

            return APIResponse(
                content=content,
                reasoning=reasoning, # <--- 传入解析出的思考过程
                tool_calls=tool_calls,
                usage=response.usage.dict() if response.usage else None
            )

        except APIStatusError as e:
            raise APIError(status_code=e.status_code, message=e.message) from e
        except APIConnectionError as e:
            raise NetworkConnectionError(f"无法连接到API at {self.provider_config.base_url}: {e}") from e
        except Exception as e:
            raise LLMException(f"发生未知错误: {e}") from e


    def _convert_messages(self, messages: List[Dict]) -> List[ChatCompletionMessageParam]:
        """将我们的消息格式转换为OpenAI API所需格式。"""
        # 这里可以添加更复杂的消息验证和转换逻辑
        # 为简化，我们假设传入的消息格式已经是兼容的
        return messages

    def _convert_tools(self, tools: List[Dict]) -> List[ChatCompletionToolParam]:
        """将我们的工具格式转换为OpenAI API所需格式。"""
        # 为简化，我们假设传入的工具格式已经是兼容的
        return tools
