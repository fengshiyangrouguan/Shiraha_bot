# llm_api/client.py
import httpx
from typing import List, Dict, Optional, Any

class APIResponse:
    """标准化的API响应对象"""
    def __init__(self, content: str = "", usage: Optional[Dict] = None):
        self.content = content
        self.usage = usage

class BaseClient:
    """所有API客户端的基类，定义了通用接口。"""
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    async def get_response(self, llm_model_name: str, messages: List[Dict], **kwargs) -> APIResponse:
        """
        发送请求并获取LLM的回复。
        所有子类都必须实现这个方法。
        """
        raise NotImplementedError

class GenericOpenAIClient(BaseClient):
    """
    一个通用的、与OpenAI API格式兼容的客户端。
    """
    async def get_response(self, llm_model_name: str, messages: List[Dict], **kwargs) -> APIResponse:
        """
        使用httpx异步发送请求。
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": llm_model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }

        api_url = f"{self.base_url.rstrip('/')}/chat/completions"

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                print(f"向 {api_url} 发送请求...")
                response = await client.post(api_url, headers=headers, json=payload)
                
                # 检查HTTP响应状态码
                response.raise_for_status()

                data = response.json()
                
                # 检查返回的数据是否符合预期
                if not data.get("choices") or not data["choices"][0].get("message"):
                    raise ValueError("API响应中缺少预期的'choices'或'message'字段")

                content = data["choices"][0]["message"].get("content", "")
                usage = data.get("usage")
                
                return APIResponse(content=content.strip(), usage=usage)

            except httpx.HTTPStatusError as e:
                print(f"API请求失败，状态码: {e.response.status_code}, 响应: {e.response.text}")
                # 将HTTP错误重新抛出，以便上层（LLMRequest）可以捕获并处理
                raise ConnectionError(f"HTTP Status {e.response.status_code}") from e
            except httpx.RequestError as e:
                print(f"网络请求失败: {e}")
                # 将网络错误重新抛出
                raise ConnectionError(f"Request failed: {e}") from e
            except ValueError as e:
                print(f"解析API响应失败: {e}")
                raise

# --- 客户端工厂 ---
# 这是一个非常简单的工厂，未来可以扩展以支持不同类型的客户端
_client_instances: Dict[str, BaseClient] = {}

def get_client_for_model(model_config: Dict[str, Any]) -> BaseClient:
    """
    根据模型配置获取或创建一个客户端实例。
    使用单例模式，确保每个base_url只创建一个客户端实例。
    """
    base_url = model_config["base_url"]
    if base_url not in _client_instances:
        # 目前我们只实现了一个通用客户端，所以直接用它
        # 未来如果需要支持不兼容OpenAI API的客户端，可以在这里添加逻辑
        print(f"为 base_url '{base_url}' 创建新的 GenericOpenAIClient 实例。")
        _client_instances[base_url] = GenericOpenAIClient(
            api_key=model_config["api_key"],
            base_url=base_url
        )
    return _client_instances[base_url]
