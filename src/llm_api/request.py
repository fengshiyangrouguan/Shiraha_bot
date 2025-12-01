# src/llm_api/request.py
import time
import base64
from typing import List, Dict, Optional, Any, Tuple, Set

from src.system.di.container import container
from src.common.config.schemas.llm_api_config import ModelConfig
from src.common.config.schemas.llm_api_config import LLMApiConfig


# 导入客户端注册表和我们新的标准异常
from .model_client.base_client import client_registry
from .exceptions import RespNotOkException, NetworkConnectionError, EmptyResponseException, ModelAttemptFailed

class LLMRequest:
    """
    轻量化的LLM请求调度器。
    负责根据任务配置进行模型选择、故障切换和状态管理。
    """
    def __init__(self, task_name: str = "default"):
        self.task_name = task_name
        llm_config = container.resolve(LLMApiConfig)

        try:
            self.task_config = llm_config.model_task_config[task_name]
        except KeyError:
            print(f"警告：任务 '{task_name}' 配置未在 llm_api_config.toml 中找到，将使用 'default' 任务配置。")
            self.task_config = llm_config.model_task_config["default"]

        self.model_pool: Dict[str, ModelConfig] = {}
        for model_name in self.task_config.model_list:
            found = next((m for m in llm_config.models if m.name == model_name), None)
            if found:
                self.model_pool[model_name] = found
            else:
                print(f"警告：任务 '{task_name}' 配置的模型 '{model_name}' 在全局模型中未定义，已忽略。")

        if not self.model_pool:
            raise ValueError(f"任务 '{task_name}' 的模型池为空，请检查配置。")

        self.model_states: Dict[str, Tuple[int, float]] = { name: (0, 0.0) for name in self.model_pool.keys() }

    def _select_model(self, exclude_models: Set[str]) -> Optional[ModelConfig]:
        available_models = { name: state for name, state in self.model_states.items() if name not in exclude_models }
        if not available_models: return None
        sorted_models = sorted(available_models.items(), key=lambda item: (item[1][0], item[1][1]))
        best_model_name = sorted_models[0][0]
        return self.model_pool[best_model_name]

    async def _execute_request_loop(self, client_method_name: str, **method_kwargs) -> Tuple[str, str]:
        """
        通用的请求执行循环，处理模型选择、故障切换和错误处理。
        """
        failed_models: Set[str] = set()
        last_exception: Optional[Exception] = None
        max_attempts = len(self.model_pool)

        for _ in range(max_attempts):
            model_config = self._select_model(exclude_models=failed_models)
            if not model_config: break

            try:
                llm_config = container.resolve(LLMApiConfig)
                provider_config = next((p for p in llm_config.api_providers if p.name == model_config.api_provider), None)
                if not provider_config:
                    raise ValueError(f"模型 '{model_config.name}' 的 API Provider '{model_config.api_provider}' 未在配置中定义。")

                client = client_registry.get_client(provider_config)
                client_method = getattr(client, client_method_name)

                # 从kwargs中提取其他参数
                extra_kwargs = method_kwargs.pop('kwargs', {})

                # 准备请求参数，将 model_config 传递给调用的方法
                request_params = {
                    "model_config": model_config,
                    "temperature": self.task_config.temperature,
                    "max_tokens": self.task_config.max_tokens,
                    **method_kwargs,
                    **extra_kwargs
                }
                
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在尝试使用模型: {model_config.name} (任务: {self.task_name})...")
                self.model_states[model_config.name] = (self.model_states[model_config.name][0], time.time())

                content = await client_method(**request_params)
                
                self.model_states[model_config.name] = (0, self.model_states[model_config.name][1])
                print(f"模型 '{model_config.name}' 请求成功。")
                return content, model_config.name

            except RespNotOkException as e:
                last_exception = e
                if 400 <= e.status_code < 500 and e.status_code != 429:
                    print(f"模型 '{model_config.name}' 遇到客户端错误 (Code: {e.status_code})，终止所有尝试。错误: {e}")
                    break 
                else:
                    print(f"模型 '{model_config.name}' 遇到可切换的API错误 (Code: {e.status_code})，尝试下一个模型。")

            except (NetworkConnectionError, EmptyResponseException) as e:
                last_exception = e
                print(f"模型 '{model_config.name}' 遇到问题，尝试下一个模型。错误: {e}")

            except Exception as e:
                last_exception = e
                print(f"模型 '{model_config.name}' 遇到未知错误，尝试下一个模型。错误: {e}")

            failure_count, last_used = self.model_states[model_config.name]
            self.model_states[model_config.name] = (failure_count + 1, last_used)
            failed_models.add(model_config.name)
        
        error_message = f"任务 '{self.task_name}' 的所有可用模型均已尝试失败。"
        print(error_message)
        if last_exception:
            raise ModelAttemptFailed(error_message, original_exception=last_exception) from last_exception
        raise ModelAttemptFailed(error_message)

    async def execute(self, prompt: str, **kwargs) -> Tuple[str, str]:
        """
        执行LLM文本请求，返回一个字符串结果。
        """
        return await self._execute_request_loop(
            client_method_name="get_response",
            prompt=prompt,
            kwargs=kwargs
        )

    async def execute_with_image(self, prompt: str, base64_image_data: str, mime_type: str = 'image/gif', **kwargs) -> Tuple[str, str]:
        """
        执行带图片的LLM请求，返回一个字符串结果。
        接收base64编码的图片数据和mime类型。
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image_data}"
                        }
                    }
                ]
            }
        ]
        return await self._execute_request_loop(
            client_method_name="get_response_with_image",
            messages=messages,
            kwargs=kwargs
        )