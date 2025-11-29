# src/llm_api/request.py
import time
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

    async def execute(self, prompt: str, **kwargs) -> Tuple[str, str]:
        """
        执行LLM请求，返回一个字符串结果。
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
                
                request_params = {
                    "temperature": self.task_config.temperature,
                    "max_tokens": self.task_config.max_tokens,
                    **kwargs
                }

                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在尝试使用模型: {model_config.name}...")
                self.model_states[model_config.name] = (self.model_states[model_config.name][0], time.time())

                # 注意：客户端内部的 tenacity 会处理临时的 NetworkError 和 RateLimitError
                # 这里捕获的是重试耗尽后，或不可重试的“硬”错误
                content = await client.get_response(model_config=model_config, prompt=prompt, **request_params)
                
                self.model_states[model_config.name] = (0, self.model_states[model_config.name][1])
                print(f"模型 '{model_config.name}' 请求成功。")
                return content, model_config.name

            except RespNotOkException as e:
                last_exception = e
                # 根据状态码决定是否应该切换模型
                if 400 <= e.status_code < 500 and e.status_code != 429:
                    # 客户端错误(4xx, 非429)，通常是请求本身有问题，切换模型也无用，应直接终止
                    print(f"模型 '{model_config.name}' 遇到客户端错误 (Code: {e.status_code})，终止所有尝试。错误: {e}")
                    break 
                else:
                    # 服务器端错误(5xx)或速率限制(429)，可以尝试切换到下一个模型
                    print(f"模型 '{model_config.name}' 遇到可切换的API错误 (Code: {e.status_code})，尝试下一个模型。")

            except (NetworkConnectionError, EmptyResponseException) as e:
                # 网络错误重试耗尽，或空响应，都适合切换到下一个模型
                last_exception = e
                print(f"模型 '{model_config.name}' 遇到问题，尝试下一个模型。错误: {e}")

            except Exception as e:
                last_exception = e
                print(f"模型 '{model_config.name}' 遇到未知错误，尝试下一个模型。错误: {e}")

            # 如果代码执行到这里，说明当前模型尝试失败
            failure_count, last_used = self.model_states[model_config.name]
            self.model_states[model_config.name] = (failure_count + 1, last_used)
            failed_models.add(model_config.name)
        
        # 循环结束，所有模型尝试失败
        error_message = f"任务 '{self.task_name}' 的所有可用模型均已尝试失败。"
        print(error_message)
        if last_exception:
            raise ModelAttemptFailed(error_message, original_exception=last_exception) from last_exception
        raise ModelAttemptFailed(error_message)