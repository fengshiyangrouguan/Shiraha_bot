# src/llm_api/request.py
import time
from typing import List, Dict, Optional, Any, Tuple, Set

# 导入新的配置加载器和客户端模块
from ..config import config, ModelConfig
from .client import client_registry, APIResponse, LLMException, APIError, EmptyResponseError

class LLMRequest:
    """
    LLM请求调度器，负责根据任务配置进行模型选择、故障切换和状态管理。
    """
    def __init__(self, task_name: str = "default"):
        """
        根据任务名称初始化调度器。
        :param task_name: 任务的名称，对应 config.toml 中的一个键。
        """
        self.task_name = task_name
        try:
            self.task_config = config.get_task(task_name)
        except ValueError as e:
            print(f"警告：任务 '{task_name}' 配置未找到，将使用 'default' 任务配置。 {e}")
            self.task_config = config.get_task("default")

        # 从全局模型配置中，筛选出此任务所需的模型配置
        self.model_pool: Dict[str, ModelConfig] = {}
        for model_name in self.task_config.model_list:
            try:
                self.model_pool[model_name] = config.get_model(model_name)
            except ValueError as e:
                print(f"警告：任务 '{task_name}' 配置的模型 '{model_name}' 在全局模型中未定义，已忽略。 {e}")

        if not self.model_pool:
            raise ValueError(f"任务 '{task_name}' 的模型池为空，请检查配置。")

        # 模型使用状态，用于负载均衡和故障切换 (失败次数, 上次使用时间)
        self.model_states: Dict[str, Tuple[int, float]] = {
            name: (0, 0.0) for name in self.model_pool.keys()
        }

    def _select_model(self, exclude_models: Set[str]) -> Optional[ModelConfig]:
        """根据失败次数和最近使用情况选择一个模型。"""
        available_models = {
            name: state
            for name, state in self.model_states.items()
            if name not in exclude_models
        }
        if not available_models:
            return None

        # 排序键：(失败次数, 上次使用时间)
        sorted_models = sorted(
            available_models.items(),
            key=lambda item: (item[1][0], item[1][1])
        )
        best_model_name = sorted_models[0][0]
        return self.model_pool[best_model_name]

    async def execute(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Tuple[APIResponse, str]:
        """
        执行LLM请求，包含完整的模型选择、客户端获取、故障切换逻辑。

        :param messages: 发送给模型的消息列表。
        :param tools: 可选的工具列表。
        :param kwargs: 运行时参数，可覆盖任务的默认参数 (如 temperature, max_tokens, extra_body)。
        :return: 一个元组，包含标准化的APIResponse对象和成功响应的模型名称。
        :raises: 如果所有模型都尝试失败，则抛出最后的异常。
        """
        failed_models: Set[str] = set()
        last_exception: Optional[Exception] = None
        
        max_attempts = len(self.model_pool)
        for _ in range(max_attempts):
            model_config = self._select_model(exclude_models=failed_models)
            if not model_config:
                break  # 没有更多可用模型

            try:
                provider_config = config.get_provider(model_config.api_provider)
                client = client_registry.get_client(provider_config)
                
                # 准备请求参数，合并不同层级的配置
                # 优先级: kwargs (运行时) > task_config (任务默认)
                request_params = {
                    "temperature": self.task_config.temperature,
                    "max_tokens": self.task_config.max_tokens,
                    **kwargs
                }

                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在尝试使用模型: {model_config.name} ({model_config.model_identifier})")
                self.model_states[model_config.name] = (self.model_states[model_config.name][0], time.time())

                response = await client.get_response(
                    model_config=model_config,
                    messages=messages,
                    tools=tools,
                    **request_params
                )
                
                # 请求成功，重置该模型的失败次数
                self.model_states[model_config.name] = (0, self.model_states[model_config.name][1])
                print(f"模型 '{model_config.name}' 请求成功。")
                return response, model_config.name

            except (APIError, EmptyResponseError, LLMException) as e:
                # 这些是我们定义的“硬”错误，应该切换到下一个模型
                last_exception = e
                failure_count, last_used = self.model_states[model_config.name]
                self.model_states[model_config.name] = (failure_count + 1, last_used)
                failed_models.add(model_config.name)
                print(f"模型 '{model_config.name}' 尝试失败: {e}")
                
                if isinstance(e, APIError) and 400 <= e.status_code < 500:
                    print(f"警告: 收到客户端错误 (状态码 {e.status_code})，这通常表示请求内容有问题。将尝试下一个模型。")
            except Exception as e:
                # 捕获其他意料之外的错误
                last_exception = e
                failure_count, last_used = self.model_states[model_config.name]
                self.model_states[model_config.name] = (failure_count + 1, last_used)
                failed_models.add(model_config.name)
                print(f"模型 '{model_config.name}' 遇到未知错误，尝试失败: {e}")

        # 所有模型尝试失败后
        error_message = f"任务 '{self.task_name}' 的所有可用模型均已尝试失败。"
        print(error_message)
        if last_exception:
            raise LLMException(error_message) from last_exception
        raise LLMException(error_message)
