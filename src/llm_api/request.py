# llm_api/request.py
import asyncio
import time
from typing import List, Dict, Optional, Any, Tuple, Set

from .client import BaseClient, APIResponse, get_client_for_model

class LLMRequest:
    """
    LLM请求类，负责模型选择、客户端复用、故障切换和请求重试。
    """
    def __init__(self, model_configs: List[Dict[str, Any]]):
        if not model_configs:
            raise ValueError("模型配置列表不能为空")
        self.model_configs = {conf["model_name"]: conf for conf in model_configs}

        # 模型使用状态，用于负载均衡和故障切换
        # (失败次数, 上次使用时间)
        self.model_states: Dict[str, Tuple[int, float]] = {
            name: (0, 0.0) for name in self.model_configs.keys()
        }

        # 缓存客户端实例，避免每次都重新连接
        self._clients: Dict[str, BaseClient] = {}

    def _select_model(self, exclude_models: Set[str]) -> Optional[Dict[str, Any]]:
        """
        根据失败次数和最近使用情况选择一个模型。
        - 优先选择失败次数最少的。
        - 如果失败次数相同，选择最久未被使用的。
        """
        available_models = {
            name: state
            for name, state in self.model_states.items()
            if name not in exclude_models
        }
        if not available_models:
            return None

        sorted_models = sorted(
            available_models.items(),
            key=lambda item: (item[1][0], item[1][1])
        )

        best_model_name = sorted_models[0][0]

        # 更新选中模型的“上次使用时间”
        self.model_states[best_model_name] = (
            self.model_states[best_model_name][0],
            time.time()
        )

        return self.model_configs[best_model_name]

    def _get_client(self, model_config: Dict[str, Any]) -> BaseClient:
        """
        获取客户端实例，复用已有客户端。
        """
        model_name = model_config["model_name"]
        if model_name not in self._clients:
            client = get_client_for_model(model_config)
            self._clients[model_name] = client
        return self._clients[model_name]

    async def generate_response_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        异步生成响应，包含模型选择、客户端复用、重试和故障切换逻辑。
        """
        failed_models: Set[str] = set()
        max_attempts = len(self.model_configs)
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(max_attempts):
            selected_config = self._select_model(exclude_models=failed_models)
            if not selected_config:
                break

            model_name = selected_config["model_name"]
            llm_model_name = selected_config["llm_model_name"]

            client = self._get_client(selected_config)

            try:
                # 调用客户端获取响应
                response: APIResponse = await client.get_response(
                    llm_model_name=llm_model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.content, model_name

            except Exception as e:
                # 增加失败次数，记录失败模型
                failure_count, last_used = self.model_states[model_name]
                self.model_states[model_name] = (failure_count + 1, last_used)
                failed_models.add(model_name)
                print(f"模型 '{model_name}' 尝试失败: {e}")

        # 所有模型尝试失败
        print("所有模型均尝试失败。")
        return None, None
