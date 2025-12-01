import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.system.di.container import container
from src.llm_api.factory import LLMRequestFactory
from src.agent.world_model import WorldModel
from src.common.logger import get_logger
from .planner_result import PlanResult

logger = get_logger("Planner")

class BasePlanner(ABC):
    """
    所有规划器的抽象基类。
    它封装了与 LLM 交互的通用逻辑，子类只需专注于构建自己的 Prompt。
    """
    def __init__(self, task_name: str):
        """
        初始化规划器，并从 DI 容器中获取其所需的 LLMRequest 实例。
        
        Args:
            task_name (str): 与 config.toml 中 [model_task_config] 对应的任务名称。
        """
        llm_factory = container.resolve(LLMRequestFactory)
        self.world_model: WorldModel = container.resolve(WorldModel)
        self.llm_request = llm_factory.get_request(task_name)
        print(f"Planner已初始化，使用任务配置 '{task_name}'。")

    @abstractmethod
    def _build_prompt(self, world_model:WorldModel) -> List[Dict[str, Any]]:
        """
        构建特定于此规划器的 LLM 提示。
        每个子类都必须实现此方法，以定义自己的思考方式。
        """
        raise NotImplementedError


    async def plan(self, world_model: WorldModel) -> Optional[PlanResult]:
        """
        执行一次完整的规划流程：构建提示 -> 调用 LLM -> 解析与校验 -> 返回标准化 PlanResult。
        """

        prompt_messages = self._build_prompt(world_model)

        try:
            # 发送给 LLM 的是序列化后的 prompt
            content, model_name = await self.llm_request.execute(
                prompt=json.dumps(prompt_messages)
            )
            
            if not content:
                logger.error(f"未返回内容")
                return None

            logger.info(f"LLM原始内容: {content}")

            # 尝试解析 JSON
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"LLM返回内容不是合法JSON。")
                return None

            # ----------- 校验 JSON 结构 ---------------
            required_top_fields = ["thought", "action"]
            missing_top = [k for k in required_top_fields if k not in parsed]

            if missing_top:
                logger.error(f"JSON缺少必要字段: {missing_top}")
                logger.error(
                    f"返回的JSON必须类似:\n"
                    '{\n'
                    '  "thought": "....",\n'
                    '  "action": {\n'
                    '    "tool_name": "xxx",\n'
                    '    "parameters": { ... }\n'
                    '  }\n'
                    '}'
                )
                return None

            # 校验 action 内部结构
            required_action_fields = ["tool_name", "parameters"]
            missing_action = [
                k for k in required_action_fields if k not in parsed["action"]
            ]

            if missing_action:
                logger.error(
                    f"action 字段缺少必要字段: {missing_action}"
                )
                logger.error(
                    f"action 必须类似:\n"
                    '  "action": {\n'
                    '    "tool_name": "xxx",\n'
                    '    "parameters": { ... }\n'
                    '  }'
                )
                return None

            # ----------- 字段类型检查 ---------------
            if not isinstance(parsed["thought"], str):
                logger.error(f"字段 thought 必须是字符串。收到: {type(parsed['thought'])}")
                return None

            if not isinstance(parsed["action"]["tool_name"], str):
                logger.error(f"字段 tool_name 必须是字符串。收到: {type(parsed['tool_name'])}")
                return None

            if not isinstance(parsed["action"]["parameters"], dict):
                logger.error(f"字段 parameters 必须是一个 dict。收到: {type(parsed['parameters'])}")
                return None

            result = PlanResult(
                thought=parsed["thought"],
                tool_name=parsed["action"]["tool_name"],
                parameters=parsed["action"]["parameters"]

            )

            # ----------- 打印成功日志 ---------------
            logger.info("规划生成成功：")
            logger.info(f"  - 思考过程: {result.thought}")
            logger.info(f"  - 计划行动: 调用工具 '{result.tool_name}' 参数: {result.parameters}")

            return result

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return None




