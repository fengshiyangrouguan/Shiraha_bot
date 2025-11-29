# src/agent/motive/motive_engine.py
import json
from typing import List, Dict, Any, Optional

from src.common.logger import get_logger
from src.llm_api.factory import LLMRequestFactory
from src.agent.world_model import WorldModel
from src.system.di.container import container

logger = get_logger("MotiveEngine")



class MotiveEngine:
    """
    动机引擎。
    负责根据世界状态和 Agent 的核心身份，生成高阶意图。
    """
    def __init__(self):
        logger.info("MotiveEngine: 初始化 (使用 DI 获取依赖)...")
        
        # 从 DI 容器中解析出 LLMRequestFactory 的单例
        llm_factory = container.resolve(LLMRequestFactory)
        # 通过工厂获取专门用于“动机生成”任务的 LLMRequest 实例
        self.llm_request = llm_factory.get_request(task_name="motive_generation")
        
        # TODO: 从 FeatureManager 获取
        self.features_summary = "无法进行任何操作"

    def _build_prompt(self, world_model: WorldModel) -> List[Dict[str, Any]]:
        """
        根据 WorldModel 和 prompt_design.md 的设计构建 LLM 提示。
        """
        # 1. 从 WorldModel 获取上下文数据
        context = world_model.get_context_for_motive()

        # 2. 构造Prompt
        prompt_template = f"""
你叫 {context['bot_name']}。
你是 {context['bot_identity']}。
你的性格是 {context['bot_personality']}，你的兴趣包括 {context['bot_interst']}。

现在是 {context['time']}。
此刻你的心理状态是：{context['mood']}。

{context['notification']}
{context['alert']}

你最近一次活动的总结如下：
{context['action_summary']}

你目前拥有以下可以主动采取的行动：
{self.features_summary}

你的任务：
基于上述全部信息，结合你的性格、情绪、兴趣和刚刚经历的事件，为自己生成一个最自然、最真实的人类式内在动机。
以简洁、情绪化、带人类思考色彩的一句话表达你此刻最想做的事情。
语言风格更像你的内心自言自语，而不是程序化的指令。

例如：
“QQ 那边的对话把我搞得有点烦，我想刷刷微博缓一缓。”
“已经这么晚了，我应该读会书，让自己静下来。”
“微博看到有新留言，我有点好奇，想进去看看发生了什么。”

只输出一句自然的“内在意图表达”。
"""

        return prompt_template


    async def generate_motive(self, world_model: WorldModel) -> str:
        """
        生成一个高阶、模糊的动机。
        """
        logger.info("正在生成动机...")
        prompt_messages = self._build_prompt(world_model)

        try:
            response:str
            response, model_name = await self.llm_request.execute(
                messages=prompt_messages
            )
            
            if response:
                intent = response.strip().replace("\"", "").replace("'", "")
                logger.info(f"新的动机: {intent}")
                return intent
            else:
                """ LLM 未返回内容，返回一个默认的休息动机 """
                logger.warning("新的动机: LLM 未返回明确动机，休息一下。")
                return "休息一下"

        except Exception as e:
            """ LLM 请求失败，返回一个处理错误的动机 """
            logger.error(f"生成动机时 LLM 请求失败: {e}")
            return "处理内部错误" 
