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
        """
        初始化动机引擎。
        """
        logger.info("MotiveEngine: 初始化...")
        
        llm_factory = container.resolve(LLMRequestFactory)
        self.llm_request = llm_factory.get_request(task_name="motive")
        
    def _format_impetus_descriptions(self, collected_impetus_descriptions: List[str]) -> str:
        """
        将收集到的内在驱动力描述格式化为字符串，供提示使用。
        """
        # 格式化内在驱动力描述为字符串
        if collected_impetus_descriptions:
            formatted_impetus = [f"- {desc}" for desc in collected_impetus_descriptions]
            impetus = "\n".join(formatted_impetus)
            return impetus
        else:
            impetus = "无可用内在驱动力。"
            logger.warning("未收集到任何内在驱动力描述。")
            return impetus
            


    def _build_prompt(self, world_model: WorldModel, impetus: str) -> str:
        """
        根据 WorldModel 和 prompt_design.md 的设计构建 LLM 提示。
        """
        context = world_model.get_context_for_motive()
        notification = context['notification']
        if context['notification'] == []:
            notification = "当前没有新的未读消息。"

        prompt_template = f"""
你叫 {context['bot_name']}。
你是 {context['bot_identity']}。
你的性格是 {context['bot_personality']}，你的兴趣包括 {context['bot_interst']}。

现在是 {context['time']}。
此刻你的心理状态是：{context['mood']}。

{notification}
{context['alert']}

你最近一次活动的总结如下：
{context['action_summary']}

你目前拥有以下**可被激发的内在驱动力**：
{impetus}

你的任务：
基于上述全部信息，结合你的性格、情绪、兴趣和刚刚经历的事件，为自己生成一个最自然、最真实的人类式内在动机。
以简洁、情绪化、带人类思考色彩的一句话表达你此刻最想做的事情。
语言风格更像你的内心自言自语，而不是程序化的指令。

你的动机可以是什么也不想干，选择休息或发呆也是可以的。

只输出一句自然的“内在意图表达”。
"""
        return prompt_template

    async def generate_motive(self, collected_impetus_descriptions: List[str]) -> str:
        """
        生成一个高阶、模糊的动机。
        """
        impetus =self._format_impetus_descriptions(collected_impetus_descriptions)
        world_model = container.resolve(WorldModel)
        logger.info("正在生成动机...")
        prompt_str = self._build_prompt(world_model,impetus)

        try:
            response:str
            response, model_name = await self.llm_request.execute(
                prompt=prompt_str
            )
            
            if response:
                motive = response.strip().replace("\"", "").replace("'", "")
                logger.info(f"新的动机: {motive}")
                world_model.motive = motive
                return motive
            else:
                """ LLM 未返回内容，返回一个默认的休息动机 """
                logger.warning("新的动机: LLM 未返回明确动机，休息一下。")
                world_model.motive = "休息一下"
                return "休息一下"

        except Exception as e:
            """ LLM 请求失败，返回一个处理错误的动机 """
            logger.error(f"生成动机时 LLM 请求失败: {e}")
            world_model.motive = "处理内部错误"
            return "处理内部错误"



# 例如：
# “QQ 那边的对话把我搞得有点烦，我想刷刷微博缓一缓。”
# “已经这么晚了，我应该读会书，让自己静下来。”
# “微博看到有新留言，我有点好奇，想进去看看发生了什么。”