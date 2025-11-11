# core/replyer.py
import logging
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

# 从我们之前创建的模块中导入
from ..config import LLM_MODELS
from ..llm_api.request import LLMRequest
from .chat import ChatStream
from .prompts import REPLYER_PROMPT  # 导入新的Prompt模板

logger = logging.getLogger("replyer")

class Replyer:
    """
    回复生成器，使用模板动态构建Prompt并调用LLM。
    """
    def __init__(self):
        self.llm_request = LLMRequest(model_configs=LLM_MODELS)
        # self.bot_name = "小白羽" 
        # self.personality = "你是一个活泼开朗、略带傲娇的少女AI助手，喜欢用颜文字和表情符号，对世界充满好奇，总是乐于助人但偶尔也会开点小玩笑。"
        # self.reply_style = "你的回复要简短、有趣，充满活力。请在回复时使用颜文字和表情符号，例如：(๑•̀ㅂ•́)و✧、(≧∇≦)ﾉ、(｡･ω･｡)ﾉ♡。语气要轻松愉快，不要过于正式。"
        # self.moderation_prompt = "请不要输出违法违规内容，不要输出色情，暴力，政治相关内容。"
        self.bot_name = "小智"
        self.personality = "你是一个聪明可爱、活泼开朗的少女AI助手，说话温柔严谨，有礼貌，会积极回答所有问题和求助，对世界充满好奇。"
        self.reply_style = "你的回复严谨，略显冷淡，但其实内心十分热情，不要过于正式。"
        self.moderation_prompt = "请不要输出违法违规内容，不要输出色情，暴力，政治相关内容。"

    async def generate_reply(
        self, 
        chat_stream: ChatStream,
        current_message: str,
        tool_results = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        为给定的聊天流生成回复。

        Args:
            chat_stream (ChatStream): 当前的聊天流对象。
            current_message (str): 用户发送的最新消息。
            tool_results (Optional[Any]): 工具调用的结果，供回复生成时参考。

        Returns:
            Tuple[Optional[str], Optional[str]]: (生成的回复内容, 使用的模型名称)
        """
        # 1. 准备Prompt模板中需要的各个部分
        history_prompt = chat_stream.get_history_prompt()
        identity_prompt = f"你的名字是{self.bot_name}，{self.personality}"

        # 2. 创建一个字典来存放所有占位符的内容
        context: Dict[str, Any] = {
            "knowledge_prompt": "",
            "tool_info_block": "",
            "extra_info_block": f"{tool_results}",
            "expression_habits_block": "",
            "memory_block": "",
            "question_block": "",
            "mood_state": "",
            "keywords_reaction_prompt": "",
            
            # 当前已实现的模块
            "time_block": f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "dialogue_prompt": history_prompt,
            "reply_target_block": f"现在用户说的“{current_message}”引起了你的注意。",
            "identity": identity_prompt,
            "reply_style": self.reply_style,
            "moderation_prompt": self.moderation_prompt,
        }

        # 3. 使用模板和上下文构建最终的Prompt
        full_prompt = REPLYER_PROMPT.format(**context)
        
        logger.info("="*20)
        logger.info("正在构建完整的Prompt:")
        logger.info(full_prompt)
        logger.info("="*20)

        # 4. 调用LLMRequest来获取回复
        reply_content, model_used = await self.llm_request.generate_response_async(full_prompt)

        # 引导LLM以机器人名称开头回复后，可能需要在实际输出时去掉这个前缀
        if reply_content and reply_content.startswith(f"{self.bot_name}:"):
            reply_content = reply_content[len(self.bot_name)+1:].lstrip()

        return reply_content, model_used
