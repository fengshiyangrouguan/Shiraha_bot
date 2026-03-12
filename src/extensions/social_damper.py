import numpy as np
import json
from typing import List, Dict, Any, Tuple

from src.common.logger import get_logger
from src.llm_api.factory import LLMRequestFactory
from src.llm_api.dto import LLMMessageBuilder
from src.agent.world_model import WorldModel

logger = get_logger(__name__)

DAMP_RESULT_TYPE = Dict[str, bool | str]
AFTEREFFECT_EVALUATION_TYPE = Dict[str, str]

class SocialDamper:
    """
    社交阻尼器/调节器。
    在Agent的意图（Motive）和实际行动之间增加一个“社交情商”层，
    用于事前审查意图和事后复盘社交效果。
    """
    def __init__(self, llm_factory: LLMRequestFactory, world_model: WorldModel):
        self.llm_factory:LLMRequestFactory = llm_factory
        self.world_model:WorldModel = world_model
        # 使用小模型进行快速、低成本的社交判断
        self.llm_request = self.llm_factory.get_request("planner")

    async def damp_intent(self, intent: str, chat_history: str) -> DAMP_RESULT_TYPE:
        """
        使用LLM审查意图，并根据社交环境决定是否修正。

        Args:
            intent: Agent的原始意图字符串。
            chat_history: 最近的聊天记录，用于提供上下文。

        Returns:
            一个字典，包含:
            - should_damp (bool): 是否需要修正意图。
            - damped_intent (str): 修正后的意图（如果should_damp为False，则与原意图相同）。
        """
        personality = self.world_model.bot_personality
        system_prompt = f"""
你是一个谨慎、有情商的社交顾问。你的任务是审查一个AI的行动意图，并判断它在当前社交环境下是否不突兀。
AI的人设是：{personality}

审查规则：
1.  只要意图和聊天内容有关、不会引起误解，就无需修正，即便看起来不礼貌也没关系。"should_damp"直接输出False。
2.  如果意图会显得突兀、重复、不合时宜，你需要仿照原意图的风格，长度，根据原意图修正一个符合当前语境的版本，去掉其中和聊天无关的干扰内容。
3.  你的所有输出必须是严格的JSON格式。绝对禁止修改原意图中的任何人名、群名、作品名

判定为 True (需要修正) 的唯一标准：
- 意图完全脱离了对话历史，自说自话，可能会导致冷场（例如：大家在聊游戏，它突然要聊书）。
- 意图复读了之前的对话，没有信息增量。
- 意图禁止含有巧妙转换引导群聊聊天内容的想法

输入：
- 原始意图 (intent)
- 对话历史 (chat_history)

输出JSON格式：
{{
    "should_damp": 布尔值,
    "damped_intent": "修正后的意图文本，如果无需修正，请原样保留原始意图；"
}}
"""
        
        user_prompt = f"""
请审查以下意图：

**原始意图**: "{intent}"

**最近对话历史**:
{chat_history}

请根据审查规则，输出JSON：
"""
        try:
            builder = LLMMessageBuilder()
            builder.add_system_message(system_prompt)
            builder.add_user_message(user_prompt)
            
            prompt = builder.get_message_dict()        
            response, model_name = await self.llm_request.execute(
                prompt=json.dumps(prompt)
            )

            result_str = response.strip().replace("```json", "").replace("```", "")
            result= json.loads(result_str)
            
            # 验证返回格式
            if "should_damp" in result and "damped_intent" in result:
                return result
            else:
                logger.warning(f"SocialDamper: damp_intent的LLM返回JSON格式不正确: {result}")
                return {"should_damp": False, "damped_intent": intent}

        except Exception as e:
            logger.error(f"SocialDamper: damp_intent在调用LLM时出错: {e}")
            return {"should_damp": False, "damped_intent": intent}


    async def handle_social_aftereffect(self, action_record: Dict[str, Any], chat_history_after_action: List[Dict[str, str]]) -> AFTEREFFECT_EVALUATION_TYPE:
        """
        在行动后，观察并评估社交影响，并将结果存入世界模型。

        Args:
            action_record: 一个字典，记录了本次行动的信息，如 `{"intent": "...", "action": "...", "result": "..."}`.
            chat_history_after_action: 行动发生后的聊天记录，包含了对方的反应。

        Returns:
            一个包含社交评估结果的字典。
        """
        personality = self.world_model.get_personality()
        system_prompt = f"""
你是一位敏锐的社交分析师。你的任务是评估一个AI助手的行动在社交上产生的后果。
AI的人设是：{personality}

分析维度：
1.  **影响 (impact)**: 评估该行动是“积极的 (positive)”、“消极的 (negative)”还是“中性的 (neutral)”。
2.  **对方情绪 (emotion)**: 根据对方的回复，推断对方当前的情绪。
3.  **目标达成 (goal_achieved)**: 判断AI的行动是否达成了其原始意图。

你的所有输出必须是严格的JSON格式。

输出JSON格式：
{{
    "impact": "positive" | "negative" | "neutral",
    "participant_emotion": "对方情绪的简短描述",
    "goal_achieved": "对目标达成情况的简要分析"
}}
"""
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history_after_action])
        user_prompt = f"""
请分析以下这次行动的社交后果：

**AI人设**: {personality}
**原始意图**: {action_record.get('intent', '未知')}
**实际行动**: {action_record.get('action', '未知')}
**行动结果**: {action_record.get('result', '未知')}

**行动后的对话（关键是对方的反应）**:
{history_str}

请根据分析维度，输出JSON：
"""
        try:
            response = await self.llm_request.generate_response_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_output=True
            )
            if isinstance(response, tuple):
                response = response[0]
            
            evaluation = self.llm_request.parse_json_response(response)
            
            # 存入世界模型作为长期记忆
            self.world_model.add_memory(
                "social_aftereffect_evaluation", 
                {"action": action_record, "evaluation": evaluation}
            )
            logger.info(f"SocialDamper: 已将社交后果评估存入世界模型: {evaluation}")

            return evaluation

        except Exception as e:
            logger.error(f"SocialDamper: handle_social_aftereffect在调用LLM时出错: {e}")
            return {
                "impact": "unknown",
                "participant_emotion": "unknown",
                "goal_achieved": f"Failed to evaluate due to error: {e}"
            }
