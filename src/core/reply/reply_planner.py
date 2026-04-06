import json
from typing import Any, Dict, List

from src.common.di.container import container
from src.common.logger import get_logger
from src.llm_api.factory import LLMRequestFactory

logger = get_logger("reply_planner")


class ReplyPlanner:
    """
    高级回复规划器。

    它不负责决定“是否要回复”，而是负责在回复器已经决定需要回应后，
    基于局部上下文生成更拟人的表达内容与表达方式。
    """

    def __init__(self):
        llm_factory: LLMRequestFactory = container.resolve(LLMRequestFactory)
        self.llm_request = llm_factory.get_request("planner")

    async def plan_reply(
        self,
        conversation_id: str,
        latest_message: str,
        conversation_context: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        prompt = f"""
你是 QQ 回复规划器。
目标会话：{conversation_id}
最新消息：{latest_message}

已有上下文：
{json.dumps(conversation_context[-12:], ensure_ascii=False, indent=2)}

请生成一条自然、拟人、不过度热情的回复，并给出表达方式选择。
只输出 JSON：
{{
  "should_reply": true,
  "reply_content": "要发送的内容",
  "style": "casual|warm|playful|brief",
  "use_emoji": false
}}
"""
        try:
            content, _ = await self.llm_request.execute(prompt=prompt)
            cleaned = content.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            logger.warning(f"高级回复规划失败，回退到简单回复: {exc}")

        return {
            "should_reply": True,
            "reply_content": "我看到了，等我想想怎么说。",
            "style": "brief",
            "use_emoji": False,
        }
