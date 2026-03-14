# src/plugins/rpg_plugin/sub_agent/planner.py
import json
import re
from typing import List, Dict, Any, Optional
from src.common.logger import get_logger
from src.common.di.container import container
from src.llm_api.factory import LLMRequestFactory

logger = get_logger("rpg_sub_planner")

class SubPlanner:
    """
    RPG 子规划器：负责在跑团子流程中决定下一步该做什么。
    """
    def __init__(self):
        self.llm_factory = container.resolve(LLMRequestFactory)
        # 使用更强大的模型进行逻辑规划
        self.planner_llm = self.llm_factory.get_request("planner")

    def _build_system_prompt(self, phase: str, world_lore: str) -> str:
        return f"""
你是一个专业的 TRPG 游戏主持人 (DM/GM) 的逻辑规划模块。
你的任务是根据当前游戏阶段和聊天上下文，规划出下一步的最优行动方案。

## 当前阶段: {phase}
## 世界观背景:
{world_lore}

## 输出规范:
你必须返回一个标准的 JSON 对象，包含以下字段：
1. "thought": 简短描述你为什么要进行这个规划（你的思考过程）。
2. "action": 具体的动作类型，可选：
   - "speak": 需要对玩家说话或回应。
   - "update_lore": 发现了新的设定，需要更新世界观。
   - "next_phase": 判定当前阶段已完成，准备进入下一阶段。
   - "wait": 暂时不采取行动，等待玩家更多输入。
3. "intent": 当 action 为 "speak" 时，描述你说话的具体意图和内容要点。
4. "parameters": 动作所需的参数（如 {{ "target_phase": "CHARACTER_CREATION" }}）。

## 约束:
- 保持角色沉浸感。
- 如果玩家正在闲聊，请规划 "speak" 动作进行引导。
- 严禁输出 JSON 以外的内容。
"""

    async def plan(self, phase: str, history: str, lore_context: str, registered_players: Dict) -> Dict[str, Any]:
        """
        生成规划决策。
        """
        system_prompt = self._build_system_prompt(phase, lore_context)
        
        user_input = f"""
## 玩家名单:
{json.dumps(registered_players, ensure_ascii=False)}

## 最近聊天记录:
{history}

请根据以上信息，规划下一步行动。
"""
        try:
            content, _ = await self.planner_llm.execute(
                prompt=user_input,
                system_prompt=system_prompt
            )
            
            # 解析 JSON
            plan_json = self._parse_json(content)
            logger.info(f"子规划器决策结果: [{plan_json.get('action')}] - {plan_json.get('thought')}")
            return plan_json

        except Exception as e:
            logger.error(f"子规划器运行异常: {e}")
            return {
                "thought": "由于内部错误，被迫维持现状",
                "action": "wait",
                "intent": ""
            }

    def _parse_json(self, content: str) -> Dict[str, Any]:
        """安全解析 LLM 返回的 JSON"""
        try:
            # 寻找首个 { 和最后一个 } 之间的内容
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(content)
        except:
            return {"action": "wait", "thought": "解析 JSON 失败"}