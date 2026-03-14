# src/agent/agent_loop.py
import asyncio
import time
from typing import Optional, List

from src.common.logger import get_logger
from src.common.di.container import container
from src.common.action_model.action_spec import ActionSpec
from src.common.action_model.tool_result import ToolResult

from src.agent.motive.motive_engine import MotiveEngine
from src.agent.planner.main_planner import MainPlanner
from src.agent.planner.planner_result import PlanResult
from src.agent.world_model import WorldModel
from src.cortices.manager import CortexManager
from src.llm_api.factory import LLMRequestFactory

logger = get_logger("agent_loop")

class AgentLoop:
    """
    Agent 的主循环，负责驱动 Agent 的整个生命周期。
    它协调 MotiveEngine, MainPlanner, WorldModel 等核心组件，
    实现“感知-思考-行动”的闭环。
    """
    agent_interrupt_event = asyncio.Event()

    def __init__(self):
        logger.info("初始化智能体循环...")
        self.motive_engine = MotiveEngine()
        self.main_planner = MainPlanner()
        self.cortex_manager: CortexManager = container.resolve(CortexManager)
        self.llm_factory: LLMRequestFactory = container.resolve(LLMRequestFactory)
        self.world_model: WorldModel = container.resolve(WorldModel)
        
        self._is_running = False
        self._main_task: Optional[asyncio.Task] = None
        self.heartbeat_interval = 5

    async def _execute_motive_plan(self, motive: str):
        """
        在一个给定的动机下，执行一次完整的“规划 -> 行动链 -> 观察”流程。
        """
        # 1. 规划 (Plan) - 获得初始行动计划
        plan_result: PlanResult = await self.main_planner.plan(motive)
        if not plan_result:
            logger.warning("主规划器未能生成有效的计划。")
            return

        # 如果计划是“完成/无事可做”，则提前结束
        if plan_result.action.tool_name == "finish":
            logger.info(f"规划结束，原因：{plan_result.reason}")
            # 也可以将此思考过程记入记忆
            self.world_model.add_memory(f"我思考了一下，决定暂时什么都不做，因为：{plan_result.reason}")
            await asyncio.sleep(self.heartbeat_interval * 2) # 适当延长等待时间
            return

        # 2. 执行行动链 (Action Chain)
        action_queue: List[ActionSpec] = [plan_result.action]
        chain_context: str = f"我的初始想法是：{plan_result.reason}。"
        full_results_for_memory: List[ToolResult] = []
        
        chain_step = 0
        max_chain_steps = 10 # 防止无限调用链

        while action_queue and chain_step < max_chain_steps:
            # 最新行动出栈
            current_action = action_queue.pop(0)
            chain_step += 1
            logger.info(f"行动链 [步骤 {chain_step}]: 执行工具 '{current_action.tool_name}'")

            # 准备参数，并注入临时的链上下文
            params = current_action.parameters or {}
            # params["chain_context"] = chain_context

            # 执行工具
            tool_result: ToolResult = await self.cortex_manager.call_tool_by_name(
                current_action.tool_name, **params
            )

            # 收集结果用于最终记忆
            full_results_for_memory.append(tool_result)

            # # 更新临时的链上下文
            # if tool_result and tool_result.summary:
            #     chain_context += f"\n上一步({next_action.tool_name})的结果是：'{tool_result.summary}'。"
            
            # 更新消息缓冲队列
            if tool_result and tool_result.follow_up_action:
                action_queue.extend(tool_result.follow_up_action)
        
        if chain_step >= max_chain_steps:
            logger.warning("行动链达到最大步骤限制，已强制终止。")

        # 3. 记忆 (Memorize) - 在整个行动链结束后，进行一次总记忆
        await self._record_chain_memory(motive, plan_result, full_results_for_memory)

    async def _record_chain_memory(self, motive: str, initial_plan: PlanResult, results: List[ToolResult]):
        """根据一个完整的行动链，生成并记录一条高质量的记忆。"""
        if not results:
            return

        llm_request = self.llm_factory.get_request("planner")
        
        # 构建行动链的日志
        action_log = ""
        for i, tool_result in enumerate(results):
            action_log += f"步骤 {i+1}: {tool_result.summary}\n"

        prompt = f"""
你是一个记忆总结器。请将一段完整的“动机-规划-行动链”日志，总结为一段流畅的、第一人称的个人日记。

## 原始日志
- **我的初始动机**: "{motive}"
- **我的初步想法**: "{initial_plan.reason}"
- **我的行动过程**:
{action_log}

## 总结要求
1. **第一人称**: 必须以“我...”开头。
2. **因果连贯**: 清晰地表达出“因为我想...所以我先做了...然后...最后...，最后说明动机是否已被满足”。
3. **概括性**: 无需拘泥于每一步的细节，抓住核心结果。
4. **自然口语**: 像在写自己的日记，而不是一份工作报告。
5. **字数**: 80字左右。
"""
        final_memory, _ = await llm_request.execute(prompt)
        
        if final_memory:
            self.world_model.add_memory(final_memory.strip())

    async def _run_once(self):
        """执行一次完整的“感知-动机-规划-行动”循环。"""
        try:
            # 1. 感知 (Perceive) & 动机 (Motive)
            capability_descriptions = self.cortex_manager.get_collected_capability_descriptions()
            motive = await self.motive_engine.generate_motive(capability_descriptions)
            
            if not motive:
                logger.info("未能生成明确动机，跳过本次循环。")
                return

            self.world_model.motive = motive

            # 2. 执行动机 (这现在会驱动整个行动链)
            await self._execute_motive_plan(motive)
            
            logger.info(f"动机 '{motive}' 对应的行动链已执行完毕。")

        except asyncio.CancelledError:
            logger.warning("当前运行循环任务被中断信号取消。")
            raise
        except Exception as e:
            logger.error(f"在执行循环时发生错误: {e}", exc_info=True)

    async def _run(self):
        """主循环的内部实现"""
        self._is_running = True
        logger.info(f"主循环已启动，心跳间隔: {self.heartbeat_interval} 秒。")
        while self._is_running:
            try:
                await self._run_once()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                logger.info("主循环任务被外部取消。")
                break
            except Exception as e:
                logger.error(f"主循环中发生未捕获错误: {e}", exc_info=True)
                await asyncio.sleep(self.heartbeat_interval)
    
    def start(self):
        """启动 Agent 的主循环"""
        if not self._is_running:
            self._main_task = asyncio.create_task(self._run())
        else:
            logger.warning("Agent 已经在运行中。")

    def stop(self):
        """停止 Agent 的主循环"""
        if self._is_running and self._main_task:
            self._is_running = False
            self._main_task.cancel()
        else:
            logger.warning("Agent 未在运行中。")
