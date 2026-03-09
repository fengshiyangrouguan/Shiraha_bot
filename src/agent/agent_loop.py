# src/agent/agent_loop.py
import asyncio
import time
from typing import Optional,List

from src.common.logger import get_logger
from src.agent.motive.motive_engine import MotiveEngine
from src.agent.planner.main_planner import MainPlanner
from src.agent.world_model import WorldModel
from src.cortices.manager import CortexManager
from src.agent.planner.planner_result import PlanResult
from src.llm_api.factory import LLMRequestFactory
from src.system.di.container import container
from src.llm_api.dto import ToolCall

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
        self.llm_factory = container.resolve(LLMRequestFactory)
        
        self.world_model:WorldModel = container.resolve(WorldModel)
        self._is_running = False
        self._main_task: Optional[asyncio.Task] = None
        self.heartbeat_interval = 5
        

    async def _execute_motive_plan(self, motive: str):
        """
        在一个给定的动机下，执行一次 Thought-Action-Observation。
        """
        
        # 1. 规划 (Plan) - 接收动机，不再需要 previous_observation
        # 在新的模式下，上一步的观察结果会通过 WorldModel 的整体上下文提供给 Planner
        plan_result: PlanResult = await self.main_planner.plan(motive)

        if plan_result.tool_name == "finish":
            tool_result = "你发呆了一会儿。"
            logger.info(f"{tool_result}")
        else:
            tool_call = ToolCall(tool_name=plan_result.tool_name, parameters=plan_result.parameters)
            tool_result = await self.cortex_manager.execute_tool(tool_call)
            logger.info(f"{tool_result}")

        # 2. 记忆 (Memorize) - 将此步骤的结果记录到世界模型
        await self._record_step_memory(motive, plan_result.thought, tool_result)

    async def _record_step_memory(self, motive: str, plan: str, tool_result: str):
        """记录单步的记忆"""
        llm_request = self.llm_factory.get_request(task_name="utils_small")
        prompt = (
            f"""
你是一个记忆总结器，

下面是一段动机-规划-行动过程原文：
你的动机：{motive}
对动机的规划想法：{plan}
行动结果：{tool_result}

请你以第一人称视角，对上面的记忆，进行50字左右的总结，
对动机，规划尽量简短，对行动结果偏详细
要求保留原文中的行文、说话风格
语气要自然，总结即可，不要出现“这让我意识到”、“综上所述”等生硬词汇。不需要上升太多思想高度，就像是一段简单的行动记忆,不要使用换行
"""
        )
        response, model_name = await llm_request.execute(prompt)

        memory_entry = (
            f"{response}"
        )
        self.world_model.add_memory(memory_entry)

    async def _run_once(self):
        """
        执行一次完整的“感知-动机-规划-行动”循环。
        """

        try:
            # 1. 感知 (Perceive) & 动机 (Motive)
            capability_descriptions = self.cortex_manager.get_collected_capability_descriptions()
            motive = await self.motive_engine.generate_motive(capability_descriptions)
            
            if not motive:
                logger.info(f"未能生成明确动机，跳过本次循环。")
                return

            self.world_model.motive = motive

            # 2. 执行动机 (Execute Motive)
            await self._execute_motive_plan(motive)
            
            logger.info(f"动机 '{motive}' 执行完毕")

        except asyncio.CancelledError:
            logger.warning("当前运行循环任务被中断信号取消。")
            raise

        except Exception as e:
            logger.error(f"在执行循环时发生错误: {e}", exc_info=True)
            import traceback
            traceback.print_exc()

    async def _run(self):
        """主循环的内部实现，包含中断监控"""
        self._is_running = True
        logger.info(f"主循环已启动，心跳间隔: {self.heartbeat_interval} 秒。")
        while self._is_running:
            current_cycle_task = None
            try:
                current_cycle_task = asyncio.create_task(self._run_once())
                interrupt_task = asyncio.create_task(AgentLoop.agent_interrupt_event.wait())

                done, pending = await asyncio.wait(
                    {current_cycle_task, interrupt_task},
                    return_when=asyncio.FIRST_COMPLETED
                )

                if interrupt_task in done:
                    logger.info("检测到中断信号，取消当前运行循环任务。")
                    current_cycle_task.cancel()
                    await asyncio.gather(current_cycle_task, return_exceptions=True)
                    AgentLoop.agent_interrupt_event.clear()
                    continue 

                await current_cycle_task
                await asyncio.sleep(self.heartbeat_interval)
            
            except asyncio.CancelledError:
                logger.info("主循环任务被外部取消。")
                if current_cycle_task and not current_cycle_task.done():
                    current_cycle_task.cancel()
                    await asyncio.gather(current_cycle_task, return_exceptions=True)
                break
            except Exception as e:
                logger.error(f"主循环中发生未捕获错误: {e}", exc_info=True)
                import traceback
                traceback.print_exc()
                await asyncio.sleep(self.heartbeat_interval)

    def start(self):
        """启动 Agent 的主循环"""
        if not self._is_running:
            AgentLoop.agent_interrupt_event.clear()
            self._main_task = asyncio.create_task(self._run())
            logger.info("主循环开始尝试运行。")
        else:
            logger.warning("Agent 已经在运行中。")

    def stop(self):
        """停止 Agent 的主循环"""
        if self._is_running and self._main_task:
            self._is_running = False
            self._main_task.cancel()
            logger.info("主循环停止请求已发送。")
        else:
            logger.warning("Agent 未在运行中。")
