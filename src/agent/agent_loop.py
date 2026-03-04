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
from src.system.di.container import container
from src.llm_api.dto import ToolCall

logger = get_logger("AgentLoop")

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
        
        self.world_model:WorldModel = container.resolve(WorldModel)
        self._is_running = False
        self._main_task: Optional[asyncio.Task] = None
        self.heartbeat_interval = 30 
        

    async def _execute_motive_plan(self, motive: str):
        """
        在一个给定的动机下，执行一次 Thought-Action-Observation。
        """
        logger.info(f"  - 执行单步规划，动机: '{motive}'")
        
        # 1. 规划 (Plan) - 接收动机，不再需要 previous_observation
        # 在新的模式下，上一步的观察结果会通过 WorldModel 的整体上下文提供给 Planner
        plan_result: PlanResult = await self.main_planner.plan(motive)

        if plan_result.tool_name == "finish":
            logger.info(f"  - 规划器选择不采取行动 (finish)。")
            plan_result.thought = "感觉没什么要干的。"
            tool_result = "你发呆了一会儿。"
        else:
            logger.info(f"  - 思考过程: {plan_result.thought}")
            logger.info(f"  - 计划行动: 调用工具 '{plan_result.tool_name}' 参数: {plan_result.parameters}")
            
            tool_call = ToolCall(tool_name=plan_result.tool_name, parameters=plan_result.parameters)

            tool_result = await self.cortex_manager.execute_tool(tool_call)
            logger.info(f"  - 工具执行结果: {tool_result}")

        # 2. 记忆 (Memorize) - 将此步骤的结果记录到世界模型
        self._record_step_memory(motive, plan_result, tool_result)

    def _record_step_memory(self, motive: str, plan: PlanResult, tool_result: str):
        """记录单步的记忆"""
        memory_entry = (
            f"{tool_result}"
        )
        self.world_model.add_memory(memory_entry)
        logger.info("  - 单步记忆已更新。")

    async def _run_once(self):
        """
        执行一次完整的“感知-动机-规划-行动”循环。
        """
        logger.debug(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AgentLoop 心跳 ---")

        try:
            # 1. 感知 (Perceive) & 动机 (Motive)
            capability_descriptions = self.cortex_manager.get_collected_capability_descriptions()
            motive = await self.motive_engine.generate_motive(capability_descriptions)
            
            if not motive or "无" in motive:
                logger.info(f"  - 结果: 未能生成明确动机，跳过本次循环。")
                return

            logger.info(f"  - 新的动机: {motive}")
            self.world_model.motive = motive
            # 2. 执行动机 (Execute Motive)
            #    此方法内部包含完整的 Plan-Act-Observe 循环
            await self._execute_motive_plan(motive)
            
            logger.info(f"--- 动机 '{motive}' 执行完毕 ---")

        except asyncio.CancelledError:
            logger.warning("AgentLoop: 当前运行循环任务被中断信号取消。")
            raise

        except Exception as e:
            logger.error(f"AgentLoop 在执行循环时发生错误: {e}", exc_info=True)
            import traceback
            traceback.print_exc()

    async def _run(self):
        """主循环的内部实现，包含中断监控"""
        self._is_running = True
        logger.info(f"Agent 主循环已启动，心跳间隔: {self.heartbeat_interval} 秒。")
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
                    logger.info("AgentLoop: 检测到中断信号！取消当前运行循环任务。")
                    current_cycle_task.cancel()
                    await asyncio.gather(current_cycle_task, return_exceptions=True)
                    AgentLoop.agent_interrupt_event.clear()
                    continue 

                await current_cycle_task
                await asyncio.sleep(self.heartbeat_interval)
            
            except asyncio.CancelledError:
                logger.info("AgentLoop: 主循环任务被外部取消。")
                if current_cycle_task and not current_cycle_task.done():
                    current_cycle_task.cancel()
                    await asyncio.gather(current_cycle_task, return_exceptions=True)
                break
            except Exception as e:
                logger.error(f"AgentLoop:   主循环中发生未捕获错误: {e}", exc_info=True)
                import traceback
                traceback.print_exc()
                await asyncio.sleep(self.heartaint_interval)

    def start(self):
        """启动 Agent 的主循环"""
        if not self._is_running:
            AgentLoop.agent_interrupt_event.clear()
            self._main_task = asyncio.create_task(self._run())
            logger.info("AgentLoop: 主循环开始尝试运行。")
        else:
            logger.warning("Agent 已经在运行中。")

    def stop(self):
        """停止 Agent 的主循环"""
        if self._is_running and self._main_task:
            self._is_running = False
            self._main_task.cancel()
            logger.info("AgentLoop: 主循环停止请求已发送。")
        else:
            logger.warning("Agent 未在运行中。")




# --- 用于演示中断的辅助函数 ---
async def simulate_interrupt(delay: int):
    print(f"模拟器: {delay} 秒后将触发中断信号...")
    await asyncio.sleep(delay)
    AgentLoop.agent_interrupt_event.set()
    print("模拟器: 中断信号已触发！")

# 这是一个简单的示例，展示如何运行 AgentLoop
async def main():
    agent_loop = AgentLoop()
    agent_loop.start()
    
    # 启动一个模拟中断的任务
    interrupt_task = asyncio.create_task(simulate_interrupt(agent_loop.heartbeat_interval * 0.5)) # 在半个心跳周期后中断

    try:
        # 让 Agent 运行一段时间
        await asyncio.sleep(agent_loop.heartbeat_interval * 3.5) # 运行足够长的时间观察效果
    finally:
        agent_loop.stop()
        if interrupt_task and not interrupt_task.done():
            interrupt_task.cancel()
            await asyncio.gather(interrupt_task, return_exceptions=True)
        # 等待所有任务完成取消
        await asyncio.sleep(1) 

if __name__ == "__main__":
    asyncio.run(main())
