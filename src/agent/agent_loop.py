# src/agent/agent_loop.py
import asyncio
import time
from typing import Optional

# 导入真正的核心模块
from src.common.logger import get_logger
from src.agent.motive.motive_engine import MotiveEngine
from src.agent.planner.main_planner import MainPlanner
from src.agent.world_model import WorldModel
# from ..features.manager import feature_manager # FeatureManager 尚未完全实现

logger = get_logger("AgentLoop")

class AgentLoop:
    """
    Agent 的主循环，负责驱动 Agent 的整个生命周期。
    它协调 MotiveEngine, MainPlanner, WorldModel 等核心组件，
    实现“感知-思考-行动”的闭环。
    """
    # 定义一个类级别异步事件，作为Agent的中断信号
    agent_interrupt_event = asyncio.Event()

    def __init__(self):
        logger.info("初始化智能体循环...")
        self.motive_engine = MotiveEngine()
        self.main_planner = MainPlanner()
        self.world_model = WorldModel()
        # self.feature_manager = feature_manager
        
        self._is_running = False
        self._main_task: Optional[asyncio.Task] = None
        
        # 心跳间隔（秒），代表 Agent 多久主动“思考”一次
        self.heartbeat_interval = 30 

    async def _run_once(self):
        """
        执行一次完整的“感知-动机-规划”循环。
        这个方法应该能够被取消 (cancellable)。
        """
        logger.debug(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AgentLoop 心跳 ---")

        try:
            # 1. 感知 (Perceive)
            # 从世界模型获取对当前世界的摘要
            logger.info("  - 感知世界中...")
            #world_summary = self.world_model.get_summary() 
            #logger.info(f"  - 世界摘要: {world_summary}")

            # 2. 动机 (Motivate)
            print("  - 生成动机中...")
            motive = await self.motive_engine.generate_motive(world_model=self.world_model)
            
            if not motive or motive == "无明确动机" or motive == "生成动机失败":
                print(f"  - 结果: MotiveEngine 未能生成明确动机: {motive}，跳过本次循环。")
                return

            print(f"  - 动机结果: {motive}")

            # 3. 规划与执行 (Plan & Act)
            # 将意图交给主规划器，执行具体的计划
            print("  - 规划与执行中...")
            plan_result = await self.main_planner.plan(motive,self.world_model)
            
            # 4. 反思与记忆 (Reflect & Memorize)
            # 将行动结果存入世界模型
            print("  - 步骤 4: 存入记忆 (调用 WorldModel)...")
            self.world_model.add_memory(plan_result)
            print(f"  - 结果: '{plan_result}' 已存入记忆。")

        except asyncio.CancelledError:
            print("AgentLoop: 当前运行循环任务被中断信号取消。")
            raise # 重新抛出取消异常，以通知上层

        except Exception as e:
            print(f"AgentLoop 在执行循环时发生错误: {e}")
            import traceback
            traceback.print_exc() # 打印完整的错误栈，便于调试

    async def _run(self):
        """主循环的内部实现，包含中断监控"""
        self._is_running = True
        print(f"Agent 主循环已启动，心跳间隔: {self.heartbeat_interval} 秒。")
        while self._is_running:
            current_cycle_task = None
            try:
                # 启动一次 _run_once 任务
                current_cycle_task = asyncio.create_task(self._run_once())
                
                # 等待任务完成 或 等待中断信号被设置
                done, pending = await asyncio.wait(
                    {current_cycle_task, AgentLoop.agent_interrupt_event.wait()},
                    return_when=asyncio.FIRST_COMPLETED
                )

                if AgentLoop.agent_interrupt_event.wait() in done:
                    # 如果是中断事件先发生
                    print("AgentLoop: 检测到中断信号！取消当前运行循环任务。")
                    current_cycle_task.cancel() # 取消当前任务
                    await asyncio.gather(current_cycle_task, return_exceptions=True) # 确保任务被取消完成
                    AgentLoop.agent_interrupt_event.clear() # 清除中断信号，等待下次触发
                    # 直接进入下一个循环迭代，不等待心跳间隔
                    continue 

                # 如果是 _run_once 任务先完成（即没有中断）
                await current_cycle_task # 收集任务结果或任何未处理的异常
                await asyncio.sleep(self.heartbeat_interval) # 等待心跳间隔
            
            except asyncio.CancelledError:
                # 整个 _run 方法被取消，通常发生在 stop() 调用时
                print("AgentLoop: 主循环任务被外部取消。")
                if current_cycle_task and not current_cycle_task.done():
                    current_cycle_task.cancel()
                    await asyncio.gather(current_cycle_task, return_exceptions=True)
                break
            except Exception as e:
                print(f"AgentLoop: 主循环中发生未捕获错误: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(self.heartbeat_interval) # 错误后也等待，避免高速循环

    def start(self):
        """启动 Agent 的主循环"""
        if not self._is_running:
            AgentLoop.agent_interrupt_event.clear() # 启动前确保中断信号是清除状态
            self._main_task = asyncio.create_task(self._run())
            print("AgentLoop: 主循环开始尝试运行。")
        else:
            print("Agent 已经在运行中。")

    def stop(self):
        """停止 Agent 的主循环"""
        if self._is_running and self._main_task:
            self._is_running = False # 设置标志位
            self._main_task.cancel() # 取消主循环任务
            print("AgentLoop: 主循环停止请求已发送。")
        else:
            print("Agent 未在运行中。")












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
