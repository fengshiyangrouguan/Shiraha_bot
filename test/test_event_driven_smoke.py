import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from src.common.config.config_service import ConfigService
from src.common.di.container import container
from src.core.context.context_builder import ContextBuilder
from src.core.kernel import EventLoop, InterruptHandler, KernelInterpreter, Scheduler
from src.core.task.task_manager import TaskManager
from src.core.task.task_store import TaskStore
from src.main_system import MainSystem


class EventDrivenMainSystemSmokeTest(unittest.TestCase):
    """
    事件驱动主链烟雾测试。

    这组测试不追求覆盖所有业务细节，而是专门守住三件事：
    1. MainSystem 能否把新的事件驱动主链装起来。
    2. Cortex/外部事件能否经由 InterruptHandler 进入任务系统。
    3. 新的 ContextBuilder 能否稳定输出统一消息结构。
    """

    def setUp(self):
        # DI 容器是全局单例。每个测试前都清空，避免上一个测试的实例污染当前场景。
        container._services.clear()
        container._factories.clear()

    def tearDown(self):
        container._services.clear()
        container._factories.clear()

    def test_main_system_registers_event_driven_components(self):
        async def _run():
            system = await self._initialize_system()
            try:
                self.assertIs(container.resolve(TaskStore), system.task_store)
                self.assertIs(container.resolve(TaskManager), system.task_manager)
                self.assertIs(container.resolve(Scheduler), system.scheduler)
                self.assertIs(container.resolve(InterruptHandler), system.interrupt_handler)
                self.assertIs(container.resolve(KernelInterpreter), system.kernel_interpreter)
                self.assertIs(container.resolve(EventLoop), system.event_loop)
                self.assertTrue(system.event_loop._is_running)
            finally:
                await system.shutdown()

        asyncio.run(_run())

    def test_interrupt_flow_creates_task_and_context_builder_outputs_messages(self):
        async def _run():
            system = await self._initialize_system()
            try:
                await system.interrupt_handler.handle_external_event(
                    source_cortex="qq",
                    signal_type="message",
                    content="测试消息",
                    source_target="group_001",
                    priority="high",
                )

                # 给事件循环一个极短时间处理队列。
                await asyncio.sleep(0.2)

                tasks = await system.task_store.list_all()
                self.assertTrue(tasks, "中断信号进入后应该创建至少一个任务")
                self.assertEqual(tasks[0].cortex, "qq_chat")
                self.assertEqual(tasks[0].target_id, "group_001")
                self.assertEqual(tasks[0].mode.value, "listen")

                builder = ContextBuilder()
                messages = await builder.build_main_context(
                    motive="处理测试消息",
                    previous_observation="中断已进入队列",
                    active_tasks=await system.world_model.get_active_tasks_structured(),
                    notifications=["qq: 新消息"],
                    mood=system.world_model.mood,
                    energy=system.world_model.energy,
                )

                self.assertGreaterEqual(len(messages), 2)
                self.assertEqual(messages[0]["role"], "system")
                self.assertEqual(messages[-1]["role"], "user")
            finally:
                await system.shutdown()

        asyncio.run(_run())

    def test_interpreter_supports_json_plan_and_task_view(self):
        async def _run():
            system = await self._initialize_system()
            try:
                results = await system.kernel_interpreter.execute_batch(
                    {
                        "thought": "先查看书架，再查看会话列表。",
                        "shell_commands": [
                            "task view --cortex reading --panel view_bookshelf",
                            "task view --cortex qq_chat --panel view_conversation_list",
                        ],
                    }
                )
                self.assertEqual(results[0]["command"], "__thought__")
                self.assertEqual(results[0]["status"], "info")
                self.assertEqual(results[1]["status"], "success")
                self.assertEqual(results[2]["status"], "success")
            finally:
                await system.shutdown()

        asyncio.run(_run())

    async def _initialize_system(self) -> MainSystem:
        """
        初始化 MainSystem，但在测试里屏蔽 idle scheduler 的动机生成，
        避免测试因为真实 LLM 调用或空闲规划而变慢、变脆弱。
        """
        system = MainSystem(ConfigService())

        with patch(
            "src.agent.motive.motive_engine.MotiveEngine.generate_motive",
            new=AsyncMock(return_value=None),
        ):
            await system.initialize()

        return system


if __name__ == "__main__":
    unittest.main()
