# src/core/main_system.py

from typing import Optional, List
import asyncio

# 从 common 导入，确保所有组件使用统一的数据模型
from src.common.message_models import BaseMessage, BaseResponse 

# 核心依赖：它们必须被注入
from src.core.llm.llm_client import LLMClient
from src.core.memory.memory_manager import MemoryManager
from src.core.function_calling.function_router import FunctionRouter
# 引入日志
from src.common.logger import get_logger

# 记住：这里没有 'get_xxx_manager()' 这种垃圾！

class MainSystem:
    """
    统一思维中枢：接收通用消息，决策，生成通用响应。
    职责：编排 LLM、记忆和工具调用的流程。
    """
    def __init__(
        self,
        llm_client: LLMClient,
        memory_manager: MemoryManager,
        function_router: FunctionRouter
        # 我们暂时不引入 Willingness, 等到需求更明确时再考虑如何简洁地注入
    ):
        # 1. 实用主义：只存储必需的依赖引用
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.function_router = function_router
        self.logger = get_logger("MainSystem")

        # 2. 状态：我们可能会有一些内部后台任务需要管理
        self._background_tasks: List[asyncio.Task] = []
        
        self.logger.debug("MainSystem 实例化完成，等待初始化。")

    async def initialize(self):
        """
        初始化系统组件：加载数据、设置连接。
        这里可以启动记忆系统的后台任务，但其管理权仍在 MemoryManager 手中。
        """
        self.logger.info("核心组件启动中...")
        
        # 1. 初始化所有依赖组件
        # 注意：这里我们假设 LLMClient 和 FunctionRouter 的初始化是轻量级的，
        # 如果需要 await，则需要添加各自的 initialize 方法。
        await self.memory_manager.initialize() 
        
        # 2. 启动核心系统的内部后台任务 (例如，记忆系统的定时任务)
        memory_tasks = self.memory_manager.get_background_tasks()
        for task_coro in memory_tasks:
            # 将 MemoryManager 提供的任务封装并加入列表，以便 cleanup 时取消
            task = asyncio.create_task(task_coro, name=task_coro.__name__)
            self._background_tasks.append(task)
            self.logger.debug(f"启动内部后台任务: {task.get_name()}")
            
        self.logger.info("MainSystem 初始化完成。")

    async def _handle_tool_flow(self, context, llm_response) -> BaseResponse:
        """
        负责处理功能调用流程（Function Calling）的私有方法。
        将二次 LLM 调用的复杂性封装起来，以保持 process_message 的扁平化。
        """
        # 1. 执行外部工具
        # FunctionRouter 必须封装所有 I/O 错误并返回结构化结果
        tool_result_content = await self.function_router.execute(llm_response.tool_call_data)
        
        # 2. 第二次 LLM 调用：决策如何回复用户
        # 假设 llm_client 提供了处理工具结果的方法
        final_response = await self.llm_client.predict_with_tool_result(
            context=context, 
            tool_call_result=tool_result_content
        )
        
        return final_response.to_base_response()

    async def process_message(self, message: BaseMessage) -> BaseResponse:
        """
        核心消息处理入口，由平台适配器调用。
        **如果这里的逻辑超过 2 层缩进，你的设计就完蛋了。**
        """
        # 1. 记忆加载/上下文设置
        context = await self.memory_manager.load_context(message.user_id, message.conversation_id)
        
        try:
            # 2. 意图分析/首次 LLM 调用
            llm_response = await self.llm_client.predict(context, message.text_content)
            
            # 3. 决策：是否需要功能调用？
            if llm_response.needs_function_call:
                # 移交控制权给私有方法，保持扁平化
                final_response = await self._handle_tool_flow(context, llm_response)
            else:
                final_response = llm_response.to_base_response()
                
            # 4. 记忆存储/更新 (无论成功与否都应尝试更新)
            await self.memory_manager.save_context(message.user_id, final_response.new_context)
            
            return final_response
            
        except Exception as e:
            # 5. 错误处理：核心系统决不能因为处理失败而崩溃
            self.logger.error(f"处理消息失败: {message.id} | 错误: {e}", exc_info=True)
            # 返回一个清晰的错误响应，让平台适配器知道如何通知用户
            return BaseResponse(
                text_content=f"系统错误：无法处理您的请求。[{type(e).__name__}]",
                is_error=True
            )

    async def cleanup(self):
        """
        系统清理：在 app_main.py 的 finally 块中被调用。
        """
        self.logger.info("MainSystem 正在清理资源...")
        
        # A. 取消所有内部后台任务 (如记忆系统的定时任务)
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # B. 等待任务完成取消
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
        # C. 清理依赖组件（如数据库连接、会话关闭）
        await self.llm_client.close()
        await self.memory_manager.close()
        
        self.logger.info("MainSystem 清理完成。")