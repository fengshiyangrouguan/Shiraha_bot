"""
Generic Actions - 通用行为范式

定义最基本的 Action 类型，所有行为都基于这些范式扩展。
"""
import time
import uuid
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from enum import Enum, auto
from dataclasses import dataclass, field

from src.core.task.models import BaseAction, Priority
from src.common.logger import get_logger

logger = get_logger("generic_actions")

if TYPE_CHECKING:
    from src.core.context.unified_context import UnifiedContext
    from src.core.memory.unified_memory import UnifiedMemory


class ActionStatus(Enum):
    """Action 执行状态"""
    PENDING = "pending"       # 等待执行
    RUNNING = "running"       # 执行中
    BLOCKED = "blocked"       # 阻塞等待
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败


@dataclass
class ActionSignal:
    """
    Action 信号

    Action 产出信号，由 TaskManager 解释并更新任务状态
    """
    signal_type: str  # NOOP, YIELD_READY, YIELD_BACKGROUND, BLOCK, WAKE_READY, FINISH, PUSH_ACTION
    payload: Dict[str, Any] = field(default_factory=dict)
    skill_update: Optional[Dict[str, Any]] = None  # 可选：skill 更新内容

    @staticmethod
    def noop() -> "ActionSignal":
        return ActionSignal(signal_type="NOOP")

    @staticmethod
    def yield_ready() -> "ActionSignal":
        return ActionSignal(signal_type="YIELD_READY")

    @staticmethod
    def yield_background() -> "ActionSignal":
        return ActionSignal(signal_type="YIELD_BACKGROUND")

    @staticmethod
    def block(event_id: str = "") -> "ActionSignal":
        return ActionSignal(signal_type="BLOCK", payload={"event_id": event_id})

    @staticmethod
    def wake_ready() -> "ActionSignal":
        return ActionSignal(signal_type="WAKE_READY")

    @staticmethod
    def finish(result: str = "") -> "ActionSignal":
        return ActionSignal(signal_type="FINISH", payload={"result": result})

    @staticmethod
    def push_action(action_name: str, **kwargs) -> "ActionSignal":
        return ActionSignal(signal_type="PUSH_ACTION", payload={
            "action_name": action_name,
            "params": kwargs
        })

    @staticmethod
    def update_skill(pattern: str, description: str) -> "ActionSignal":
        """产出信号，请求更新 skill"""
        return ActionSignal(
            signal_type="NOOP",
            skill_update={
                "pattern": pattern,
                "description": description
            }
        )


class GenericAction(BaseAction):
    """
    通用 Action 基类

    所有具体 Action 都应基于此类，扩展 execute 和 on_perception 方法
    """
    def __init__(
        self,
        action_id: str,
        priority: Priority = Priority.MEDIUM,
        skill_name: str = ""
    ):
        super().__init__(action_id, priority)
        self.skill_name = skill_name
        self.status = ActionStatus.PENDING
        self._start_time: Optional[float] = None
        self._context: Optional["UnifiedContext"] = None
        self._memory: Optional["UnifiedMemory"] = None

    def set_runtime(self, context: "UnifiedContext", memory: "UnifiedMemory"):
        """设置运行时依赖"""
        self._context = context
        self._memory = memory

    async def execute(self, ctx: Dict[str, Any]) -> Optional[str]:
        """
        执行 Action

        Args:
            ctx: 运行时上下文

        Returns:
            结束消息（可选）
        """
        self.status = ActionStatus.RUNNING
        self._start_time = time.time()

        logger.info(f"Action {self.action_id} ({self.skill_name}) 开始执行")

        # 子类实现具体逻辑
        signal = await self.do_execute(ctx)

        # 处理返回的信号
        if signal:
            logger.debug(f"Action {self.action_id} 产出信号: {signal.signal_type}")
            # TODO: 将信号传递给 TaskManager

        return signal

    async def do_execute(self, ctx: Dict[str, Any]) -> Optional[ActionSignal]:
        """子类实现的具体执行逻辑"""
        return ActionSignal.noop()

    def on_perception(self, data: Any) -> Optional[ActionSignal]:
        """
        感知事件处理（可被子类覆写）

        在 Action 不处于执行态时，处理传入的事件数据

        Args:
            data: 事件数据

        Returns:
            处理结果信号

        注意：
        - 默认返回 NOOP 信号
        - 子类可以覆写此方法实现自定义的感知逻辑
        - 感知处理应该是轻量级的，避免调用 LLM
        """
        return ActionSignal.noop()

    def get_duration(self) -> float:
        """获取执行时长"""
        if self._start_time:
            return time.time() - self._start_time
        return 0.0


class SequentialAction(GenericAction):
    """
    顺序执行型 Action

    按顺序执行一系列步骤，适用于：阅读、规划、多步任务
    """

    def __init__(
        self,
        action_id: str,
        skill_name: str,
        steps: List[str],
        priority: Priority = Priority.MEDIUM
    ):
        super().__init__(action_id, priority, skill_name)
        self.steps = steps
        self.current_step = 0
        self.step_data: List[Any] = [None] * len(steps)

    async def do_execute(self, ctx: Dict[str, Any]) -> Optional[ActionSignal]:
        """执行当前步骤"""
        if self.current_step >= len(self.steps):
            # 所有步骤完成
            return ActionSignal.finish()

        logger.debug(f"SequentialAction: 执行步骤 {self.current_step + 1}/{len(self.steps)}: {self.steps[self.current_step]}")

        # TODO: 调用子规划器执行当前步骤
        # result = await self._execute_step(self.steps[self.current_step], ctx)
        # self.step_data[self.current_step] = result

        # 推进到下一步
        self.current_step += 1

        if self.current_step >= len(self.steps):
            return ActionSignal.finish()
        else:
            # 让出焦点，下次继续
            return ActionSignal.yield_ready()

    def on_perception(self, data: Any) -> Optional[ActionSignal]:
        """顺序执行型 Action 通常不处理感知"""
        return ActionSignal.noop()


class BlockingAction(GenericAction):
    """
    阻塞等待型 Action

    等待特定条件或事件，适用于：等回复、等工具结果
    """

    def __init__(
        self,
        action_id: str,
        skill_name: str,
        wait_criteria: str,
        timeout: float = 300,
        priority: Priority = Priority.MEDIUM
    ):
        super().__init__(action_id, priority, skill_name)
        self.wait_criteria = wait_criteria
        self.timeout = timeout
        self._start_wait: Optional[float] = None
        self._waits_for: str = ""  # 等待的事件ID

    async def do_execute(self, ctx: Dict[str, Any]) -> Optional[ActionSignal]:
        """进入阻塞状态"""
        self._start_wait = time.time()
        self._waits_for = f"{self.action_id}_wait"

        logger.debug(f"BlockingAction: 开始等待 - {self.wait_criteria}")
        # TODO: 通知系统开始等待该事件

        # 返回 BLOCK 信号
        return ActionSignal.block(event_id=self._waits_for)

    def on_perception(self, data: Any) -> Optional[ActionSignal]:
        """检查等待条件是否满足"""
        if not self._start_wait:
            return ActionSignal.noop()

        # 检查是否超时
        elapsed = time.time() - self._start_wait
        if elapsed > self.timeout:
            logger.warning(f"BlockingAction: 等待超时 ({self.timeout}s)")
            return ActionSignal.finish(result="超时")

        # 检查条件是否满足
        if self._check_condition(data):
            logger.debug(f"BlockingAction: 条件满足 - {self.wait_criteria}")
            return ActionSignal.finish(result="条件满足")

        return ActionSignal.noop()

    def _check_condition(self, data: Any) -> bool:
        """检查等待条件（子类可重写）"""
        # 默认实现：简单检查数据中是否有关键信息
        if isinstance(data, dict):
            return any(keyword in str(data).lower()
                      for keyword in self.wait_criteria.lower().split())
        return False


class PerceptionAction(GenericAction):
    """
    感知监听型 Action

    在后台监听特定信号，适用于：监听消息、等待触发词
    """

    def __init__(
        self,
        action_id: str,
        skill_name: str,
        focus_keywords: List[str] = None,
        perception_patterns: List[str] = None,
        priority: Priority = Priority.LOW
    ):
        super().__init__(action_id, priority, skill_name)
        self.focus_keywords = focus_keywords or []
        self.perception_patterns = perception_patterns or []
        self._matches: List[Dict[str, Any]] = []

    async def do_execute(self, ctx: Dict[str, Any]) -> Optional[ActionSignal]:
        """
        感知型 Action 默认不主动执行

        只有在 on_perception 中捕获到信号时才会被激活
        """
        # 检查是否有累积的匹配
        if self._matches:
            # 有匹配，处理并清空
            return self._process_matches()

        # 没有匹配，回 BACKGROUND
        return ActionSignal.yield_background()

    def on_perception(self, data: Any) -> Optional[ActionSignal]:
        """
        轻量级匹配，命中则记录

        这是感知型 Action 的核心，必须快速完成，不要调用 LLM
        """
        if not data:
            return ActionSignal.noop()

        data_str = str(data).lower()

        # 关键词匹配
        for keyword in self.focus_keywords:
            if keyword.lower() in data_str:
                logger.debug(f"PerceptionAction: 命中关键词 '{keyword}'")
                self._matches.append({
                    "type": "keyword",
                    "keyword": keyword,
                    "data": data,
                    "timestamp": time.time()
                })
                # 可以选择立即唤醒
                return ActionSignal.wake_ready()

        # 模式匹配
        for pattern in self.perception_patterns:
            if pattern.lower() in data_str:
                logger.debug(f"PerceptionAction: 命中模式 '{pattern}'")
                self._matches.append({
                    "type": "pattern",
                    "pattern": pattern,
                    "data": data,
                    "timestamp": time.time()
                })
                return ActionSignal.wake_ready()

        return ActionSignal.noop()

    def _process_matches(self) -> ActionSignal:
        """处理累积的匹配（子类可重写）"""
        # 默认：记录到记忆并继续后台监听
        if self._memory:
            for match in self._matches:
                content = f"感知到: {match.get('keyword') or match.get('pattern')}"
                # TODO: asyncio.run(self.memory.store(content, ...))

        # 清空匹配
        self._matches.clear()

        # 继续后台监听
        return ActionSignal.yield_background()


class SingleStepAction(GenericAction):
    """
    一次性执行型 Action

    执行一次即完成，适用于：发送消息、更新状态、简单工具调用
    """

    def __init__(
        self,
        action_id: str,
        skill_name: str,
        command: str,
        command_params: Dict[str, Any] = None,
        priority: Priority = Priority.MEDIUM
    ):
        super().__init__(action_id, priority, skill_name)
        self.command = command
        self.command_params = command_params or {}

    async def do_execute(self, ctx: Dict[str, Any]) -> Optional[ActionSignal]:
        """执行单步命令"""
        logger.debug(f"SingleStepAction: 执行命令 - {self.command}")

        # TODO: 执行命令
        # result = await self._execute_command(self.command, self.command_params, ctx)

        # 记录结果到记忆
        if self._memory:
            result_text = f"执行 {self.command} 完成"
            # TODO: asyncio.run(self.memory.store(result_text, ...))

        # 直接完成
        return ActionSignal.finish(result=f"命令 {self.command} 已执行")

    def on_perception(self, data: Any) -> Optional[ActionSignal]:
        """单步Action不处理感知"""
        return ActionSignal.noop()


class LoopAction(GenericAction):
    """
    循环执行型 Action

    持续执行直到条件满足，适用于：长时间任务、持续监控
    """

    def __init__(
        self,
        action_id: str,
        skill_name: str,
        stop_condition: str,
        max_iterations: int = 100,
        priority: Priority = Priority.LOW
    ):
        super().__init__(action_id, priority, skill_name)
        self.stop_condition = stop_condition
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self._loop_data: List[Any] = []

    async def do_execute(self, ctx: Dict[str, Any]) -> Optional[ActionSignal]:
        """执行一次循环"""
        if self.current_iteration >= self.max_iterations:
            logger.warning(f"LoopAction: 达到最大迭代次数 {self.max_iterations}")
            return ActionSignal.finish(result="达到最大迭代次数")

        logger.debug(f"LoopAction: 迭代 {self.current_iteration + 1}/{self.max_iterations}")

        # 检查停止条件
        if self._check_stop_condition(ctx):
            logger.debug(f"LoopAction: 停止条件满足 - {self.stop_condition}")
            return ActionSignal.finish(result=f"完成 {self.current_iteration} 次迭代")

        # 执行循环体
        # TODO: await self._execute_loop_body(ctx)
        self.current_iteration += 1

        # 让出焦点，下次继续
        return ActionSignal.yield_ready()

    def on_perception(self, data: Any) -> Optional[ActionSignal]:
        """循环Action可以让外部信号提前终止"""
        if self._check_stop_condition({"external": data}):
            logger.debug(f"LoopAction: 外部信号触发停止 - {self.stop_condition}")
            return ActionSignal.finish(result="外部信号终止")

        return ActionSignal.noop()

    def _check_stop_condition(self, data: Dict[str, Any]) -> bool:
        """检查停止条件（子类可重写）"""
        # 默认实现：简单字符串匹配
        data_str = str(data).lower()
        return self.stop_condition.lower() in data_str


# 工厂函数
def create_action(
    action_type: str,
    skill_name: str,
    **kwargs
) -> GenericAction:
    """
    动态创建 Action

    Args:
        action_type: Action 类型
        skill_name: 关联的 skill 名称
        **kwargs: Action 特定参数

    Returns:
        GenericAction 实例
    """
    action_id = kwargs.get("action_id", f"act_{uuid.uuid4().hex[:8]}")
    priority = kwargs.get("priority", Priority.MEDIUM)

    if action_type == "sequential":
        return SequentialAction(
            action_id=action_id,
            skill_name=skill_name,
            steps=kwargs.get("steps", []),
            priority=priority
        )
    elif action_type == "blocking":
        return BlockingAction(
            action_id=action_id,
            skill_name=skill_name,
            wait_criteria=kwargs.get("wait_criteria", ""),
            timeout=kwargs.get("timeout", 300),
            priority=priority
        )
    elif action_type == "perception":
        return PerceptionAction(
            action_id=action_id,
            skill_name=skill_name,
            focus_keywords=kwargs.get("focus_keywords", []),
            perception_patterns=kwargs.get("perception_patterns", []),
            priority=priority
        )
    elif action_type == "single_step":
        return SingleStepAction(
            action_id=action_id,
            skill_name=skill_name,
            command=kwargs.get("command", ""),
            command_params=kwargs.get("command_params", {}),
            priority=priority
        )
    elif action_type == "loop":
        return LoopAction(
            action_id=action_id,
            skill_name=skill_name,
            stop_condition=kwargs.get("stop_condition", ""),
            max_iterations=kwargs.get("max_iterations", 100),
            priority=priority
        )
    else:
        logger.warning(f"未知的 Action 类型: {action_type}，使用 GenericAction")
        return GenericAction(
            action_id=action_id,
            priority=priority,
            skill_name=skill_name
        )
