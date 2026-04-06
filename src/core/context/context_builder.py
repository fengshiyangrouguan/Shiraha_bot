"""
Context Builder - 统一上下文构建器

这里是新的唯一主链上下文拼接入口。
它直接面向 MainPlanner，负责把系统身份、运行状态、内核指令说明、
Cortex 摘要、记忆结果与当前工作上下文动态拼接为标准消息列表。
"""
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.common.di.container import container
from src.common.logger import get_logger
from src.core.context.unified_context import Role, UnifiedContext
from src.core.memory import UnifiedMemory
from src.cortex_system import CortexManager

if TYPE_CHECKING:
    from src.agent.world_model import WorldModel

logger = get_logger("context_builder")


class ContextBuilder:
    """
    主规划器上下文构建器。

    设计原则：
    1. 只保留一层真正负责“拼 prompt”的实现，避免 Mind 与 ContextBuilder 双轨并存。
    2. 对外统一输出 `{role, content}` 结构，内部仍借助 UnifiedContext 做裁剪与聚合。
    3. 所有动态内容都尽量从当前运行态直接读取，不再依赖旧桥接层或旧 agent_loop。
    """

    def __init__(self):
        self.identity_path = Path("data/identity.md")
        self.identity_content = self._load_identity()

        self.world_model: Optional["WorldModel"] = None
        self.unified_memory: Optional[UnifiedMemory] = None
        self.cortex_manager: Optional[CortexManager] = None

        try:
            from src.agent.world_model import WorldModel
            self.world_model = container.resolve(WorldModel)
        except Exception as exc:
            logger.debug(f"WorldModel 尚未可用: {exc}")

        try:
            self.unified_memory = container.resolve(UnifiedMemory)
        except Exception as exc:
            logger.debug(f"UnifiedMemory 尚未可用: {exc}")

        try:
            self.cortex_manager = container.resolve(CortexManager)
        except Exception as exc:
            logger.debug(f"CortexManager 尚未可用: {exc}")

    def refresh_identity(self):
        """重新加载人格与系统身份描述。"""
        self.identity_content = self._load_identity()

    def _load_identity(self) -> str:
        """
        加载身份提示词。

        如果磁盘上没有 `data/identity.md`，则退回到一个简洁但可运行的默认身份。
        """
        if self.identity_path.exists():
            try:
                return self.identity_path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                logger.warning(f"读取 identity.md 失败，将使用默认身份: {exc}")

        return (
            "# 默认身份\n\n"
            "你是藤原白羽，一个运行在事件驱动主链上的自主 AI 智能体。\n"
            "你具备单注意力焦点，会根据外部信号、记忆和当前任务决定下一步 Shell 指令。"
        )

    async def build_main_context(
        self,
        motive: str,
        previous_observation: str = "",
        active_tasks: Optional[List[Dict[str, Any]]] = None,
        notifications: Optional[List[str]] = None,
        mood: str = "",
        energy: Optional[int] = None,
        memory_limit: int = 5,
        input_source: str = "idle_input",
        latest_signal: Optional[Dict[str, Any]] = None,
        debug_request: str = "",
    ) -> List[Dict[str, str]]:
        """
        构建供 MainPlanner 使用的完整上下文。

        输出顺序固定：
        1. system: 身份 + 状态 + Shell 指令 + Cortex 摘要
        2. memory: 相关记忆
        3. user: 当前工作上下文
        """
        context = UnifiedContext()

        cortex_summary = await self._build_cortex_summary()
        system_prompt = "\n\n".join(
            [
                self.identity_content,
                self._build_runtime_prompt(mood=mood, energy=energy),
                self._build_shell_instruction_prompt(),
                cortex_summary,
            ]
        ).strip()

        context.append(
            {
                "role": Role.SYSTEM.value,
                "content": system_prompt,
            }
        )

        for memory_message in await self._build_memory_messages(limit=memory_limit):
            context.append(memory_message)

        context.append(
            {
                "role": Role.USER.value,
                "content": self._build_working_context_prompt(
                    motive=motive,
                    previous_observation=previous_observation,
                    active_tasks=active_tasks or [],
                    notifications=notifications or [],
                    input_source=input_source,
                    latest_signal=latest_signal or {},
                    debug_request=debug_request,
                ),
            }
        )

        context.truncate(4096)
        return context.to_list()

    async def _build_cortex_summary(self) -> str:
        """
        获取当前已加载 Cortex 的概览。

        这里优先读取 CortexManager 实时汇总后的文本；如果尚未可用，则退回默认说明。
        """
        if self.cortex_manager:
            try:
                await self.cortex_manager.update_cortices_summaries()
            except Exception as exc:
                logger.warning(f"更新 Cortex 摘要失败: {exc}")

        if self.world_model and self.world_model.cortices_summaries:
            return "## Cortex 状态概览\n\n" + self.world_model.cortices_summaries

        return (
            "## Cortex 状态概览\n\n"
            "当前 Cortex 摘要尚未准备完毕，请基于已有任务、通知与记忆谨慎输出 Shell 指令。"
        )

    def _build_runtime_prompt(self, mood: str = "", energy: Optional[int] = None) -> str:
        """
        注入运行态信息，让 Planner 在同一个 system prompt 中掌握环境基线。
        """
        now = datetime.now()
        lines = [
            "## 当前运行态",
            f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}",
            "架构模式：事件驱动主链",
            "注意力模型：单焦点，多任务待调度",
            "输出协议：输出 JSON，其中包含 thought 与 shell_commands",
        ]

        if mood:
            lines.append(f"当前情绪：{mood}")
        if energy is not None:
            lines.append(f"当前能量：{energy}/100")

        return "\n".join(lines)

    def _build_shell_instruction_prompt(self) -> str:
        """
        提供内核可识别的主指令说明。

        这里保留高信号、低噪音的命令集，避免 Planner 再依赖旧 Mind 中那套拼接文本。
        """
        return """## Kernel Shell 指令说明

你必须输出严格 JSON，而不是纯文本 shell。

可用主命令：
- `task create --cortex <领域> --target <目标ID> --mode <once|listen|loop|cron> --pri <优先级> --motive "动机"`
- `task exec --id <任务ID>`
- `task run --cortex <领域> --action <原子动作> ...`
- `task view --cortex <领域> --panel <控制面板> ...`
- `task suspend --id <任务ID>`
- `task resume --id <任务ID>`
- `task block --id <任务ID>`
- `task kill --id <任务ID>`
- `task adjust_prio --id <任务ID> --pri <优先级>`
- `memory store --content "内容" --type <记忆类型> --cortex <域> --target <目标>`
- `memory retrieve --query "查询" --limit <数量>`

优先级：`critical > high > medium > low`

输出规则：
1. 必须输出 JSON，对象中至少包含：
   - `thought`: 你的想法、动机或判断
   - `shell_commands`: shell 命令数组
2. `shell_commands` 里的每一项必须是一条完整命令字符串。
3. 未显式查看控制面板前，禁止假设其内部内容。
4. Skill 与 cortex 信息只基于已加载片段推理，禁止编造未加载能力。

输出示例：
{
  "thought": "这条消息先交给回复器监听任务判断是否要回应。",
  "shell_commands": [
    "task exec --id reply_listener_qq_main"
  ]
}"""

    async def _build_memory_messages(self, limit: int) -> List[Dict[str, str]]:
        """
        从 UnifiedMemory 拉取近期重要记忆。

        记忆消息使用独立的 `memory` role，方便后续继续区分系统指令与历史经验。
        """
        if not self.unified_memory:
            return []

        try:
            memories = await self.unified_memory.retrieve(
                query="最近的重要任务、对话和观察",
                limit=limit,
                semantic=True,
            )
        except Exception as exc:
            logger.warning(f"检索统一记忆失败: {exc}")
            return []

        result: List[Dict[str, str]] = []
        for memory in memories:
            result.append(
                {
                    "role": Role.MEMORY.value,
                    "content": f"[{memory.created_at}] {memory.content}",
                }
            )
        return result

    def _build_working_context_prompt(
        self,
        motive: str,
        previous_observation: str,
        active_tasks: List[Dict[str, Any]],
        notifications: List[str],
        input_source: str,
        latest_signal: Dict[str, Any],
        debug_request: str,
    ) -> str:
        """
        构建当前工作上下文。

        这部分固定使用 `user` 角色，等价于把“现在你看到的现场”直接交给 Planner。
        """
        lines = ["## 当前工作上下文"]
        lines.append(f"输入来源：{input_source}")

        if motive:
            lines.append(f"当前动机：{motive}")
        else:
            lines.append("当前动机：暂无明确动机，请根据当前状态判断是否需要创建任务。")

        if active_tasks:
            lines.append(f"活跃任务数量：{len(active_tasks)}")
            for task in active_tasks[:5]:
                lines.append(
                    f"- {task.get('id', task.get('task_id', 'unknown'))} | "
                    f"状态={task.get('status', 'unknown')} | "
                    f"模式={task.get('mode', 'unknown')} | "
                    f"域={task.get('cortex', 'unknown')} | "
                    f"目标={task.get('target', task.get('target_id', 'unknown'))} | "
                    f"优先级={task.get('pri', task.get('priority', 'unknown'))}"
                )
        else:
            lines.append("当前没有活跃任务。")

        if notifications:
            lines.append("待处理通知：")
            for item in notifications:
                lines.append(f"- {item}")

        if latest_signal:
            lines.append("最新中断信号：")
            lines.append(str(latest_signal))

        if debug_request:
            lines.append("调试台输入：")
            lines.append(debug_request)

        if previous_observation:
            lines.append("上一次观察：")
            lines.append(previous_observation)

        lines.append("请基于以上信息输出下一步 JSON 规划结果。")
        return "\n".join(lines)
