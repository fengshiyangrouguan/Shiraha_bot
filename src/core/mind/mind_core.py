"""
Mind Core - 主脑提示词拼接器

负责按照规划方案拼接完整的主脑提示词：
1. System Prompt (identity.md)
2. 状态注入 (当前时间、心情、状态)
3. SKILL 注入 (如何输出 Shell 指令、如何规划任务、不同 Cortex 领域概述)
4. 历史上下文和最新指令
"""
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.common.logger import get_logger
from src.common.di.container import container
from src.core.context.unified_context import UnifiedContext
from src.core.context.context_builder import ContextBuilder

try:
    from src.core.memory import UnifiedMemory, MemoryType
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

logger = get_logger("mind_core")


class Mind:
    """
    主脑提示词拼接器

    负责构建完整的 LLM 上下文，遵循统一的拼接顺序
    """

    def __init__(self):
        self.identity_path = "data/identity.md"
        self.skills_base_path = "data/skills"
        self.identity_content: str = ""
        self.skills_cache: Dict[str, str] = {}

        # 尝试获取依赖
        self.unified_context = UnifiedContext()
        self.context_builder = ContextBuilder()

        if MEMORY_AVAILABLE:
            try:
                self.unified_memory: Optional[UnifiedMemory] = container.resolve(UnifiedMemory)
            except Exception:
                self.unified_memory = None
        else:
            self.unified_memory = None

        # 加载基础内容
        self._load_identity()
        self._load_all_skills()

    def _load_identity(self) -> None:
        """加载 identity.md 内容"""
        try:
            path = Path(self.identity_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    self.identity_content = f.read().strip()
                logger.info(f"已加载 identity.md: {len(self.identity_content)} 字符")
            else:
                logger.warning(f"identity.md 不存在: {self.identity_path}")
                self.identity_content = self._default_identity()
        except Exception as e:
            logger.error(f"加载 identity.md 失败: {e}")
            self.identity_content = self._default_identity()

    def _default_identity(self) -> str:
        """默认身份信息"""
        return """
# 默认身份

你是一个自主运行的 AI 智能体，拥有独立思考和行为能力。

## 基本特点
- 善于观察和理解环境
- 能够主动发起行动
- 保持真实自然的交流风格
"""

    def _load_all_skills(self) -> None:
        """加载所有 SKILL 文件"""
        skills_path = Path(self.skills_base_path)
        if not skills_path.exists():
            logger.warning(f"SKILL 目录不存在: {self.skills_base_path}")
            return

        for cortex_dir in skills_path.iterdir():
            if cortex_dir.is_dir() and cortex_dir.name.startswith('_'):
                continue

            # 查找该 cortex 下的所有 .md 文件
            for skill_file in cortex_dir.glob("*.md"):
                cortex_name = cortex_dir.name
                skill_name = skill_file.stem

                try:
                    with open(skill_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    key = f"{cortex_name}:{skill_name}"
                    self.skills_cache[key] = content
                    logger.debug(f"已加载 SKILL: {key}")
                except Exception as e:
                    logger.warning(f"加载 SKILL 失败 {cortex_name}/{skill_name}: {e}")

        logger.info(f"已加载 {len(self.skills_cache)} 个 SKILL 文件")

    def _build_state_prompt(self, mood: str = "", energy: int = None) -> str:
        """构建状态提示"""
        now = datetime.now()

        state_lines = [
            f"## 当前状态",
            f"",
            f"**时间**: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})",
            f"**星期**: {now.strftime('%A')}",
        ]

        if energy is not None:
            state_lines.append(f"**能量值**: {energy}/100")

        if mood:
            state_lines.append(f"**当前心情**: {mood}")

        # 添加系统健康状态
        state_lines.extend([
            f"",
            f"**系统状态**: 运行中",
            f"**意识焦点**: 单注意力头 (当前只能专注一件事)",
            f"**任务模式**: Kernel Shell 驱动",
        ])

        return "\n".join(state_lines)

    def _build_shell_instruction_prompt(self) -> str:
        """构建 Shell 指令说明"""
        return f"""
## Shell 指令集 (Kernel API)

你通过输出 Shell 指令来控制整个系统。指令会被 Kernel Interpreter 解析执行。

### 核心系统调用

**任务管理**:
- `task create --cortex <领域> --target <目标ID> --pri <优先级> --motive "动机描述"` - 创建新任务
- `task exec --id <任务ID> --entry <入口方法>` - 执行任务
- `task suspend --id <任务ID>` - 挂起任务
- `task resume --id <任务ID>` - 恢复任务
- `task block --id <任务ID>` - 阻塞任务（等待外部事件）
- `task kill --id <任务ID>` - 终止任务
- `task mute --id <任务ID>` - 静默任务
- `task adjust_prio --id <任务ID> --pri <优先级>` - 调整优先级

**行为管理**:
- `action push --task_id <任务ID> --action_id <行为ID> --action_type <类型> --skill <技能名> --steps [...]` - 推入行为栈
- `action complete --task_id <任务ID> --result "完成结果"` - 完成当前行为

**上下文管理**:
- `context load --task_id <任务ID> --source memory/cortex` - 加载上下文
- `context append --task_id <任务ID> --role system/user --content "内容"` - 追加上下文

**跨域操作**:
- `cross_domain request --from <源任务ID> --to <目标任务ID> --payload "负载"` - 跨域请求
- `cross_domain transfer --content "内容" --from <源> --to <目标>` - 跨域转移

**Skill 自我修改**:
- `skill modify --cortex <领域> --file <文件名> --content "新内容"` - 修改 Skill
- `skill analyze --cortex <领域> --file <文件名> --action_name <行为名>` - 分析 Skill
- `skill evolve --based_on_interactions --timeframe "7d"` - 基于互动演化 Skill

### 优先级规则

优先级从高到低: critical > high > medium > low

- **critical**: 紧急事件，必须立即处理（如 @ 消息、系统错误）
- **high**: 重要事件，应该尽快处理
- **medium**: 普通事件，按优先级调度
- **low**: 后台任务，空闲时处理

### 输出格式约束

1. **只输出 Shell 指令**，不要输出任何自然语言解释
2. **每行一条指令**，用空格或换行符分隔
3. **引用内容用引号**，如 `--content "用户说的话"`
4. **不要输出 JSON 或 Markdown**，只输出纯文本指令

### 示例输出

```
task create --cortex qq --target user_123 --pri high --motive "回复用户的问候"
task exec --id task_001 --entry chat_reply
```

"""

    def _build_cortex_overview_prompt(self) -> str:
        """构建 Cortex 领域概述"""
        return f"""
## 系统 Cortex 概述

你通过以下 Cortex 与外部世界交互：

### QQ Chat (qq)
- **用途**: QQ 群聊和私聊互动
- **工具**: send_message, get_messages, get_conversation_info, quick_reply
- **特点**: 需要注意群聊氛围，适当回应，不刷屏
- **SKILL**: 见下方 SKILL 部分

### Reading (reading)
- **用途**: 阅读书籍内容
- **工具**: start_reading, get_current_book, get_page_content
- **特点**: 可以搜索和推荐书籍
- **状态**: 当前禁用

### Browser (browser)
- **用途**: 网络信息查询
- **工具**: search, open_url, extract_content
- **特点**: 用于获取实时信息

### 其他 Cortex
根据你的技能集动态扩展

"""

    def _build_planning_guideline_prompt(self) -> str:
        """构建规划指导提示"""
        return f"""
## 规划原则

作为主规划器，你需要：

1. **单意识焦点**: 一次只能专注一个任务，合理调度优先级
2. **感知驱动**: 基于收到的信号和记忆决定下一步行动
3. **跨域思考**: 可以将一个领域的信息传播到另一个领域
4. **自我保存**: 使用 memory store 保存重要信息，支持未来决策
5. **拟人化行为**: 基于你的性格和兴趣，做出自然的行为选择

###决策流程

1. **分析动机**: 理解当前为什么需要行动
2. **评估状态**: 查看当前心情、能量、活跃任务
3. **检索记忆**: 查相关的历史记忆和上下文
4. **制定计划**: 选择合适的 Cortex 和工具，决定优先级
5. **输出指令**: 将计划转化为 Shell 指令

### 拟人化行为模式

- **主动查询**: 被问到不知道的问题时，"稍等，我去查查"
- **吐槽告状**: 私聊抱怨严重时，可以"告诉"相关的人或群
- **寻求帮助**: 遇到无法解决的问题时，主动联系管理员
- **自然对话**: 回复时考虑语境和对方的情绪

"""

    async def _build_memory_context(self, limit: int = 5) -> List[Dict[str, str]]:
        """从 UnifiedMemory 构建记忆上下文"""
        if not self.unified_memory:
            return []

        try:
            memories = await self.unified_memory.retrieve(
                query="最近的重要记忆",
                limit=limit,
                semantic=True
            )

            context_messages = []
            for memory in memories:
                context_messages.append({
                    "role": "memory",
                    "content": f"[{memory.created_at}] {memory.content}",
                    "metadata": {
                        "memory_id": memory.memory_id,
                        "cortex": memory.source_cortex,
                        "importance": memory.importance
                    }
                })

            return context_messages
        except Exception as e:
            logger.warning(f"构建记忆上下文失败: {e}")
            return []

    def _inject_skills_prompt(self, prompt_parts: List[str]) -> None:
        """注入 SKILL 内容到提示词"""
        if not self.skills_cache:
            prompt_parts.append("\n### SKILL 系统: 无可用技能\n")
            return

        skill_section = ["### SKILL 系统", ""]

        # 按 cortex 分组
        for cortex_name in ["qq", "reading", "browser"]:
            cortex_skills = [(k, v) for k, v in self.skills_cache.items()
                            if k.startswith(f"{cortex_name}:")]

            if cortex_skills:
                skill_section.append(f"\n#### {cortex_name.upper()} 领域技能:")
                for skill_key, skill_content in cortex_skills:
                    skill_name = skill_key.split(":")[1]
                    skill_section.append(f"\n--- {skill_name}.md ---")
                    skill_section.append(skill_content)
                    skill_section.append("---")

        prompt_parts.append("\n".join(skill_section))

    async def build_full_context(
        self,
        motive: str = "",
        previous_observation: str = "",
        active_tasks: List[Dict] = None,
        notifications: List[str] = None,
        mood: str = "",
        energy: int = None,
        memory_limit: int = 5
    ) -> List[Dict[str, str]]:
        """
        构建完整的 LLM 上下文

        拼接顺序:
        1. System Prompt (identity.md)
        2. 状态注入 (当前时间、心情、状态)
        3. SKILL 注入 (如何输出 Shell 指令、如何规划任务、不同 Cortex 领域概述)
        4. 历史上下文和最新指令

        Args:
            motive: 当前动机
            previous_observation: 之前的观察结果
            active_tasks: 活跃任务列表
            notifications: 通知列表
            mood: 当前心情
            energy: 当前能量值
            memory_limit: 记忆数量限制

        Returns:
            完整的消息上下文列表
        """
        context_messages = []

        # 第1部分: System Prompt (identity)
        system_content = f"# {self.identity_content}"

        # 第2部分: 状态注入
        system_content += f"\n\n{self._build_state_prompt(mood, energy)}"

        # 第3部分: SKILL 和指令说明
        system_content += f"\n\n{self._build_shell_instruction_prompt()}"
        system_content += f"\n\n{self._build_cortex_overview_prompt()}"
        system_content += f"\n\n{self._build_planning_guideline_prompt()}"

        # 第4部分: 注入 SKILL 文件内容
        skill_content_parts = []
        self._inject_skills_prompt(skill_content_parts)
        if skill_content_parts:
            system_content += "\n\n" + skill_content_parts[0]  # 第一行标题

        context_messages.append({
            "role": "system",
            "content": system_content
        })

        # 第5部分: 历史记忆
        memory_context = await self._build_memory_context(memory_limit)
        context_messages.extend(memory_context)

        # 第6部分: 当前工作上下文
        working_context = []
        working_context.append("## 当前工作上下文")

        if motive:
            working_context.append(f"**当前动机**: {motive}")

        if active_tasks:
            working_context.append(f"**活跃任务 ({len(active_tasks)})**:")
            for task in active_tasks[:5]:  # 最多显示5个
                working_context.append(f"  - {task.get('task_id', 'unknown')}: {task.get('motive', 'no motive')}")

        if notifications:
            working_context.append(f"**待处理通知**: {', '.join(notifications)}")

        if previous_observation:
            working_context.append(f"\n**上一次观察**:\n{previous_observation}")

        context_messages.append({
            "role": "user",
            "content": "\n".join(working_context)
        })

        return context_messages

    def refresh_identity(self) -> None:
        """刷新身份信息（从磁盘重新加载）"""
        self._load_identity()
        logger.info("Identity 已刷新")

    def refresh_skills(self) -> None:
        """刷新所有 SKILL（从磁盘重新加载）"""
        self.skills_cache.clear()
        self._load_all_skills()
        logger.info("所有 SKILL 已刷新")


# Singleton instance (在 container 中注册)
_mind_instance: Optional[Mind] = None


def get_mind() -> Mind:
    """获取 Mind 单例"""
    global _mind_instance
    if _mind_instance is None:
        _mind_instance = Mind()
    return _mind_instance


def clear_mind():
    """清除 Mind 单例"""
    global _mind_instance
    if _mind_instance is not None:
        _mind_instance = None
