# src/agent/world_model.py
import time
from collections import deque
from typing import List, Dict, Any, Deque, Optional
from pydantic import BaseModel  # 导入 BaseModel 用于类型提示和校验

# 导入配置
from src.common.di.container import container
from src.common.config.schemas.bot_config import BotConfig
from src.common.logger import get_logger
from src.core.task.task_store import TaskStore as CoreTaskStore

try:
    from src.core.memory import UnifiedMemory
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

logger = get_logger("world_model")


class TaskSnapshotStore:
    """
    轻量任务快照仓库。

    旧版 WorldModel 依赖已经删除的 `src.agent.task` 模块来做任务摘要缓存，
    这会直接阻断新的事件驱动主链启动。
    这里改为一个最小但稳定的内联实现，只负责：
    1. 从 core.task.Task 读取关键字段。
    2. 为 Planner 生成结构化任务摘要。
    """

    def __init__(self):
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    def upsert_from_instance(self, task) -> None:
        """把内核任务实例转换为 Planner 可消费的摘要。"""
        self._snapshots[task.task_id] = {
            "id": task.task_id,
            "task_id": task.task_id,
            "status": getattr(task.status, "value", str(task.status)),
            "mode": getattr(task.mode, "value", str(getattr(task, "mode", ""))),
            "cortex": task.cortex,
            "target": task.target_id,
            "target_id": task.target_id,
            "pri": getattr(task.priority, "value", str(task.priority)),
            "priority": getattr(task.priority, "value", str(task.priority)),
            "motive": task.motive,
            "updated_at": task.updated_at,
        }

    def summarize(self) -> List[Dict[str, Any]]:
        """按更新时间倒序返回当前任务快照。"""
        return sorted(
            self._snapshots.values(),
            key=lambda item: item.get("updated_at", 0),
            reverse=True,
        )

class WorldModel:
    """
    世界模型 - Agent 状态管理器。
    负责管理 Agent 的所有状态，包括：
    1. 静态身份 (Identity)
    2. 动态内在状态 (Internal State: motive, mood, energy)
    3. 动态外部感知 (External Perception: stimuli)
    4. 记忆 (Memory)
    """
    def __init__(self, short_term_memory_max_len: int = 10):
        print("WorldModel: 初始化...")
        
        # --- 1. 加载静态身份 ---

        bot_config: BotConfig = container.resolve(BotConfig)
        persona_config = bot_config.persona
        mood_config = bot_config.mood
        
        self.bot_name: str = persona_config.bot_name
        self.bot_nickname: List[str] = persona_config.bot_nickname
        self.bot_identity: str = persona_config.bot_identity
        self.bot_personality: str = persona_config.bot_personality
        self.bot_interest: str = ", ".join(persona_config.bot_interest)
        self.bot_expression_style: str = persona_config.expression_style

        # --- 2. 初始化动态内在状态 ---
        self.motive: str = ""
        self.mood: str = mood_config.initial_mood
        self.energy: int = mood_config.initial_energy
        self.cortices_summaries: str = ""  # 存储各个 Cortex 的实时状态摘要
        self.last_observation: str = ""
        self.current_focus_task: Optional[str] = None

        # --- 3. 初始化动态外部感知 ---
        # 使用列表来存储刺激物，方便管理
        self.notifications: Dict[str, Any] = {}
        self.alerts: List[str] = []

        # --- 4. 初始化记忆 ---

        # 使用字典来存储不同来源的事件流/状态 (兼容旧代码)
        # 这是 WorldModel 的核心，用于存储各个 Cortex 维护的特定数据，纯内存操作
        self.cortex_data: Dict[str, BaseModel] = {}

        # 新：统一记忆系统
        self.unified_memory: Optional[UnifiedMemory] = None
        if MEMORY_AVAILABLE:
            try:
                self.unified_memory = container.resolve(UnifiedMemory)
            except Exception:
                pass

        # 短期记忆（保持兼容）
        self.short_term_memory: Deque[str] = deque(maxlen=short_term_memory_max_len)

        # 心流缓存，缓存一些阅读，编程，等产生的感悟，形成侧回路，避免影响其他cortex
        self.flow_cache: Deque[str] = deque(maxlen=15)

        # Prompt 摘要缓存
        self.task_snapshot_store = TaskSnapshotStore()
        # 内核任务仓库（生命周期控制）
        try:
            self.core_task_store: CoreTaskStore = container.resolve(CoreTaskStore)
        except Exception:
            # 容器中尚未注册时，使用内置实例
            self.core_task_store = CoreTaskStore()

    async def initialize_memory(self, unified_memory: UnifiedMemory):
        """初始化统一记忆系统"""
        self.unified_memory = unified_memory
        logger.info("WorldModel: 统一记忆系统已连接")

    async def get_cortex_data(self, key: str) -> Optional[BaseModel]:
        """
        从 WorldModel 中获取指定键的 Cortex 数据。
        此方法纯内存操作。

        Args:
            key (str): 数据键名，例如 "qq_chat_data"。
            data_type (Type[BaseModel]): 期望返回的数据类型（Pydantic BaseModel）。

        Returns:
            Optional[BaseModel]: 如果找到匹配类型的数据则返回，否则返回 None。
        """
        data = self.cortex_data.get(key)
        if data:
            return data
        # 如果未找到，返回 None，由调用者处理创建新实例
        return None

    async def save_cortex_data(self, key: str, data: BaseModel) -> None:
        """
        将 Cortex 数据保存到 WorldModel 中。
        此方法纯内存操作。

        Args:
            key (str): 数据键名，例如 "qq_chat_data"。
            data (BaseModel): 要保存的数据对象（Pydantic BaseModel）。
        """
        self.cortex_data[key] = data
        logger.debug(f"上下文数据 '{key}' 已更新（内存中）。")


    def add_memory(self, action_summary: str):
        """将新的行动总结存入短期记忆。"""
        if not action_summary:
            return
        memory_entry = f"[{time.strftime('%H:%M:%S')}] {action_summary}"
        self.short_term_memory.append(memory_entry)
        logger.info(f"短期记忆已添加 - '{memory_entry}'")

    def add_flow_cache(self,flow_summary:str):
        cache = f"[{time.strftime('%H:%M:%S')}] {flow_summary}"
        self.flow_cache.append(cache)
        logger.info(f"灵感已添加 - '{flow_summary}'")


    def update_notification(self, notification: Optional[str] = None, type: Optional[str] = None):
        self.notifications[type] = notification
        logger.info(f"收到新通知 - '{notification}'")

    def set_last_observation(self, observation: Optional[str]) -> None:
        self.last_observation = (observation or "").strip()

    def get_last_observation(self) -> str:
        return self.last_observation

    def update_internal_state(self, mood: Optional[str] = None, energy_delta: Optional[int] = None):
        """更新 Agent 的内在状态。"""
        if mood:
            self.mood = mood
            logger.info(f"情绪更新为 -> {mood}")
        if energy_delta:
            self.energy = max(0, min(100, self.energy + energy_delta)) # 确保精力在0-100之间
            print(f"WorldModel: 精力变化 {energy_delta} -> 当前精力: {self.energy}")

    def _get_time_period(self) -> str:
        """根据当前小时返回时间段描述。"""
        hour = time.localtime().tm_hour

        if 6 <= hour < 9:
            return "清晨"
        elif 9 <= hour < 12:
            return "上午"
        elif 12 <= hour < 14:
            return "中午"
        elif 14 <= hour < 18:
            return "下午"
        elif 18 <= hour < 24:
            return "晚上"
        else:  # 0 <= hour < 6
            return "凌晨"
            
    def get_current_time_string(self) -> str:
        """获取全中文格式化的时间字符串。"""
        now = time.localtime()
        # 映射星期：time.localtime().tm_wday 返回 0-6 (周一到周日)
        week_days = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
        week_str = week_days[now.tm_wday]
        
        # 格式化基础时间
        time_format = time.strftime('%Y年%m月%d日 %H:%M:%S', now)
        period = self._get_time_period()
        
        return f"现在是{time_format} {week_str} {period}"

    def get_cortices_summaries(self) -> str:
        return self.cortices_summaries

    # ---------- 新增：任务与注意力摘要 ----------
    def set_focus_task(self, task_id: Optional[str]):
        self.current_focus_task = task_id

    async def refresh_task_snapshots(self):
        """从 core task store 拉取最新任务摘要，为 Planner Prompt 提供结构化输入。"""
        tasks = await self.core_task_store.list_all()
        for t in tasks:
            self.task_snapshot_store.upsert_from_instance(t)

    def get_task_summary_text(self) -> str:
        items = self.task_snapshot_store.summarize()
        if not items:
            return "当前没有活跃任务。"
        parts = []
        for item in items:
            parts.append(
                f"[{item['status']}] {item['id']} cortex={item['cortex']} target={item['target']} pri={item['pri']}"
            )
        return "\n".join(parts)

    def get_focus_summary_text(self) -> str:
        if not self.current_focus_task:
            return "当前无焦点任务。"
        return f"当前注意力: {self.current_focus_task}"

    async def get_active_tasks_structured(self) -> List[Dict[str, Any]]:
        await self.refresh_task_snapshots()
        return self.task_snapshot_store.summarize()

    def get_context_for_motive(self) -> Dict[str, Any]:
        """
        打包并返回 MotiveEngine Prompt 所需的全部上下文信息。
        返回的字典键名与 prompt_design.md 中的占位符完全对应。
        """
        # --- 格式化通知和警报 ---
        notification_str = ""
        if self.notifications != {}:
            result_parts = []
            for key, value in self.notifications.items():
                # 1. 格式化键名，并在后面加上中文冒号
                key_str = f"{key}："
                notification_description = value
                entry = f"{key_str}{notification_description}\n"
                result_parts.append(entry)
                #TODO 需建立更新通知的方法，禁止直接修改字典
            # 将所有格式化后的部分连接起来
            # 使用 "" 作为分隔符，因为每个 entry 已经自带换行符
            notification_str = "未读消息列表：\n- " + "\n".join(result_parts).strip()
        else:
            notification_str = "当前没有未读消息：\n- " + "\n- ".join(self.notifications)
        
        alert_str = ""
        if self.alerts:
            alert_str = "注意！你收到了以下紧急警报：\n- " + "\n- ".join(self.alerts)
        
        # # 清空已处理的刺激物
        # self.notifications.clear()
        # self.alerts.clear()

        # --- 格式化近期活动 ---
        action_summary_str = "以下是按时间顺序排列的近期活动：\n"+"\n".join(self.short_term_memory) 
        # action_summary_str = "\n".join(self.short_term_memory) if self.short_term_memory else "你刚刚睡醒，思维缓存正在加载中，感觉有点迷糊，需要一会儿才能完全进入状态。"
        
        context = {
            "bot_name": self.bot_name,
            "bot_identity": self.bot_identity,
            "bot_personality": self.bot_personality,
            "bot_interest": self.bot_interest, 
            "time": time.strftime('%Y年%m月%d日 %H:%M:%S 星期%A'),
            "mood": self.mood,
            "notifications": notification_str,
            "alert": alert_str or "当前没有紧急警报。",
            "action_summary": action_summary_str,
            "current_focus": self.get_focus_summary_text(),
            "task_summary": self.get_task_summary_text(),
            "last_observation": self.get_last_observation(),
        }
        return context

    async def get_full_system_state(self) -> Dict[str, Any]:
        """为 MainPlanner 提供的系统总览。"""
        active_tasks = await self.get_active_tasks_structured()
        return {
            "active_tasks": active_tasks,
            "notifications": self.notifications,
            "alerts": self.alerts,
            "last_observation": self.last_observation,
        }
