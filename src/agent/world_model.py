# src/agent/world_model.py
import time
from collections import deque
from typing import List, Dict, Any, Deque, Optional, Type
from pydantic import BaseModel # 导入 BaseModel 用于类型提示和校验

# 导入配置
from src.system.di.container import container
from src.common.config.schemas.bot_config import BotConfig

class WorldModel:
    """
    世界模型 - Agent 状态管理器。
    负责管理 Agent 的所有状态，包括：
    1. 静态身份 (Identity)
    2. 动态内在状态 (Internal State: motive, mood, energy)
    3. 动态外部感知 (External Perception: stimuli)
    4. 记忆 (Memory)
    """
    def __init__(self, short_term_memory_max_len: int = 12):
        print("WorldModel: 初始化...")
        
        # --- 1. 加载静态身份 ---

        bot_config: BotConfig = container.resolve(BotConfig)
        persona_config = bot_config.persona
        mood_config = bot_config.mood
        
        self.bot_name: str = persona_config.bot_name
        self.bot_identity: str = persona_config.bot_identity
        self.bot_personality: str = persona_config.bot_personality
        self.bot_interest: str = ", ".join(persona_config.bot_interest)
        self.bot_expression_style: str = persona_config.expression_style

        # --- 2. 初始化动态内在状态 ---
        self.motive: str = ""
        self.mood: str = mood_config.initial_mood
        self.energy: int = mood_config.initial_energy

        # --- 3. 初始化动态外部感知 ---
        # 使用列表来存储刺激物，方便管理
        self.notifications: Dict[str,Any] = {}
        self.alerts: List[str] = []

        # --- 4. 初始化记忆 ---
        
        # 使用字典来存储不同来源的事件流/状态
        # 这是 WorldModel 的核心，用于存储各个 Cortex 维护的特定数据，纯内存操作
        self.cortex_data: Dict[str, BaseModel] = {} 

        self.short_term_memory: Deque[str] = deque(maxlen=short_term_memory_max_len)
        self.long_term_memory = None # 长期记忆占位符

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
        print(f"WorldModel: Cortex数据 '{key}' 已更新（内存中）。")


    def add_memory(self, action_summary: str):
        """将新的行动总结存入短期记忆。"""
        if not action_summary:
            return
        memory_entry = f"[{time.strftime('%H:%M:%S')}] {action_summary}"
        self.short_term_memory.append(memory_entry)
        print(f"WorldModel: 短期记忆已添加 - '{memory_entry}'")

    def update_stimuli(self, notification: Optional[str] = None, alert: Optional[str] = None):
        """由外部服务调用，用于更新外部世界的刺激物。"""
        if notification:
            self.notifications.append(notification)
            print(f"WorldModel: 收到新通知 - '{notification}'")
        if alert:
            self.alerts.append(alert)
            print(f"WorldModel: 收到新警报 - '{alert}'")

    def update_internal_state(self, mood: Optional[str] = None, energy_delta: Optional[int] = None):
        """更新 Agent 的内在状态。"""
        if mood:
            self.mood = mood
            print(f"WorldModel: 情绪更新为 -> {mood}")
        if energy_delta:
            self.energy = max(0, min(100, self.energy + energy_delta)) # 确保精力在0-100之间
            print(f"WorldModel: 精力变化 {energy_delta} -> 当前精力: {self.energy}")

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
                value_str = f"收到未读消息 {value} 条"
                entry = f"{key_str}\n{value_str}\n"
                result_parts.append(entry)
                
            # 将所有格式化后的部分连接起来
            # 使用 "" 作为分隔符，因为每个 entry 已经自带换行符
            notification_str = "未读消息列表：\n- " + "\n".join(result_parts).strip()
        else:
            notification_str = "当前没有未读消息：\n- " + "\n- ".join(self.notifications)
        
        alert_str = ""
        if self.alerts:
            alert_str = "注意！你收到了以下紧急警报：\n- " + "\n- ".join(self.alerts)
        
        # 清空已处理的刺激物
        self.notifications.clear()
        self.alerts.clear()

        # --- 格式化近期活动 ---
        action_summary_str = "\n".join(self.short_term_memory) if self.short_term_memory else "你最近没有活动。"

        context = {
            "bot_name": self.bot_name,
            "bot_identity": self.bot_identity,
            "bot_personality": self.bot_personality,
            "bot_interest": self.bot_interest, 
            "time": time.strftime('%Y年%m月%d日 %H:%M:%S 星期%A'),
            "mood": self.mood,
            "notifications": notification_str,
            "alert": alert_str or "当前没有紧急警报。",
            "action_summary": action_summary_str
        }
        return context
