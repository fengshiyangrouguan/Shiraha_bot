# src/agent/world_model.py
import time
from collections import deque
from typing import List, Dict, Any, Deque, Optional

# 导入通用配置服务和我们为 bot.toml 设计的 Schema
from src.system.di.container import container
from src.common.config.schemas.bot_config import BotConfig

class WorldModel:
    """
    世界模型 - Agent 状态管理器。
    负责管理 Agent 的所有状态，包括：
    1. 静态身份 (Identity)
    2. 动态内在状态 (Internal State: mood, energy)
    3. 动态外部感知 (External Perception: stimuli)
    4. 记忆 (Memory)
    """
    def __init__(self, short_term_memory_max_len: int = 12):
        print("WorldModel: 初始化...")
        
        # --- 1. 加载静态身份 ---

        bot_config:BotConfig = container.resolve(BotConfig)
        persona_config = bot_config.persona
        mood_config = bot_config.mood
        
        self.bot_name: str = persona_config.bot_name
        self.bot_identity: str = persona_config.bot_identity
        self.bot_personality: str = persona_config.bot_personality
        self.bot_interest: List[str] = persona_config.bot_interest

        # --- 2. 初始化动态内在状态 ---
        self.mood: str = mood_config.initial_mood
        self.energy: int = mood_config.initial_energy

        # --- 3. 初始化动态外部感知 ---
        # 使用列表来存储刺激物，方便管理
        self.notifications: List[str] = []
        self.alerts: List[str] = []

        # --- 4. 初始化记忆 ---
        self.short_term_memory: Deque[str] = deque(maxlen=short_term_memory_max_len)
        self.long_term_memory = None # 长期记忆占位符

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
        if self.notifications:
            notification_str = "你收到了以下新通知：\n- " + "\n- ".join(self.notifications)
        
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
            "bot_interst": ", ".join(self.bot_interest), 
            "time": time.strftime('%Y年%m月%d日 %H:%M:%S 星期%A'),
            "mood": self.mood,
            "notification": notification_str or "当前没有新通知。",
            "alert": alert_str or "当前没有紧急警报。",
            "action_summary": action_summary_str
        }
        return context
