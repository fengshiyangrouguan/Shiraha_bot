# src/agent/world_model.py
import time
from collections import deque
from typing import List, Dict, Any, Deque, Optional, Type
from pydantic import BaseModel # 导入 BaseModel 用于类型提示和校验

# 导入配置
from src.common.di.container import container
from src.common.config.schemas.bot_config import BotConfig
from src.common.logger import get_logger

logger = get_logger("world_model")

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
        # TODO: 以后来一个随机的初始动作
        # self.short_term_memory.append(f"[{time.strftime('%H:%M:%S')}] 我刚刚睡醒，感觉有点迷糊，需要一会儿才能完全进入状态。")

        self.long_term_memory = None # 长期记忆占位符

        # 心流缓存，缓存一些阅读，编程，等产生的感悟，形成侧回路，避免影响其他cortex
        self.flow_cache:Deque[str] = deque(maxlen=15)

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
        self.notifications[type]=notification
        logger.info(f"收到新通知 - '{notification}'")

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
            "action_summary": action_summary_str
        }
        return context
