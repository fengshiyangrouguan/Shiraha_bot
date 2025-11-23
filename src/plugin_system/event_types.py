from enum import Enum

class EventType(Enum):
    """
    事件类型枚举类，模仿 MaiBot 的事件类型
    """
    ON_START = "on_start"  # 启动事件
    ON_STOP = "on_stop"  # 停止事件
    ON_MESSAGE_PRE_PROCESS = "on_message_pre_process" # 消息预处理前
    ON_MESSAGE = "on_message" # 消息处理中
    ON_PLAN = "on_plan" # 规划前
    POST_LLM = "post_llm" # LLM处理后（结果未确定）
    AFTER_LLM = "after_llm" # LLM处理后（结果已确定）
    POST_SEND_PRE_PROCESS = "post_send_pre_process" # 发送前预处理
    POST_SEND = "post_send" # 发送中
    AFTER_SEND = "after_send" # 发送后

    # 系统内部事件
    PLATFORM_DISCONNECTED = "platform_disconnected" # 平台适配器断开连接事件
    PLATFORM_CONNECTED = "platform_connected" # 平台适配器连接成功事件
    
    UNKNOWN = "unknown"  # 未知事件类型

    def __str__(self) -> str:
        return self.value
