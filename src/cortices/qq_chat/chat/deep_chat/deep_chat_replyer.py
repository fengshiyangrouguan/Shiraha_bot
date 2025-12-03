# src/cortices/qq_chat/replyer_planner.py
from typing import List, TYPE_CHECKING
from src.cortices.replyer_planner_base import BaseReplyerPlanner, ReplyIntent

if TYPE_CHECKING:
    from src.platform.platform_base import MessageSegment
    # 假设 qq_napcat 平台适配器在 platform/sources/qq_napcat/adapter.py 中定义
    from src.platform.sources.qq_napcat.adapter import QQNapcatAdapter 

class QQReplyerPlanner(BaseReplyerPlanner):
    """
    针对 QQ 平台的 Replyer Planner。
    使用 QQNapcatAdapter 的 Segment API 来构建消息。
    """
    def __init__(self, platform_adapter: "QQNapcatAdapter"):
        super().__init__(platform_adapter)

    async def plan(self, intent: ReplyIntent) -> List["MessageSegment"]:
        """
        根据回复意图构建 QQ 消息段列表。
        """
        segments: List["MessageSegment"] = []

        # 1. 处理 @ 用户
        for user_id in intent.at_users:
            # 假设 adapter 有 build_at_segment 方法
            if hasattr(self.platform_adapter, 'build_at_segment'):
                segments.append(self.platform_adapter.build_at_segment(user_id))
        
        # 2. 处理文本内容
        if intent.text:
            # 如果有 @ 的用户，在文本开头加一个空格，避免 @ 和文本粘连
            text_to_send = intent.text
            if intent.at_users:
                text_to_send = " " + text_to_send
            
            if hasattr(self.platform_adapter, 'build_text_segment'):
                segments.append(self.platform_adapter.build_text_segment(text_to_send))

        # 3. 处理图片
        for image_path_or_url in intent.images:
            # 假设 adapter 有 build_image_segment 方法
            if hasattr(self.platform_adapter, 'build_image_segment'):
                segments.append(self.platform_adapter.build_image_segment(image_path_or_url))

        # 可以在这里添加更多的 Segment 类型处理，例如语音、文件等

        return segments
