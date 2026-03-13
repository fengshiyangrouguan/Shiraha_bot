from dataclasses import dataclass, field
from typing import Any, Optional, List

from src.common.action_model.action_spec import ActionSpec

@dataclass
class ToolResult:
    """
    封装工具执行结果的标准数据结构。
    所有工具的 execute 方法都应返回此对象。
    """
    success: Optional[bool] = True
    """执行是否成功。"""

    summary: Optional[str] = ""
    """
    对结果的AI/人类可读的总结。
    这个总结主要用于：
    1. 在链式调用的下一步中，作为上下文提供给LLM。
    2. 在最终的长期记忆中，作为对该步骤的简短记录。
    3. 工具返回的核心的、结构化的数据。
        例如：
        - 读取文件工具，这里可能是文件内容字符串。
        - 发送消息工具，这里可能是包含消息ID的字典。
        - 一个计算工具，这里可能是计算结果（数字或字典）。
    """
    error_message: Optional[str] = None
    """如果 success 为 False，这里应包含具体的错误信息。"""

    follow_up_action: List[ActionSpec] = field(default_factory=list)
    """
    建议的后续动作。
    如果此字段不为 None，AgentLoop 将会继续执行链式调用。
    如果为 None，则表示当前行动链结束。
    """

    def add_action(self, action:ActionSpec):
        self,self.follow_up_action.append(action)
