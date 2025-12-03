# src/llm_api/dto.py
from typing import List, Dict, Optional, Any, Literal, Union
from pydantic import BaseModel

class ToolCall(BaseModel):
    """
    LLM 请求调用工具时，系统内部使用的标准化数据结构。
    """
    # call_id: str
    tool_name: str
    parameters: Optional[Dict[str, Any]] = None

class LLMMessage(BaseModel):
    """
    定义了与LLM交互的单条消息结构。
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        将 LLMMessage 对象转换为字典，并排除值为 None 的字段。
        """
        return self.model_dump(exclude_none=True)



class APIResponse(BaseModel):
    """
    对 LLM API 响应的内部标准化数据结构。
    无论底层是哪个 LLM，都应该被解析成这个格式。
    !!!暂时还没有使用!!!
    """
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[Dict[str, int]] = None

class LLMMessageBuilder:
    """
    一个辅助类，用于安全、方便地构建 LLMMessage 对象列表。
    """
    def __init__(self):
        self.messages: List[LLMMessage] = []

    def add_system_message(self, content: str):
        self.messages.append(LLMMessage(role="system", content=content))
        return self

    def add_user_message(self, content: Union[str, List[Dict[str, Any]]]):
        self.messages.append(LLMMessage(role="user", content=content))
        return self

    def add_assistant_message(self, content: str):
        self.messages.append(LLMMessage(role="assistant", content=content))
        return self

    def add_assistant_tool_calls(self, tool_calls: List[ToolCall]):
        self.messages.append(LLMMessage(role="assistant", content=None, tool_calls=tool_calls))
        return self

    def add_tool_response(self, tool_call_id: str, content: str):
        self.messages.append(LLMMessage(role="tool", tool_call_id=tool_call_id, content=content))
        return self
    
    def get_message_dict(self) -> List[Dict[str, Any]]:
        return [msg.model_dump(exclude_none=True) for msg in self.messages]
    
    def get_message_list(self) -> List[LLMMessage]:
        return self.messages