from pydantic import BaseModel, Field
from typing import Literal, List


class Chat(BaseModel):
    """
    定义聊天过滤（黑白名单）的配置
    """
    group_list_type: Literal["whitelist", "blacklist"] = Field(
        "whitelist", 
        description="群组名单类型。whitelist: 仅名单内可聊天; blacklist: 名单内不可聊天。"
    )
    group_list: List[int] = Field(
        [], 
        description="群组 QQ 号列表。"
    )
    
    private_list_type: Literal["whitelist", "blacklist"] = Field(
        "blacklist", 
        description="私聊名单类型。whitelist: 仅名单内可聊天; blacklist: 名单内不可聊天。"
    )
    private_list: List[int] = Field(
        [], 
        description="私聊用户 QQ 号列表。"
    )

class ConfigSchema(BaseModel):
    """
    QQ Napcat 平台适配器的配置 Schema。
    定义了每个 QQ Napcat 适配器实例所需的配置字段。
    """
    platform_type: Literal["qq_napcat"] = Field("qq_napcat", description="平台类型，固定为 'qq_napcat'。")
    adapter_id: str = Field(..., description="此 QQ Napcat 适配器实例的唯一标识符。")
    host: str = Field("127.0.0.1", description="Napcat HTTP API 服务的主机地址。")
    port: int = Field(8080, description="Napcat HTTP API 服务的端口。")
    chat: Chat = Field(default_factory=Chat, description="黑白名单及聊天对象配置")
    