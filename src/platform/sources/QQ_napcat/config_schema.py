from pydantic import BaseModel, Field
from typing import Literal

class ConfigSchema(BaseModel):
    """
    QQ Napcat 平台适配器的配置 Schema。
    定义了每个 QQ Napcat 适配器实例所需的配置字段。
    """
    platform_type: Literal["qq_napcat"] = Field("qq_napcat", description="平台类型，固定为 'qq_napcat'。")
    adapter_id: str = Field(..., description="此 QQ Napcat 适配器实例的唯一标识符。")
    host: str = Field("127.0.0.1", description="Napcat HTTP API 服务的主机地址。")
    port: int = Field(8080, description="Napcat HTTP API 服务的端口。")
