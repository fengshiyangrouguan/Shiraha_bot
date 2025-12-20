from typing import List # 导入 List
from pydantic import Field
from src.cortices.cortex_config_loader import BaseCortexConfigSchema
from src.platform.sources.qq_napcat.config_schema import ConfigSchema # 导入 QQNapcatConfigSchema

class CortexConfigSchema(BaseCortexConfigSchema):
    """
    QQ Chat Cortex 的配置 Schema。
    定义了该 Cortex 所需的所有配置字段及其类型和默认值。
    """
    enable: bool = Field(True, description="是否启用 QQ Chat Cortex。")
    #TODO: 可能得改成int或str输入都变为str
    bot_id: str = Field(..., description="机器人在QQ平台上的唯一ID。")    
    adapter: ConfigSchema = Field(..., description="QQ Napcat 平台适配器的配置。")