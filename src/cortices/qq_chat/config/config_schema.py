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
    # 移除 platform 字段，因为具体的平台配置将通过 napcat_adapters 列表提供
    
    adapter: ConfigSchema = Field(..., description="QQ Napcat 平台适配器的配置。")