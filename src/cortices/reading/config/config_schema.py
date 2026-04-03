from typing import List # 导入 List
from pydantic import Field
from src.cortex_system.cortex_config_loader import BaseCortexConfigSchema

class CortexConfigSchema(BaseCortexConfigSchema):
    """
    Reading Cortex 的配置 Schema。
    定义了该 Cortex 所需的所有配置字段及其类型和默认值。
    """
    enable: bool = Field(False, description="是否启用 Reading Cortex (当前禁用)")
