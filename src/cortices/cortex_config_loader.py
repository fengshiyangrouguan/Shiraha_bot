from pathlib import Path
from typing import Type, Any, Dict

import toml
from pydantic import BaseModel, ValidationError

class BaseCortexConfigSchema(BaseModel):
    """
    所有 Cortex 配置的基础 Schema。
    个别 Cortex 应该继承此 Schema 来定义其特有的配置字段。
    """
    # 可以在此处添加所有 Cortex 通用的配置字段
    pass

def load_cortex_config(cortex_dir: Path, schema_model: Type[BaseModel]) -> BaseModel:
    """
    加载并验证给定 Cortex 目录的 config.toml 配置文件。

    Args:
        cortex_dir (Path): Cortex 的根目录路径。
        schema_model (Type[BaseModel]): 用于验证配置的 Pydantic Schema 模型。

    Returns:
        BaseModel: 经过加载和验证的配置对象实例。

    Raises:
        FileNotFoundError: 如果未找到 config/config.toml 文件。
        ValidationError: 如果配置文件不符合提供的 Schema 定义。
        Exception: 发生其他解析或加载错误时。
    """
    config_file_path = cortex_dir / "config" / "config.toml"
    if not config_file_path.exists():
        raise FileNotFoundError(f"Cortex 配置文件未找到: {config_file_path}")

    try:
        config_data: Dict[str, Any] = toml.load(config_file_path)
        validated_config = schema_model.model_validate(config_data)
        return validated_config
    except ValidationError as e:
        raise ValueError(
            f"Cortex 配置文件验证失败，路径: {config_file_path}\n{e}"
        ) from e
    except Exception as e:
        raise Exception(
            f"Cortex 配置时发生未知错误，路径: {config_file_path}\n{e}"
        ) from e

