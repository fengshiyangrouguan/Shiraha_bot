# src/common/config/config_service.py
import toml
from pathlib import Path
from typing import Type, TypeVar, Dict, Optional

from pydantic import BaseModel, ValidationError

# 导入所有需要被管理的配置 Schema
from .schemas.llm_api_config import LLMApiConfig
from .schemas.bot_config import BotConfig

# 使用 TypeVar 来获得更精确的类型提示
T = TypeVar('T', bound=BaseModel)

class ConfigService:
    """
    一个通用的、带缓存的配置加载服务。
    通过一个集中的注册表来管理不同的配置文件，调用者使用逻辑名称获取配置。

    可获取config：
        llm_api，bot，
    """
    _instance = None
    _cache: Dict[str, BaseModel] = {}

    # 配置注册表: '逻辑名称' -> ('文件路径', Schema类)
    _config_registry: Dict[str, tuple[str, Type[BaseModel]]] = {
        "llm_api": ("configs/llm_api_config.toml", LLMApiConfig),
        "bot": ("configs/bot_config.toml", BotConfig),
        # 未来若有新配置，在此处添加即可
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigService, cls).__new__(cls)
        return cls._instance

    def get_config(self, name: str) -> T:
        """
        根据逻辑名称加载、验证并缓存一个配置文件。

        Args:
            name (str): 在配置注册表中定义的逻辑名称 (e.g., "llm_api", "bot").

        Returns:
            一个经过验证的 Pydantic 模型实例。

        Raises:
            RuntimeError: 如果配置名称未注册、文件未找到或文件内容不符合 Schema。
        """
        if name in self._cache:
            # 直接从缓存返回
            return self._cache[name]  # type: ignore

        if name not in self._config_registry:
            raise RuntimeError(f"未注册的配置名称: '{name}'。请在 ConfigService 的 _config_registry 中定义它。")

        config_path_str, schema_class = self._config_registry[name]
        config_path = Path(config_path_str)

        if not config_path.exists():
            raise RuntimeError(f"配置文件未找到: {config_path.resolve()}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = toml.load(f)
            
            # 使用 Pydantic 进行验证和解析
            validated_config = schema_class.model_validate(data)
            
            # 存入缓存
            self._cache[name] = validated_config
            
            return validated_config  # type: ignore

        except ValidationError as e:
            raise RuntimeError(f"配置文件 '{config_path_str}' (用于 '{name}') 格式错误:\n{e}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件 '{config_path_str}' (用于 '{name}') 时发生未知错误: {e}")

    def clear_cache(self, name: Optional[str] = None):
        """
        清空配置缓存。
        如果不提供名称，则清空所有缓存；否则只清空指定名称的缓存。
        """
        if name:
            if name in self._cache:
                del self._cache[name]
        else:
            self._cache.clear()