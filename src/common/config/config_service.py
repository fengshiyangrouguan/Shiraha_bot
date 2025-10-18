import toml
from pathlib import Path
from typing import Any, Dict, Union
import logging
from pydantic import ValidationError

# 导入配置 Schema (假设这个路径在您的项目中是有效的)
from backend.config_schema import AppConfig

logger = logging.getLogger("ConfigService")
# 假设项目的根目录是当前文件的三级父目录
current_file_path = Path(__file__).resolve()

# 根据您的约定路径定义配置文件的相对位置
CONFIG_FILE_PATH = Path("configs/config.toml")
ROOT_PATH = current_file_path.parent.parent.parent

class ConfigService:
    """
    提供应用配置的读写和持久化服务，基于TOML文件和Pydantic Schema。
    """
    
    def __init__(self, config_path: Path = CONFIG_FILE_PATH):
        self.config_path = config_path

        # 存储 Pydantic 模型的实例
        self._config: AppConfig 
        self._load_config()

    def _load_config(self):
        """加载配置文件，通过Pydantic验证并设置默认值。"""
        loaded_data: Dict[str, Any] = {}
        
        # 确保配置文件路径是绝对路径（如果 CONFIG_FILE_PATH 是相对路径）
        # 假设 config.toml 文件位于与 service.py 同级的根目录或可以通过相对路径访问
        
        if self.config_path.exists():
            try:

                with open(self.config_path, mode="r", encoding='utf-8') as f:
                    loaded_data = toml.load(f)
            except Exception as e:
                logger.error(f"加载 TOML 配置文件失败: {e}。将使用默认配置或部分配置。")

        try:
            # Pydantic V2 推荐使用 model_validate 或 model_construct
            # model_validate 会执行完整的验证和转换
            self._config = AppConfig.model_validate(loaded_data)
            logger.info("配置加载成功。")
            
        except ValidationError as e:
            logger.error(f"配置 Schema 验证失败: {e}。将尝试使用有效部分并用默认值填充无效部分。")
            
            # 即使验证失败，我们仍尝试使用 Pydantic 实例化，让其用默认值填充
            # Pydantic model_validate 已经尝试最大程度地恢复，但为了记录，可以再次尝试
            try:
                 self._config = AppConfig.model_validate(loaded_data)
            except Exception:
                 self._config = AppConfig() # 致命错误，退回完全默认
            
            self._save_config() # 建议保存一次，用默认值修复配置文件
            
        except Exception as e:
           # 如果是其他严重错误，退回到完全默认配置
           logger.error(f"配置加载发生未预期的错误: {e}。退回到完全默认配置。")
           self._config = AppConfig()

    def _save_config(self):
        """持久化当前配置模型到文件。"""
        # 确保配置文件的父目录存在
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Pydantic V2 推荐使用 model_dump 方法
            data_to_save = self._config.model_dump(by_alias=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as configfile:
                # 使用 toml 库将字典保存为 TOML 文件
                toml.dump(data_to_save, configfile)
            logger.info("TOML 配置已保存到文件。")
        except Exception as e:
            logger.error(f"保存 TOML 配置失败: {e}")
            
    # 公共 API 变得更简洁和类型安全
    
    def get(self) -> AppConfig:
        """返回整个配置模型实例。"""
        return self._config

    def set_and_save(self, section: str, option: str, value: Any):
        """
        在内存中设置配置项，通过 Pydantic 验证后，立即持久化到文件。
        为了确保类型安全和验证，最可靠的做法是创建一个新模型并验证。
        """
        try:
            # 1. 将当前配置转换为字典
            current_data = self._config.model_dump(by_alias=True)
            
            # 2. 修改目标值
            if section not in current_data:
                raise AttributeError(f"配置段 '{section}' 不存在。")

            # 检查子模型是否存在（例如 device, diagnosis）
            if not isinstance(current_data[section], dict):
                # 如果配置段不是字典（例如它是一个顶级字段，但这不适用于您的 Schema）
                raise AttributeError(f"配置段 '{section}' 结构不正确。")

            current_data[section][option] = value
            
            # 3. 核心：使用修改后的数据重新创建模型，触发 Pydantic 验证
            new_config = AppConfig.model_validate(current_data)
            self._config = new_config # 验证成功，更新内存中的配置
            
            # 4. 持久化到文件
            self._save_config()
            logger.info(f"配置已更新并保存: [{section}] {option} = {value}")
            
        except AttributeError as e:
            logger.error(f"{e}")
        except ValidationError as e:
            logger.error(f"新值 '{value}' (用于 [{section}].{option}) 未通过 Pydantic 验证: {e.errors()}")
        except Exception as e:
            logger.error(f"配置设置失败: {e}")