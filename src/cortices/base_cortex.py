from __future__ import annotations
import inspect
import importlib
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, List, Type
from src.cortices.tools_base import BaseTool

from pydantic import BaseModel
from src.common.logger import get_logger

if TYPE_CHECKING:
    from src.agent.world_model import WorldModel
    from src.cortices.manager import CortexManager


logger = get_logger("cortex")

class BaseCortex(ABC):
    """
    Cortex (皮层) 的抽象基类。
    每个 Cortex 都是一个独立的、可管理的“应用”或“插件”，拥有自己的生命周期。
    这个基类提供了通用的工具动态加载和依赖注入机制。
    """

    def __init__(self):
        self._world_model: WorldModel | None = None
        self._cortex_manager: CortexManager | None = None
        self.config: BaseModel | None = None

    async def setup(self, world_model: 'WorldModel', config: BaseModel, cortex_manager: 'CortexManager'):
        """
        初始化并启动该 Cortex。
        这个方法在系统启动或热加载该 Cortex 时被调用。
        它会保存核心依赖，子类可以重写此方法以添加自己的启动逻辑。
        """
        self._world_model = world_model
        self.config = config
        self._cortex_manager = cortex_manager
        logger.info(f"Cortex '{self.__class__.__name__}' setup complete.")

    @abstractmethod
    async def teardown(self):
        """
        停止并清理该 Cortex。
        这个方法在系统关闭或热卸载该 Cortex 时被调用。
        """
        pass
    

    @abstractmethod
    async def get_cortex_summary(self) -> str:
        """
        返回该皮层的现状摘要。
        例如 QQ 会话：“有 3 个活跃会话，其中 A 群有人提问，B 私聊有待办。”
        """
        pass


    def get_tools(self) -> List['BaseTool']:
        """
        动态实例化并返回此 Cortex 提供的所有工具。
        它会自动扫描子类所在目录下的 'tools' 子目录，并对发现的工具进行依赖注入。
        """
        if not self._world_model or not self._cortex_manager:
            raise RuntimeError(f"Cortex '{self.__class__.__name__}'尚未完全初始化(setup未被调用)，无法获取工具。")

        loaded_tools: List['BaseTool'] = []
        # 使用 inspect 确保路径相对于子类文件，而不是这个基类文件
        try:
            subclass_file = inspect.getfile(self.__class__)
            tools_dir = Path(subclass_file).parent / "tools"
        except TypeError:
            # 如果在交互式环境或内存中定义的类，则无法找到文件路径
            logger.warning(f"无法确定 Cortex '{self.__class__.__name__}' 的文件路径，跳过动态工具加载。")
            return []

        if not tools_dir.exists() or not tools_dir.is_dir():
            logger.info(f"Cortex '{self.__class__.__name__}' 没有 'tools' 目录，不加载任何工具。")
            return []
            
        # 准备可注入的依赖
        available_dependencies = {
            "world_model": self._world_model,
            "cortex_manager": self._cortex_manager,
            "cortex": self,  # 注入自身实例
        }
        # 有些cortex有adapter，有些没有，动态添加
        if hasattr(self, 'adapter'):
            available_dependencies['adapter'] = getattr(self, 'adapter')
        if hasattr(self, 'llm_request_factory'):
            available_dependencies['llm_request_factory'] = getattr(self, 'llm_request_factory')
        if hasattr(self, 'database_manager'):
            available_dependencies['database_manager'] = getattr(self, 'database_manager')

        for tool_file in tools_dir.iterdir():
            if tool_file.suffix == ".py" and not tool_file.name.startswith("_"):
                # e.g., self.__module__ is 'src.cortices.qq_chat.cortex'
                # We need the parent package 'src.cortices.qq_chat'
                parent_package = self.__module__.rsplit('.', 1)[0]
                relative_module_name = f".tools.{tool_file.stem}"
                try:
                    # 使用相对导入，这在处理复杂的包结构时更健壮
                    tool_module = importlib.import_module(relative_module_name, package=parent_package)
                    
                    for name, obj in inspect.getmembers(tool_module):
                        if inspect.isclass(obj) and issubclass(obj, BaseTool) and obj is not BaseTool:
                            tool_class: Type['BaseTool'] = obj
                            
                            # 动态依赖注入
                            try:
                                sig = inspect.signature(tool_class.__init__)
                                params_to_inject = {}
                                for param_name, param in sig.parameters.items():
                                    if param_name == "self":
                                        continue
                                    if param_name in available_dependencies:
                                        params_to_inject[param_name] = available_dependencies[param_name]
                                    elif param.default is inspect.Parameter.empty:
                                        logger.warning(f"工具 '{tool_class.__name__}' 在模块 '{relative_module_name}' 中缺少必要的依赖 '{param_name}'，无法实例化。")
                                        params_to_inject = None 
                                        break
                                
                                if params_to_inject is not None:
                                    tool_instance = tool_class(**params_to_inject)
                                    loaded_tools.append(tool_instance)
                                    logger.info(f"在 '{self.__class__.__name__}' 中动态加载并实例化工具: {tool_class.__name__}")

                            except TypeError as te:
                                logger.error(f"实例化工具 '{tool_class.__name__}' 时依赖注入失败: {te}", exc_info=True)
                            except Exception as e:
                                logger.error(f"实例化工具 '{tool_class.__name__}' 时发生未知错误: {e}", exc_info=True)
                            break 
                except Exception as e:
                    logger.error(f"加载工具模块 '{relative_module_name}' (包: {parent_package}) 失败: {e}", exc_info=True)
        return loaded_tools