# src/cortices/base_cortex.py
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.agent.world_model import WorldModel
    from src.cortices.manager import CortexManager

class BaseCortex(ABC):
    """
    Cortex (皮层) 的抽象基类。
    每个 Cortex 都是一个独立的、可管理的“应用”或“插件”，拥有自己的生命周期。
    """

    @abstractmethod
    async def setup(self, world_model: 'WorldModel', config: BaseModel, cortex_manager: 'CortexManager'):
        """
        初始化并启动该 Cortex。
        这个方法在系统启动或热加载该 Cortex 时被调用。

        Args:
            world_model: WorldModel 的实例，以便 Cortex 可以感知和修改世界状态。
            config: 当前 Cortex 经过验证的配置对象（由 config.toml 加载并经 Pydantic Schema 验证）。
            cortex_manager: CortexManager 的实例，用于注册工具或执行其他交互。
        """
        pass

    @abstractmethod
    async def teardown(self):
        """
        停止并清理该 Cortex。
        这个方法在系统关闭或热卸载该 Cortex 时被调用。
        """
        pass

    @abstractmethod
    def get_tools(self) -> Dict[str, Any]:
        """
        返回该 Cortex 提供的工具列表。
        在 Cortex 内部应该实例化并返回工具对象。
        这些工具将被注册到 CortexManager 中，以供系统其他部分使用。

        Returns:
            Dict[str, Any]: 工具名称到工具实例的映射字典。
        """
        pass