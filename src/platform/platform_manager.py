import asyncio
import importlib
import os
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List

# 从当前包导入基类和类型定义
from .platform_base import PlatformAdapterBase, PostMethod

# 假设日志和配置系统已经存在
import logging
logger = logging.getLogger(__name__)


class PlatformManager:
    """
    平台管理器。
    负责动态加载、初始化、运行和管理所有的平台适配器。
    """

    def __init__(self, post_method: PostMethod, platform_configs: List[Dict[str, Any]]):
        """
        初始化平台管理器。

        :param post_method: EventManager 的 post 方法，将传递给所有适配器。
        :param platform_configs: 一个包含所有平台配置的列表。
        """
        self.post_method = post_method
        self.platform_configs = platform_configs
        self.adapters: Dict[str, PlatformAdapterBase] = {}
        self._adapter_tasks: List[asyncio.Task] = []

    def _register_adapter_class(self, platform_name: str) -> Any:
        """
        根据平台名称动态注册适配器类。
        它会在 'sources' 目录下查找对应的模块。

        :param platform_name: 平台名称，应与 'sources' 下的模块目录名一致。
        :return: 加载到的适配器类。
        """
        try:
            # 构建模块的导入路径，例如：Shiraha_bot.src.platform.sources.onebot_v11.adapter
            module_path = f"Shiraha_bot.src.platform.sources.{platform_name}.adapter"
            
            # 使用 importlib 动态导入模块
            adapter_module = importlib.import_module(module_path)
            
            # 在模块中寻找继承自 PlatformAdapterBase 的类
            for attr_name in dir(adapter_module):
                attr = getattr(adapter_module, attr_name)
                if isinstance(attr, type) and issubclass(attr, PlatformAdapterBase) and attr is not PlatformAdapterBase:
                    logger.info(f"成功找到适配器类 '{attr.__name__}' 于模块 '{module_path}'")
                    return attr
            
            raise ImportError(f"在模块 '{module_path}' 中未找到适配器类。")
        except (ImportError, AttributeError) as e:
            logger.error(f"加载平台 '{platform_name}' 的适配器失败: {e}")
            raise

    def load_adapters(self):
        """
        根据配置加载所有启用的平台适配器。
        """
        logger.info("开始加载平台适配器...")
        for config in self.platform_configs:
            platform_name = config.get("name")
            is_enabled = config.get("enabled", False)
            
            if not platform_name:
                logger.warning(f"发现一个没有 'name' 字段的平台配置，已跳过。")
                continue

            if not is_enabled:
                logger.info(f"平台 '{platform_name}' 未启用，已跳过。")
                continue

            try:
                AdapterClass = self._register_adapter_class(platform_name)
                # 在这里，我们将 post_method 注入到适配器实例中
                adapter_instance = AdapterClass(post_method=self.post_method, platform_config=config)
                
                adapter_id = config.get("id")
                if not adapter_id:
                    logger.error(f"平台 '{platform_name}' 的配置中缺少 'id' 字段，加载失败。")
                    continue

                self.adapters[adapter_id] = adapter_instance
                logger.info(f"成功加载并实例化了适配器 '{adapter_id}' (平台: {platform_name})。")

            except Exception as e:
                logger.error(f"处理平台 '{platform_name}' 时发生错误: {e}")

    def start_all(self):
        """
        启动所有已加载的适配器。
        """
        if not self.adapters:
            logger.warning("没有任何已加载的适配器可供启动。")
            return

        logger.info("正在启动所有平台适配器...")
        for adapter_id, adapter in self.adapters.items():
            try:
                task = adapter.run()
                self._adapter_tasks.append(task)
                logger.info(f"适配器 '{adapter_id}' 已成功启动。")
            except Exception as e:
                logger.error(f"启动适配器 '{adapter_id}' 失败: {e}")

    async def stop_all(self):
        """
        优雅地停止所有正在运行的适配器。
        """
        if not self._adapter_tasks:
            return

        logger.info("正在停止所有平台适配器...")
        # 触发所有适配器的终止方法
        await asyncio.gather(*(adapter.terminate() for adapter in self.adapters.values()), return_exceptions=True)
        
        # 等待所有任务完成
        await asyncio.gather(*self._adapter_tasks, return_exceptions=True)
        logger.info("所有平台适配器已停止。")

    def get_adapter(self, adapter_id: str) -> PlatformAdapterBase | None:
        """
        通过ID获取一个已加载的适配器实例。

        :param adapter_id: 适配器的唯一ID。
        :return: 适配器实例，如果不存在则返回 None。
        """
        return self.adapters.get(adapter_id)