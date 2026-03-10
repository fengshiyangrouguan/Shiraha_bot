import random
from typing import List, Dict, Any, Type, Tuple
from src.plugin_system import (
    BasePlugin,
    BaseTool,
    ToolInfo,
)


class GetCurrentWeatherTool(BaseTool):
    """
    获取指定地点的当前天气信息。
    """


    async def execute(self, location: str, unit: str = "celsius") -> Dict[str, Any]:
        """
        模拟获取天气的实现。
        """
        temperatures = {
            "北京": 25,
            "上海": 28,
            "东京": 22,
        }
        temperature = temperatures.get(location, random.randint(15, 30))

        return {"result":{
            "location": location,
            "temperature": f"{temperature}°{unit[0].upper()}",
            "condition": random.choice(["晴", "多云", "小雨"])}
        }


class WeatherPlugin(BasePlugin):
    """
    一个提供天气相关工具的插件。
    """
    plugin_name: str = "weather_plugin"  # 内部标识符
    enable_plugin: bool = True
    dependencies: List[str] = []  # 插件依赖列表
    python_dependencies: List[str] = []  # Python包依赖列表
    config_file_name: str = "config.toml"  # 配置文件名

    def get_plugin_tools(self) -> List[Tuple[ToolInfo, Type[BaseTool]]]:
        return [
            (self.get_declared_tool_info("get_current_weather"), GetCurrentWeatherTool)
        ]

    def get_tools(self) -> List[Type[BaseTool]]:
        return [GetCurrentWeatherTool]
