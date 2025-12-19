import random
from typing import List, Dict, Any, Type, Union, Tuple
from src.plugin_system import (
    BasePlugin,
    BaseTool,
    ToolInfo,
    ConfigField,
)


class GetCurrentWeatherTool(BaseTool):
    """
    获取指定地点的当前天气信息。
    """
    name = "get_current_weather"
    description = "当用户询问某个地方的天气时，使用此工具。"
    parameters = [
        {
            "name": "location",
            "type": "string",
            "description": "要查询天气的城市名称",
            "required": True,
            "choices": None
        },
        {
            "name": "unit",
            "type": "string",
            "description": "温度单位",
            "required": False,
            "choices": None
        }
    ]

    async def execute(self, location: str, unit: str = "celsius") -> Dict[str, Any]:
        """
        模拟获取天气的实现。
        """
        print(f"[Debug] 天气工具被调用: location={location}, unit={unit}")
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

    # 配置Schema定义
    config_schema: dict = {
        "plugin": {
            "name": ConfigField(type=str, default="weather_plugin", description="插件名称"),
            "version": ConfigField(type=str, default="3.0.0", description="插件版本"),
            "enabled": ConfigField(type=bool, default=True, description="是否启用插件"),
        },
        "time": {"format": ConfigField(type=str, default="%Y-%m-%d %H:%M:%S", description="时间显示格式")},

    }

    def get_plugin_components(self) -> List[
        Union[
            Tuple[ToolInfo, Type[BaseTool]],
        ]
    ]:
        return [
            (GetCurrentWeatherTool.get_tool_info(), GetCurrentWeatherTool)
        ]

    def get_tools(self) -> List[Type[BaseTool]]:
        return [GetCurrentWeatherTool]
