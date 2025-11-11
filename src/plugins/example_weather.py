# src/plugins/example_weather.py
import random
from typing import List, Dict, Any, Type

# 假设 base.py 在 src/plugin_system/base.py
from src.plugin_system.base import BasePlugin, BaseTool

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
            "description": "城市名，例如：北京",
            "required": True
        },
        {
            "name": "unit",
            "type": "string",
            "description": "温度单位，可以是 'celsius' 或 'fahrenheit'",
            "required": False
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
        
        return {
            "location": location,
            "temperature": f"{temperature}°{unit[0].upper()}",
            "condition": random.choice(["晴", "多云", "小雨"])
        }

class WeatherPlugin(BasePlugin):
    """
    一个提供天气相关工具的插件。
    """
    def get_tools(self) -> List[Type[BaseTool]]:
        return [GetCurrentWeatherTool]
