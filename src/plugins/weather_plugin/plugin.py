from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type

import httpx

from src.common.logger import get_logger
from src.plugin_system.base import BasePlugin, BaseTool, ToolInfo

logger = get_logger("weather_plugin")


def _weather_code_to_text(code: int) -> str:
	"""将 Open-Meteo 天气代码转换为中文描述。"""
	mapping = {
		0: "晴朗",
		1: "基本晴",
		2: "局部多云",
		3: "阴天",
		45: "有雾",
		48: "冻雾",
		51: "小毛毛雨",
		53: "中毛毛雨",
		55: "大毛毛雨",
		56: "小冻毛毛雨",
		57: "大冻毛毛雨",
		61: "小雨",
		63: "中雨",
		65: "大雨",
		66: "小冻雨",
		67: "大冻雨",
		71: "小雪",
		73: "中雪",
		75: "大雪",
		77: "冰粒",
		80: "小阵雨",
		81: "中阵雨",
		82: "强阵雨",
		85: "小阵雪",
		86: "强阵雪",
		95: "雷暴",
		96: "雷暴夹小冰雹",
		99: "雷暴夹大冰雹",
	}
	return mapping.get(code, f"未知天气(code={code})")


class GetCurrentWeatherTool(BaseTool):
	"""查询指定城市当前天气。"""

	async def execute(self, location: str, unit: str = "celsius", **kwargs) -> Dict[str, Any]:
		location = (location or "").strip()
		if not location:
			return {"error": "location 不能为空"}

		normalized_unit = (unit or "celsius").strip().lower()
		if normalized_unit not in {"celsius", "fahrenheit"}:
			return {"error": "unit 仅支持 celsius 或 fahrenheit"}

		timeout_seconds = float(self.get_config("weather.timeout_seconds", 12.0))
		temperature_unit = "celsius" if normalized_unit == "celsius" else "fahrenheit"
		speed_unit = "kmh" if normalized_unit == "celsius" else "mph"
		temperature_symbol = "°C" if normalized_unit == "celsius" else "°F"

		headers = {"User-Agent": "ShirahaBot-WeatherPlugin/1.0"}

		try:
			async with httpx.AsyncClient(timeout=timeout_seconds, headers=headers) as client:
				# 1) 用城市名做地理编码
				geo_resp = await client.get(
					"https://geocoding-api.open-meteo.com/v1/search",
					params={"name": location, "count": 1, "language": "zh", "format": "json"},
				)
				geo_resp.raise_for_status()
				geo_data = geo_resp.json()

				results = geo_data.get("results") or []
				if not results:
					return {"error": f"未找到城市：{location}"}

				place = results[0]
				city_name = place.get("name") or location
				country = place.get("country") or ""
				admin1 = place.get("admin1") or ""
				latitude = place.get("latitude")
				longitude = place.get("longitude")

				if latitude is None or longitude is None:
					return {"error": f"城市坐标缺失：{location}"}

				# 2) 查询当前天气
				weather_resp = await client.get(
					"https://api.open-meteo.com/v1/forecast",
					params={
						"latitude": latitude,
						"longitude": longitude,
						"current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
						"temperature_unit": temperature_unit,
						"wind_speed_unit": speed_unit,
						"timezone": "auto",
					},
				)
				weather_resp.raise_for_status()
				weather_data = weather_resp.json()

			current = weather_data.get("current")
			if not isinstance(current, dict):
				return {"error": "天气接口返回异常：缺少 current 字段"}

			weather_code = int(current.get("weather_code", -1))
			weather_text = _weather_code_to_text(weather_code)
			temperature = current.get("temperature_2m")
			wind_speed = current.get("wind_speed_10m")
			humidity = current.get("relative_humidity_2m")
			observed_time = current.get("time") or "未知时间"

			place_parts = [p for p in [city_name, admin1, country] if p]
			full_place = ", ".join(place_parts)
			speed_symbol = "km/h" if speed_unit == "kmh" else "mph"

			summary = (
				f"{full_place} 当前天气：{weather_text}，温度 {temperature}{temperature_symbol}，"
				f"湿度 {humidity}% ，风速 {wind_speed} {speed_symbol}（观测时间 {observed_time}）。"
			)

			logger.info(f"天气查询成功: location={location}, resolved={full_place}, unit={normalized_unit}")
			return {"result": summary}

		except httpx.HTTPStatusError as exc:
			logger.error(f"天气接口 HTTP 错误: {exc}", exc_info=True)
			return {"error": f"天气接口请求失败（HTTP {exc.response.status_code}）"}
		except httpx.RequestError as exc:
			logger.error(f"天气接口网络错误: {exc}", exc_info=True)
			return {"error": "天气接口网络异常，请稍后重试"}
		except Exception as exc:
			logger.error(f"天气查询失败: {exc}", exc_info=True)
			return {"error": f"天气查询失败: {exc}"}


class WeatherPlugin(BasePlugin):
	"""天气查询插件。"""

	def get_plugin_tools(self) -> List[Tuple[ToolInfo, Type[BaseTool]]]:
		weather_info = self.get_declared_tool_info("get_current_weather")
		return [(weather_info, GetCurrentWeatherTool)]

