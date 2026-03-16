from __future__ import annotations

from typing import List, Tuple, Type

import httpx

from src.common.action_model.tool_result import ToolResult
from src.common.logger import get_logger
from src.plugin_system.base import BasePlugin, BaseTool, ToolInfo

logger = get_logger("weather_plugin")


def _weather_code_to_text(code: int) -> str:
    mapping = {
        0: "晴",
        1: "大体晴朗",
        2: "局部多云",
        3: "阴",
        45: "雾",
        48: "冻雾",
        51: "小毛毛雨",
        53: "毛毛雨",
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
        77: "雪粒",
        80: "小阵雨",
        81: "中阵雨",
        82: "大阵雨",
        85: "小阵雪",
        86: "大阵雪",
        95: "雷暴",
        96: "雷暴伴小冰雹",
        99: "雷暴伴大冰雹",
    }
    return mapping.get(code, f"未知天气(code={code})")


class GetCurrentWeatherTool(BaseTool):
    async def execute(self, location: str, unit: str = "celsius", **kwargs) -> ToolResult:
        location = (location or "").strip()
        if not location:
            return ToolResult(success=False, summary="天气查询失败：缺少地点。", error_message="location is required")

        normalized_unit = (unit or "celsius").strip().lower()
        if normalized_unit not in {"celsius", "fahrenheit"}:
            return ToolResult(
                success=False,
                summary="天气查询失败：温度单位无效。",
                error_message="unit must be celsius or fahrenheit",
            )

        timeout_seconds = float(self.get_config("weather.timeout_seconds", 12.0))
        temperature_unit = "celsius" if normalized_unit == "celsius" else "fahrenheit"
        speed_unit = "kmh" if normalized_unit == "celsius" else "mph"
        temperature_symbol = "°C" if normalized_unit == "celsius" else "°F"
        headers = {"User-Agent": "ShirahaBot-WeatherPlugin/1.0"}

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds, headers=headers) as client:
                geo_resp = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1, "language": "zh", "format": "json"},
                )
                geo_resp.raise_for_status()
                geo_data = geo_resp.json()

                results = geo_data.get("results") or []
                if not results:
                    return ToolResult(
                        success=False,
                        summary=f"天气查询失败：未找到地点“{location}”。",
                        error_message=f"place not found: {location}",
                    )

                place = results[0]
                city_name = place.get("name") or location
                country = place.get("country") or ""
                admin1 = place.get("admin1") or ""
                latitude = place.get("latitude")
                longitude = place.get("longitude")

                if latitude is None or longitude is None:
                    return ToolResult(
                        success=False,
                        summary=f"天气查询失败：地点“{location}”缺少有效坐标。",
                        error_message=f"missing coordinates for {location}",
                    )

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
                return ToolResult(
                    success=False,
                    summary="天气查询失败：返回数据缺少 current 字段。",
                    error_message="missing current field",
                )

            weather_code = int(current.get("weather_code", -1))
            weather_text = _weather_code_to_text(weather_code)
            temperature = current.get("temperature_2m")
            wind_speed = current.get("wind_speed_10m")
            humidity = current.get("relative_humidity_2m")
            observed_time = current.get("time") or "未知时间"

            place_parts = [part for part in [city_name, admin1, country] if part]
            full_place = ", ".join(place_parts)
            speed_symbol = "km/h" if speed_unit == "kmh" else "mph"
            summary = (
                f"已查询 {full_place} 的天气：{weather_text}，气温 {temperature}{temperature_symbol}，"
                f"湿度 {humidity}%，风速 {wind_speed} {speed_symbol}，观测时间 {observed_time}。"
            )

            logger.info(
                f"天气查询成功: location={location}, resolved={full_place}, unit={normalized_unit}"
            )
            return ToolResult(success=True, summary=summary)

        except httpx.HTTPStatusError as exc:
            logger.error(f"天气查询 HTTP 异常: {exc}", exc_info=True)
            return ToolResult(
                success=False,
                summary="天气查询失败：天气服务返回异常状态。",
                error_message=f"http status {exc.response.status_code}",
            )
        except httpx.RequestError as exc:
            logger.error(f"天气查询请求异常: {exc}", exc_info=True)
            return ToolResult(
                success=False,
                summary="天气查询失败：无法连接天气服务。",
                error_message="request error",
            )
        except Exception as exc:
            logger.error(f"天气查询异常: {exc}", exc_info=True)
            return ToolResult(
                success=False,
                summary="天气查询失败：执行过程中出现异常。",
                error_message=str(exc),
            )


class WeatherPlugin(BasePlugin):
    def get_plugin_tools(self) -> List[Tuple[ToolInfo, Type[BaseTool]]]:
        weather_info = self.get_declared_tool_info("get_current_weather")
        return [(weather_info, GetCurrentWeatherTool)]
