# src/features/common_tools.py
from .manager import feature_manager

class CommonTools:
    """
    封装了 Agent 最常用、最基础的通用工具。
    """
    @feature_manager.tool(scope="main")
    async def web_search(self, query: str) -> str:
        """
        当需要获取关于某个主题的外部、实时信息时，使用此工具。
        
        Args:
            query (str): 要搜索的关键词或问题。
        
        Returns:
            str: 搜索结果的摘要。
        """
        print(f"  - [工具执行]: 正在执行 web_search(query='{query}')...")
        # 这是一个模拟实现
        # 未来的实现会在这里调用真正的搜索引擎API
        await asyncio.sleep(1) # 模拟网络延迟
        return f"关于 '{query}' 的模拟搜索结果：AI 领域最近发布了名为 'SuperModel' 的新模型，它在多项基准测试中表现出色。"

    @feature_manager.tool(scope="main")
    async def task_complete(self, summary: str) -> str:
        """
        当认为当前的高阶意图已经完全达成时，调用此工具来结束任务。
        
        Args:
            summary (str): 对整个任务完成情况的最终总结。
            
        Returns:
            str: 一个确认任务已完成的字符串。
        """
        print(f"  - [工具执行]: 任务完成，总结: '{summary}'")
        # 这个工具的返回值会成为 MainPlanner 的最终输出
        return summary

# 实例化类，以便在模块加载时，装饰器能够执行并注册工具
common_tools = CommonTools()
