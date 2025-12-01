# run_cortices_test.py
import asyncio
import sys
from pathlib import Path

# 在导入项目模块之前，确保 src 目录在 Python 路径中
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# 导入所有需要的模块
from src.agent.world_model import WorldModel
from src.common.database.database_manager import DatabaseManager
from src.platform.platform_manager import PlatformManager
from src.cortices.manager import CortexManager
from src.system.di.container import container
from src.common.config.config_service import ConfigService
from src.common.config.schemas.bot_config import BotConfig
from src.common.config.schemas.llm_api_config import LLMApiConfig
from src.llm_api.factory import LLMRequestFactory

async def main():
    """
    主函数，通过 CortexManager 设置和运行所有可用的 Cortex 进行联调测试。
    """
    print("--- 启动 Cortex 联调测试环境 ---")

    db_manager = None
    cortex_manager = None

    try:
        # 1. 初始化核心服务和依赖注入容器
        print("步骤 1: 初始化核心服务...")
        
        # 加载主配置文件
        config_service = ConfigService()
        bot_config = config_service.get_config("bot")
        llm_api_config = config_service.get_config("llm_api")
        
        # 初始化数据库和平台管理器
        db_manager = DatabaseManager()
        await db_manager.initialize_database(echo=False) # echo=True 可查看 SQL
        
        platform_manager = PlatformManager()
        llm_request_factory = LLMRequestFactory()

        
        # 创建 WorldModel
        
        container.register_instance(BotConfig, instance=bot_config)
        container.register_instance(LLMApiConfig, instance=llm_api_config)
        container.register_instance(DatabaseManager, instance=db_manager)
        container.register_instance(PlatformManager, instance=platform_manager)
        container.register_instance(LLMRequestFactory, instance=llm_request_factory)
        world_model = WorldModel()
        container.register_instance(WorldModel, instance=world_model)
        
        print("核心服务和DI容器已准备就绪。")

        # 2. 初始化并启动 CortexManager
        print("步骤 2: 启动 CortexManager 并加载所有 Cortex...")
        cortex_manager = CortexManager()
        await cortex_manager.load_all_cortices()
        
        # 3. 持续运行并等待外部事件
        print("\n--- 测试环境正在运行 ---")
        print("所有适配器已启动，正在等待外部事件...")
        print("按 Ctrl+C 停止测试。")
        
        # 使用一个永不被设置的 Event 来无限期等待
        stop_event = asyncio.Event()
        await stop_event.wait()

    except KeyboardInterrupt:
        print("\n收到停止信号...")
    except Exception as e:
        print(f"\n在运行过程中发生致命错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 4. 优雅关闭
        print("\n--- 开始执行优雅关闭 ---")
        if cortex_manager:
            await cortex_manager.shutdown_all_cortices()
        if db_manager:
            await db_manager.shutdown()
        print("--- 测试环境已关闭 ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被强制退出。")
