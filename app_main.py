import os
import asyncio
import sys
import mimetypes
import argparse
# add parent path to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_env():
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
        logger.error("请使用 Python3.10+ 运行本项目。")
        exit()

    os.makedirs("data/config", exist_ok=True)
    os.makedirs("data/plugins", exist_ok=True)
    os.makedirs("data/temp", exist_ok=True)


def main():

    

if __name__ == "__main__":
    check_env()

    # start log broker
    log_broker = LogBroker()
    LogManager.set_queue_handler(logger, log_broker)

    # check dashboard files
    webui_dir = asyncio.run(check_dashboard_files(args.webui_dir))

    db = db_helper

    # print logo


    # 依赖注入db，启动主程序
    core_lifecycle = InitialLoader(db, log_broker)
    asyncio.run(core_lifecycle.start())
