# src/plugins/rpg_plugin/sub_agent/state_manager.py
import os
import csv
from typing import Dict, Any, List
from src.common.logger import get_logger

logger = get_logger("rpg_state_manager")

# 定义数据文件路径
PLUGIN_DATA_DIR = os.path.join("data", "rpg_plugin")
PLAYER_STATUS_FILE = os.path.join(PLUGIN_DATA_DIR, "player_status.csv")
WORLD_STATE_FILE = os.path.join(PLUGIN_DATA_DIR, "world_state.csv")
RULES_FILE = os.path.join("src", "plugins", "rpg_plugin", "skill.md")

class RPGStateManager:
    """
    负责跑团插件所有状态的读取和写入。
    """
    def __init__(self):
        self.log_prefix = "[RPGStateManager]"
        # 预先加载字段名，或在加载时动态确定
        self.player_fieldnames = ['player_id', 'hp', 'location', 'description']
        self.world_fieldnames = ['key', 'value', 'description']

    def get_rules(self) -> str:
        """读取并返回规则文件内容。"""
        try:
            with open(RULES_FILE, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"{self.log_prefix} Rules file not found: {RULES_FILE}")
            return "No rules found."
        except Exception as e:
            logger.error(f"{self.log_prefix} Error reading rules file: {e}", exc_info=True)
            return "Error loading rules."

    def load_player_statuses(self) -> Dict[str, Dict[str, Any]]:
        """加载玩家状态CSV文件。"""
        return self._read_csv_to_dict(PLAYER_STATUS_FILE, 'player_id')

    def load_world_state(self) -> Dict[str, Dict[str, Any]]:
        """加载世界状态CSV文件。"""
        return self._read_csv_to_dict(WORLD_STATE_FILE, 'key')

    def save_player_statuses(self, data: Dict[str, Dict[str, Any]]):
        """保存玩家状态到CSV文件。"""
        self._write_dict_to_csv(PLAYER_STATUS_FILE, data, self.player_fieldnames)

    def save_world_state(self, data: Dict[str, Dict[str, Any]]):
        """保存世界状态到CSV文件。"""
        self._write_dict_to_csv(WORLD_STATE_FILE, data, self.world_fieldnames)

    def _read_csv_to_dict(self, file_path: str, key_column: str) -> Dict[str, Dict[str, Any]]:
        """通用CSV读取方法。"""
        data = {}
        try:
            with open(file_path, mode='r', encoding='utf-8', newline='') as infile:
                reader = csv.DictReader(infile)
                # 更新字段名以防万一
                if key_column in self.player_fieldnames:
                    self.player_fieldnames = reader.fieldnames
                elif key_column in self.world_fieldnames:
                    self.world_fieldnames = reader.fieldnames
                
                for row in reader:
                    key = row[key_column]
                    data[key] = row
        except FileNotFoundError:
            logger.warning(f"{self.log_prefix} CSV file not found: {file_path}")
        except Exception as e:
            logger.error(f"{self.log_prefix} Error reading CSV {file_path}: {e}", exc_info=True)
        return data

    def _write_dict_to_csv(self, file_path: str, data: Dict[str, Dict[str, Any]], fieldnames: List[str]):
        """通用CSV写入方法。"""
        try:
            with open(file_path, mode='w', encoding='utf-8', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data.values())
        except Exception as e:
            logger.error(f"{self.log_prefix} Error writing CSV {file_path}: {e}", exc_info=True)
