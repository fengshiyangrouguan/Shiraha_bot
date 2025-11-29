# src/common/database/database_manager.py

class DatabaseManager:
    """
    数据库管理器（占位符）。
    未来将封装所有与数据库（如 SQLite, PostgreSQL）的交互。
    """
    def __init__(self, db_path: str = "data/agent.db"):
        self.db_path = db_path
        self._connection = None
        print(f"DatabaseManager: 初始化，数据库路径: '{self.db_path}'")

    def connect(self):
        """建立数据库连接。"""
        print("DatabaseManager: 正在连接到数据库...")
        # self._connection = sqlite3.connect(self.db_path)
        print("DatabaseManager: 数据库连接成功 (模拟)。")

    def get_connection(self):
        """获取数据库连接实例。"""
        if not self._connection:
            self.connect()
        return self._connection
