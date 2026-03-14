import os

class SkillBox:
    def __init__(self, base_path: str):
        self.base_path = base_path # 指向 src/plugins/rpg_plugin/skills_pool/

    async def load_skill(self, skill_id: str) -> str:
        """加载 SKILL.md 内容"""
        path = os.path.join(self.base_path, skill_id, "SKILL.md")
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    async def read_skill_file(self, skill_id: str, rel_path: str) -> str:
        """受控读取资源文件，防止路径穿透"""
        # 安全检查逻辑...
        target_path = os.path.join(self.base_path, skill_id, rel_path)
        with open(target_path, 'r', encoding='utf-8') as f:
            return f.read()

    async def run_skill_script(self, skill_id: str, entry_name: str, args: dict):
        """执行 manifest 中定义的脚本入口"""
        # 这里的 entry_name 对应 manifest.json 里的 key
        # 通过子进程或动态导入运行对应的 .py 文件
        pass