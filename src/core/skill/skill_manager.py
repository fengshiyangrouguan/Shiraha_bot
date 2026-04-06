"""
Skill Manager - Skill 管理器

管理 AI 的技能文档（Skill.md），支持加载、更新、自我修改等功能。
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import re

from src.common.logger import get_logger

logger = get_logger("skill_manager")


@dataclass
class SkillMetadata:
    """Skill 元数据"""
    cortex: str                    # 所属 cortex
    name: str                      # Skill 名称
    version: str = "1.0"           # 版本号
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    usage_count: int = 0           # 使用次数
    success_count: int = 0         # 成功次数
    patterns: List[str] = field(default_factory=list)  # 行为模式


@dataclass
class SkillPattern:
    """行为模式"""
    pattern_id: str
    trigger_condition: str         # 触发条件描述
    action_sequence: str           # 行为序列
    expected_outcome: str          # 期望结果
    confidence: float = 0.5        # 置信度（0-1）
    success_rate: float = 0.0      # 成功率（0-1）


class SkillManager:
    """
    Skill 管理器

    管理 AI 的技能文档，支持动态加载和自我修改
    """

    def __init__(self, skills_dir: str = "src/core/skill/skills"):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        # 存储 skill 内容和元数据
        self._skills: Dict[str, str] = {}  # name -> content
        self._metadata: Dict[str, SkillMetadata] = {}  # name -> metadata

        # 存储 cortex 到 skill 的映射
        self._cortex_skills: Dict[str, List[str]] = {}

        # 行为模式记录
        self._patterns: List[SkillPattern] = []

        # 初始化时从文件加载
        self._load_all_skills()

    def _load_all_skills(self):
        """从文件加载所有 skill"""
        # 按 cortex 组织
        for cortex_dir in self.skills_dir.iterdir():
            if cortex_dir.is_dir() and not cortex_dir.name.startswith("__"):
                cortex_name = cortex_dir.name
                self._cortex_skills[cortex_name] = []

                for skill_file in cortex_dir.glob("*.md"):
                    skill_name = skill_file.stem
                    with open(skill_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        self._skills[f"{cortex_name}.{skill_name}"] = content

                        # 提取元数据
                        metadata = self._extract_metadata(content, cortex_name, skill_name)
                        self._metadata[f"{cortex_name}.{skill_name}"] = metadata
                        self._cortex_skills[cortex_name].append(skill_name)

                        logger.debug(f"加载 Skill: {cortex_name}.{skill_name}")

        logger.info(f"已加载 {len(self._skills)} 个 Skill")

    def _extract_metadata(self, content: str, cortex: str, name: str) -> SkillMetadata:
        """从 skill 内容提取元数据"""
        metadata = SkillMetadata(cortex=cortex, name=name)

        # 提取版本号
        version_match = re.search(r'version:\s*["\']?([0-9.]+)["\']?', content, re.IGNORECASE)
        if version_match:
            metadata.version = version_match.group(1)

        # 提取行为模式
        # 查找类似 "pattern: ..." 或 "When ... then ..." 的模式描述
        pattern_matches = re.findall(
            r'(?:pattern|when|trigger)\s*[:：]\s*(.+?)(?:\n|$)',
            content,
            re.IGNORECASE
        )
        metadata.patterns = [m.strip() for m in pattern_matches if m.strip()]

        return metadata

    def get_skill(self, cortex: str, skill_name: str) -> Optional[str]:
        """
        获取 Skill 内容

        Args:
            cortex: cortex 名称
            skill_name: skill 名称

        Returns:
            Skill 内容，如果不存在则返回 None
        """
        key = f"{cortex}.{skill_name}"
        return self._skills.get(key)

    def get_all_skills(self, cortex: str = "") -> Dict[str, str]:
        """
        获取所有 Skill

        Args:
            cortex: 可选的 cortex 过滤器

        Returns:
            {skill_key: content} 字典
        """
        if cortex:
            return {
                f"{cortex}.{name}": content
                for name in self._cortex_skills.get(cortex, [])
                if f"{cortex}.{name}" in self._skills
            }
        return self._skills.copy()

    def get_metadata(self, cortex: str, skill_name: str) -> Optional[SkillMetadata]:
        """获取 Skill 元数据"""
        key = f"{cortex}.{skill_name}"
        return self._metadata.get(key)

    def update_skill(
        self,
        cortex: str,
        skill_name: str,
        content: str,
        update_type: str = "append"
    ) -> bool:
        """
        更新 Skill

        Args:
            cortex: cortex 名称
            skill_name: skill 名称
            content: 新内容
            update_type: 更新类型 ("replace", "append", "prepend")

        Returns:
            是否更新成功
        """
        key = f"{cortex}.{skill_name}"
        old_content = self._skills.get(key, "")

        if update_type == "replace":
            new_content = content
        elif update_type == "append":
            new_content = old_content + "\n\n" + content if old_content else content
        elif update_type == "prepend":
            new_content = content + "\n\n" + old_content if old_content else content
        else:
            logger.error(f"未知的更新类型: {update_type}")
            return False

        # 更新内存
        self._skills[key] = new_content

        # 更新元数据
        if key in self._metadata:
            self._metadata[key].updated_at = time.time()

        # 保存到文件
        skill_file = self.skills_dir / cortex / f"{skill_name}.md"
        skill_file.parent.mkdir(parents=True, exist_ok=True)
        with open(skill_file, "w", encoding="utf-8") as f:
            f.write(new_content)

        logger.info(f"Skill 已更新: {key} ({update_type})")
        return True

    def record_usage(self, cortex: str, skill_name: str, success: bool = True):
        """记录 Skill 使用情况"""
        key = f"{cortex}.{skill_name}"

        if key in self._metadata:
            self._metadata[key].usage_count += 1
            if success:
                self._metadata[key].success_count += 1

        logger.debug(f"记录 Skill 使用: {key} (success={success})")

    def add_pattern(
        self,
        cortex: str,
        skill_name: str,
        trigger: str,
        action: str,
        outcome: str,
        confidence: float = 0.5
    ):
        """添加行为模式"""
        pattern_id = f"pattern_{len(self._patterns)}_{int(time.time())}"
        pattern = SkillPattern(
            pattern_id=pattern_id,
            trigger_condition=trigger,
            action_sequence=action,
            expected_outcome=outcome,
            confidence=confidence
        )
        self._patterns.append(pattern)

        # 同时更新 skill 文档
        pattern_text = f"""
## Pattern: {pattern_id}

**触发条件**: {trigger}
**行为序列**: {action}
**期望结果**: {outcome}
**置信度**: {confidence}

"""
        self.update_skill(cortex, skill_name, pattern_text, update_type="append")

        logger.info(f"添加行为模式: {pattern_id} -> {cortex}.{skill_name}")

    def evolve_skill(
        self,
        cortex: str,
        skill_name: str,
        timeframe_hours: float = 168
    ) -> List[Dict[str, Any]]:
        """
        基于 useage record 演化 Skill

        Args:
            cortex: cortex 名称
            skill_name: skill 名称
            timeframe_hours: 时间范围（小时）

        Returns:
            演化建议列表
        """
        suggestions = []

        metadata = self.get_metadata(cortex, skill_name)
        if not metadata:
            suggestions.append({
                "type": "error",
                "message": f"Skill {cortex}.{skill_name} 不存在"
            })
            return suggestions

        # 计算成功率
        if metadata.usage_count > 0:
            success_rate = metadata.success_count / metadata.usage_count
        else:
            success_rate = 0.0

        # 基于成功率给出建议
        if success_rate < 0.5:
            suggestions.append({
                "type": "improve",
                "message": f"成功率较低 ({success_rate:.2%})，建议优化行为模式",
                "priority": "high"
            })
        elif success_rate < 0.8:
            suggestions.append({
                "type": "review",
                "message": f"成功率中等 ({success_rate:.2%})，可考虑微调",
                "priority": "medium"
            })
        else:
            suggestions.append({
                "type": "maintain",
                "message": f"成功率很高 ({success_rate:.2%})，保持现状",
                "priority": "low"
            })

        # 检查模式数量
        if len(metadata.patterns) == 0:
            suggestions.append({
                "type": "enhance",
                "message": "没有定义行为模式，建议添加",
                "priority": "medium"
            })

        return suggestions

    def analyze_action(
        self,
        cortex: str,
        skill_name: str,
        action_name: str,
        timeframe_hours: float = 168
    ) -> Dict[str, Any]:
        """
        分析特定行为的使用情况

        Args:
            cortex: cortex 名称
            skill_name: skill 名称
            action_name: 行为名称
            timeframe_hours: 时间范围

        Returns:
            分析结果
        """
        # TODO: 实现基于记录的详细分析
        return {
            "action_name": action_name,
            "cortex": cortex,
            "skill": skill_name,
            "usage_count": 0,
            "avg_confidence": 0.0,
            "message": "分析功能待实现"
        }

    def merge_patterns(
        self,
        from_cortex: str,
        to_cortex: str,
        pattern_filter: str = ""
    ) -> int:
        """
        合并行为模式

        Args:
            from_cortex: 源 cortex
            to_cortex: 目标 cortex
            pattern_filter: 模式过滤器

        Returns:
            合并的模式数量
        """
        merged_count = 0

        # 获取源 cortex 的所有 patterns
        from_patterns = [
            p for p in self._patterns
            if p.pattern_id.startswith(from_cortex)
        ]

        # 过滤
        if pattern_filter:
            pattern_filter_lower = pattern_filter.lower()
            from_patterns = [
                p for p in from_patterns
                if pattern_filter_lower in str(p.__dict__).lower()
            ]

        # 添加到目标 cortex
        for pattern in from_patterns:
            # 生成新的 pattern 内容
            pattern_text = f"""
## Merged Pattern from {from_cortex}

**触发条件**: {pattern.trigger_condition}
**行为序列**: {pattern.action_sequence}
**期望结果**: {pattern.expected_outcome}
**原始置信度**: {pattern.confidence}
**合并时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}

"""
            # 添加到目标的第一个 skill（或创建新的）
            target_skills = self._cortex_skills.get(to_cortex, [])
            if target_skills:
                self.update_skill(to_cortex, target_skills[0], pattern_text, "append")
                merged_count += 1

        logger.info(f"合并模式: {from_cortex} -> {to_cortex} (共 {merged_count} 个)")
        return merged_count

    def get_capability_summary(self, cortex: str = "") -> str:
        """
        获取能力摘要（用于系统概述）

        Args:
            cortex: 可选的 cortex 过滤器

        Returns:
            摘要文本
        """
        if cortex:
            skills = self._cortex_skills.get(cortex, [])
            summary_lines = [f"### {cortex.title()} Cortex 能力"]
            for skill_name in skills:
                metadata = self.get_metadata(cortex, skill_name)
                if metadata:
                    success_rate = ""
                    if metadata.usage_count > 0:
                        success_rate = f" (成功率: {metadata.success_count/metadata.usage_count:.1%})"
                    summary_lines.append(f"- {skill_name}: 使用 {metadata.usage_count} 次{success_rate}")
            return "\n".join(summary_lines)
        else:
            lines = ["## 系统能力总览"]
            for ctx, skill_list in self._cortex_skills.items():
                if skill_list:
                    lines.append(f"\n**{ctx.title()}**: {', '.join(skill_list)}")
            return "\n".join(lines)

    def create_default_skill(
        self,
        cortex: str,
        skill_name: str,
        description: str = ""
    ):
        """
        创建默认 Skill 模板

        新 cortex 加载时调用，引导 Agent 自我探索和测试
        """
        content = f"""# {skill_name.title()}

## 能力介绍

你获得了新的能力：{description or '这个 cortex 域的功能'}

## 系统初始化

这是一个新的能力领域。请通过以下方式探索：
1. 尝试使用这个 cortex 提供的基本工具
2. 观察工具的输入输出
3. 记录工具的使用场景和效果
4. 分享你发现的有趣用法

## 探索任务

作为系统初始化的一部分，建议你：

**任务 1**：了解可用的工具
- 尝试列出所有工具
- 理解每个工具的用途

**任务 2**：测试基础功能
- 使用工具进行简单操作
- 记录执行结果

**任务 3**：记录观察
- 将你的发现记录到这里
- 更新这个 Skill 文档

## 待记录事项

- [ ] 工具列表和用途
- [ ] 常用操作流程
- [ ] 特殊情况处理
- [ ] 有趣的发现
- [ ] 改进建议

---

*记录于: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        self.update_skill(cortex, skill_name, content, "replace")
        logger.info(f"创建默认 Skill: {cortex}.{skill_name}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_skills": len(self._skills),
            "cortices": len(self._cortex_skills),
            "patterns": len(self._patterns),
            "total_usage": sum(m.usage_count for m in self._metadata.values())
        }
