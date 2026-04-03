"""
Model Selector - 模型选择器

根据任务类型和复杂度智能选择合适的 LLM 模型。
"""
from enum import Enum
from typing import Optional, Dict, Any, List
from src.common.logger import get_logger

logger = get_logger("model_selector")


class TaskType(Enum):
    """任务类型"""
    # 大模型任务（深度思考）
    MOTIVE = "motive"                 # 动机生成
    PLANNER = "planner"               # 主规划
    REPLY = "reply"                   # 聊天回复
    MOOD_UPDATE = "mood_update"       # 心情更新
    SUMMARY = "summary"               # 行为总结

    # 中模型任务（中等复杂度）
    SUB_PLANNER = "sub_planner"       # 子任务规划

    # 小模型任务（快速判断）
    STATE_EVALUATION = "state_evaluation"  # 状态评估
    INTEREST_JUDGMENT = "interest_judgment"  # 兴趣判断
    CLASSIFICATION = "classification"      # 分类
    PRIORITY_FILTER = "priority_filter"    # 优先级初步筛选

    # 专用模型任务
    EMBEDDING = "embedding"           # 向量生成
    VISION = "vision"                 # 图像识别


class ModelSelector:
    """
    模型选择器

    根据任务类型和复杂度自动选择最合适的模型
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._load_model_mapping()

    def _load_model_mapping(self):
        """
        加载模型映射配置

        基于你提供的 llm_api_config.toml 中的配置
        """
        # 大模型（深度思考）
        self.large_models = [
            "siliconflow-deepseek-v3.2",
        ]

        # 中等模型（中等复杂度）
        self.medium_models = [
            "siliconflow-glm-4.6",
        ]

        # 小模型（快速判断）
        self.small_models = [
            "qwen3-30b",
        ]

        # 专用模型
        self.embedding_model = "bge-m3"
        self.vision_model = "qwen3-vl-30"

        # 任务类型到模型组的映射
        self.task_mapping = {
            # 大模型任务
            TaskType.MOTIVE: self.large_models,
            TaskType.PLANNER: self.large_models,
            TaskType.REPLY: self.large_models,
            TaskType.MOOD_UPDATE: self.large_models,
            TaskType.SUMMARY: self.large_models,

            # 中等模型任务
            TaskType.SUB_PLANNER: self.medium_models + self.large_models,

            # 小模型任务
            TaskType.STATE_EVALUATION: self.small_models,
            TaskType.INTEREST_JUDGMENT: self.small_models,
            TaskType.CLASSIFICATION: self.small_models,
            TaskType.PRIORITY_FILTER: self.small_models,

            # 专用模型
            TaskType.EMBEDDING: [self.embedding_model],
            TaskType.VISION: [self.vision_model],
        }

        # 模型的温度配置
        self.temperature_config = {
            TaskType.MOTIVE: 0.7,
            TaskType.PLANNER: 0.3,
            TaskType.REPLY: 0.3,
            TaskType.MOOD_UPDATE: 0.5,
            TaskType.SUMMARY: 0.7,
            TaskType.SUB_PLANNER: 0.3,
            TaskType.STATE_EVALUATION: 0.3,
            TaskType.INTEREST_JUDGMENT: 0.5,
            TaskType.CLASSIFICATION: 0.7,
            TaskType.PRIORITY_FILTER: 0.3,
        }

        # 模型的最大 token 配置
        self.max_tokens_config = {
            TaskType.MOTIVE: 2048,
            TaskType.PLANNER: 2048,
            TaskType.REPLY: 2048,
            TaskType.MOOD_UPDATE: 1024,
            TaskType.SUMMARY: 2048,
            TaskType.SUB_PLANNER: 2048,
            TaskType.STATE_EVALUATION: 512,
            TaskType.INTEREST_JUDGMENT: 512,
            TaskType.CLASSIFICATION: 512,
            TaskType.PRIORITY_FILTER: 256,
        }

        logger.info("ModelSelector 初始化完成")

    def select_model(
        self,
        task_type: TaskType,
        complexity: float = 0.5,
        preferred_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        选择合适的模型

        Args:
            task_type: 任务类型
            complexity: 复杂度（0-1），用于在中大模型间选择
            preferred_model: 首选模型（如果有）

        Returns:
            {"model": str, "temperature": float, "max_tokens": int}
        """
        # 如果指定了首选模型，直接使用
        if preferred_model:
            return {
                "model": preferred_model,
                "temperature": self.temperature_config.get(task_type, 0.3),
                "max_tokens": self.max_tokens_config.get(task_type, 2048)
            }

        # 获取任务类型对应的候选模型列表
        candidates = self.task_mapping.get(task_type, self.large_models)

        # 根据复杂度选择
        selected = self._select_by_complexity(candidates, complexity, task_type)

        result = {
            "model": selected,
            "temperature": self.temperature_config.get(task_type, 0.3),
            "max_tokens": self.max_tokens_config.get(task_type, 2048),
            "task_type": task_type.value,
            "complexity": complexity
        }

        logger.debug(f"选择模型: {selected} for {task_type.value} (complexity={complexity})")
        return result

    def _select_by_complexity(
        self,
        candidates: List[str],
        complexity: float,
        task_type: TaskType
    ) -> str:
        """
        根据复杂度从候选模型中选择

        Args:
            candidates: 候选模型列表
            complexity: 复杂度（0-1）
            task_type: 任务类型

        Returns:
            选择的模型名称
        """
        if not candidates:
            # 默认返回第一个大模型
            return self.large_models[0] if self.large_models else "siliconflow-deepseek-v3.2"

        # 小模型任务直接返回第一个候选
        if task_type in [
            TaskType.STATE_EVALUATION,
            TaskType.INTEREST_JUDGMENT,
            TaskType.CLASSIFICATION,
            TaskType.PRIORITY_FILTER
        ]:
            return candidates[0]

        # 大模型任务
        if task_type in [
            TaskType.MOTIVE,
            TaskType.PLANNER,
            TaskType.REPLY,
            TaskType.MOOD_UPDATE,
            TaskType.SUMMARY
        ]:
            # 直接用大模型，不受复杂度影响
            return self.large_models[0]

        # 子规划任务，根据复杂度选择
        if task_type == TaskType.SUB_PLANNER:
            if complexity > 0.7:
                # 高复杂度，用大模型
                return self.large_models[0] if self.large_models else candidates[0]
            else:
                # 中低复杂度，用中等模型
                return candidates[0] if candidates else self.large_models[0]

        # 默认返回第一个候选
        return candidates[0]

    def get_model_config(self, task_name: str) -> Dict[str, str]:
        """
        获取常见任务的模型配置

        Args:
            task_name: 任务名称（如 "planner", "motive" 等）

        Returns:
            模型配置
        """
        # 将 task_name 映射到 TaskType
        task_mapping = {
            "planner": TaskType.PLANNER,
            "motive": TaskType.MOTIVE,
            "replyer": TaskType.REPLY,
            "utils_small": TaskType.CLASSIFICATION,
            "embedding": TaskType.EMBEDDING,
            "vlm": TaskType.VISION,
        }

        task_type = task_mapping.get(task_name.lower())
        if task_type:
            return self.select_model(task_type)

        # 默认：planner
        return self.select_model(TaskType.PLANNER)

    def get_task_type_from_config(self, config_name: str) -> TaskType:
        """
        从配置名称获取任务类型

        用于 LLMRequestFactory 的兼容
        """
        config_mapping = {
            "motive": TaskType.MOTIVE,
            "main_planner": TaskType.PLANNER,
            "planner": TaskType.SUB_PLANNER,
            "replyer": TaskType.REPLY,
            "utils_small": TaskType.CLASSIFICATION,
            "embedding": TaskType.EMBEDDING,
            "vlm": TaskType.VISION,
            "default": TaskType.PLANNER,
        }

        return config_mapping.get(config_name, TaskType.PLANNER)

    def get_available_models(self) -> Dict[str, List[str]]:
        """获取可用的模型列表，按大小分类"""
        return {
            "large": self.large_models,
            "medium": self.medium_models,
            "small": self.small_models,
            "specialized": [self.embedding_model, self.vision_model]
        }

    def evaluate_complexity(
        self,
        content: str,
        context_length: int = 0
    ) -> float:
        """
        评估任务的复杂度

        基于内容长度、关键词等进行评估

        Args:
            content: 任务内容
            context_length: 上下文长度

        Returns:
            复杂度（0-1）
        """
        complexity = 0.0

        # 基于长度（越长越复杂）
        length_score = min(len(content) / 1000, 0.3)
        complexity += length_score

        # 基于上下文长度
        context_score = min(context_length / 5000, 0.3)
        complexity += context_score

        # 基于关键词
        complex_keywords = [
            "分析", "判断", "决策", "策略", "推理", "推导",
            "compare", "analyze", "decide", "strategy", "reasoning"
        ]
        for keyword in complex_keywords:
            if keyword in content.lower():
                complexity += 0.05

        # 限制在 0-1 范围内
        return min(complexity, 1.0)

    def suggest_model_for_text(self, text: str) -> Dict[str, Any]:
        """
        根据文本内容建议模型

        Args:
            text: 文本内容

        Returns:
            模型建议
        """
        # 估算复杂度
        complexity = self.evaluate_complexity(text)

        # 基于复杂度和长度决定
        if complexity > 0.7 or len(text) > 500:
            # 高复杂度或长文本，用大模型
            return {
                "model": self.large_models[0],
                "reason": f"高复杂度 ({complexity:.2f}) 或长文本 ({len(text)} 字符)"
            }
        elif complexity > 0.4:
            # 中等复杂度
            return {
                "model": self.medium_models[0] if self.medium_models else self.large_models[0],
                "reason": f"中等复杂度 ({complexity:.2f})"
            }
        else:
            # 低复杂度，用小模型
            return {
                "model": self.small_models[0],
                "reason": f"低复杂度 ({complexity:.2f})"
            }
