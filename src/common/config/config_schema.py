# schema.py
from pydantic import BaseModel, Field
from typing import Literal, Annotated

# 定义允许的日志级别
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class SystemConfig(BaseModel):
    """系统配置"""
    version: str = Field(
        default="v_0.1.1",
        description="应用程序版本号"
    )


class DeviceConfig(BaseModel):
    """设备配置"""
    cuda_device: Annotated[int, Field(
        ge=0,
        description="要使用的CUDA设备ID"
    )] = 0

    use_cpu: bool = Field(
        default=False,
        description="是否回退到使用CPU"
    )


class FeatureExtractConfig(BaseModel):
    """特征提取配置"""
    cleanup_openface_output: bool = Field(
        default=True,
        description="特征提取完成后是否清理OpenFace输出文件"
    )


class DiagnosisConfig(BaseModel):
    """诊断/分析配置"""
    confidence_threshold: float = Field(
        ge=0.0, le=1.0,
        default=0.5,
        description="诊断置信度阈值"
    )

    min_frames_required: Annotated[int, Field(
        ge=1,
        description="进行诊断所需的最小帧数"
    )] = 10


class ModelConfig(BaseModel):
    """模型配置"""
    model_name: str = Field(
        default="frame_128_20250102_161312.pth",
        description="模型文件的名称（例如, .pth 文件名）"
    )


class LoggingConfig(BaseModel):
    """日志配置"""
    log_level: LogLevel = Field(
        default="INFO",
        description="日志级别 (e.g., INFO, DEBUG, WARNING)"
    )


class AppConfig(BaseModel):
    """
    应用程序的完整配置模型。
    所有字段都有默认值，确保即使传入 {} 也能创建实例（但建议仍提供完整配置）。
    """
    system: SystemConfig = Field(default_factory=SystemConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    feature_extract: FeatureExtractConfig = Field(default_factory=FeatureExtractConfig)
    diagnosis: DiagnosisConfig = Field(default_factory=DiagnosisConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)