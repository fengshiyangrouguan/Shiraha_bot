# src/common/config/schemas/llm_api_config.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class APIProviderConfig(BaseModel):
    """API服务提供商的配置 Schema"""
    name: str
    base_url: str
    api_key: str
    client_type: str = "openai"
    max_retry: int = 2
    timeout: int = 120
    retry_interval: int = 10

class ModelConfig(BaseModel):
    """单个模型的配置 Schema"""
    model_identifier: str
    name: str
    api_provider: str
    extra_params: Dict[str, Any] = Field(default_factory=dict)

class TaskConfig(BaseModel):
    """单个任务的模型及参数配置 Schema"""
    model_list: List[str]
    temperature: float = 0.7
    max_tokens: int = 2048

class LLMApiConfig(BaseModel):
    """llm_api_config.toml 的主 Schema"""
    api_providers: List[APIProviderConfig] = Field(default_factory=list)
    models: List[ModelConfig] = Field(default_factory=list)
    model_task_config: Dict[str, TaskConfig] = Field(default_factory=dict)
