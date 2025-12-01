# src/llm_api/model_client/__init__.py

# 在此导入所有具体的客户端实现，以确保它们的类装饰器能够被执行，
# 从而将它们自己注册到全局的 client_registry 中。

from . import openai_client
# 如果未来有其他客户端，例如 anthorpic_client, google_client 等，
# 也在这里添加它们的导入语句。
# from . import anthorpic_client
# from . import google_client

print("模型客户端已加载并注册：OpenAI")