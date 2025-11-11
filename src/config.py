import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
# 这行代码会自动寻找项目根目录下的 .env 文件并加载
load_dotenv()

# 从环境变量中读取配置并构建模型列表
LLM_MODELS = []

# --- 配置 DeepSeek 模型 ---
if os.getenv("DEEPSEEK_API_KEY"):
    LLM_MODELS.append({
        "model_name": "deepseek",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL"),
        "llm_model_name": os.getenv("DEEPSEEK_MODEL_NAME")
    })

# --- 配置 OpenAI 模型 (示例) ---
if os.getenv("OPENAI_API_KEY"):
    LLM_MODELS.append({
        "model_name": "openai",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "llm_model_name": os.getenv("OPENAI_MODEL_NAME")
    })

# 如果没有配置任何模型，可以抛出错误或设置一个默认提示
if not LLM_MODELS:
    print("警告：未在 .env 文件中配置任何 LLM 模型。")
    # raise ValueError("请在 .env 文件中配置至少一个 LLM 模型。")