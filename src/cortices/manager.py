import inspect
import json
import os
import importlib
from pathlib import Path
from typing import Callable, Any, Dict, List, Coroutine, Type, Optional

from src.llm_api.dto import ToolCall
from src.cortices.base_cortex import BaseCortex # 导入 BaseCortex
from src.agent.world_model import WorldModel # CortexManager需要解析并传递WorldModel给Cortex
from src.system.di.container import container # 用于解析WorldModel

# 导入Cortex配置加载器
from src.cortices.cortex_config_loader import load_cortex_config
from pydantic import BaseModel, ValidationError # 用于配置验证

# 类型提示
CallableTool = Callable[..., Coroutine[Any, Any, Any]]

# Manifest 文件常量
CORTEX_MANIFEST_FILE = "manifest.json"

class CortexManager:
    """
    Cortex 与工具的管理者。
    它使用装饰器模式来自动发现和注册工具，并负责执行它们。
    这是一个单例。
    """
    _instance = None
    _tool_schemas: Dict[str, Dict[str, Dict]]
    _tool_implementations: Dict[str, CallableTool]
    _cortices: Dict[str, BaseCortex] # 存储已实例化的 Cortex
    _collected_impetus_descriptions: List[str] # 收集到的所有Cortex的内在驱动力描述

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CortexManager, cls).__new__(cls)
            cls._instance._tool_schemas = {}
            cls._instance._tool_implementations = {}
            cls._instance._cortices = {} # 初始化cortices字典
            cls._instance._collected_impetus_descriptions = [] # 初始化内在驱动力描述列表
        return cls._instance

    def tool(self, scope: str = "main"):
        """
        工具注册装饰器。
        """
        def decorator(func: CallableTool) -> CallableTool:
            tool_name = func.__name__
            if tool_name in self._tool_implementations:
                print(f"警告：工具 '{tool_name}' 被重复定义。")
            
            sig = inspect.signature(func)
            docstring = inspect.getdoc(func)
            
            description = docstring.strip().split('\n')[0] if docstring else "无描述"

            parameters = {"type": "object", "properties": {}}
            required_params = []

            type_mapping = { str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object" }

            for name, param in sig.parameters.items():
                if name in ('self', 'cls'): continue
                
                param_type = type_mapping.get(param.annotation, "string")
                parameters["properties"][name] = {"type": param_type}
                
                if param.default is inspect.Parameter.empty:
                    required_params.append(name)

            if required_params:
                parameters["required"] = required_params

            schema = {
                "type": "function",
                "function": { "name": tool_name, "description": description, "parameters": parameters }
            }

            if scope not in self._tool_schemas:
                self._tool_schemas[scope] = {}
            self._tool_schemas[scope][tool_name] = schema
            self._tool_implementations[tool_name] = func
            
            print(f"工具 '{tool_name}' 已成功注册到作用域 '{scope}'。")
            return func
        return decorator

    def get_tool_schemas(self, scope: str) -> List[Dict]:
        """获取指定作用域下的所有工具 Schema 列表，用于构建 Prompt。"""
        return list(self._tool_schemas.get(scope, {}).values())
    
    def get_main_scope_tool_schemas(self) -> List[Dict]:
        """
        获取 'main' 作用域下的所有工具 Schema 列表。
        这些工具代表了 Agent 的高阶能力，可用于MotiveEngine生成意图的上下文。
        """
        return self.get_tool_schemas(scope="main")
    
    def get_collected_impetus_descriptions(self) -> List[str]:
        """
        获取所有已加载 Cortex 的内在驱动力描述列表。
        """
        return self._collected_impetus_descriptions

    async def execute_tool(self, tool_call: ToolCall) -> Any:
        """根据 ToolCall 对象执行相应的工具。"""
        tool_name = tool_call.func_name
        if tool_name not in self._tool_implementations:
            return f"错误：未找到名为 '{tool_name}' 的工具实现。"
        
        func = self._tool_implementations[tool_name]
        args = tool_call.args or {}

        try:
            # CortexManager 假定所有工具都是异步函数
            return await func(**args)
        except Exception as e:
            return f"执行工具 '{tool_name}' 时出错: {e}"

    async def load_all_cortices(self):
        """
        扫描 src/cortices 目录，加载所有 Cortex 模块，
        并调用它们的 setup 方法。
        """
        print("[CortexManager] 开始加载所有 Cortex...")
        cortices_base_path = Path(os.path.dirname(os.path.abspath(__file__))) # src/cortices 的路径

        # 解析WorldModel实例，供所有Cortex使用
        world_model: WorldModel = container.resolve(WorldModel)

        for cortex_dir in cortices_base_path.iterdir():
            if cortex_dir.is_dir() and not cortex_dir.name.startswith('__'):
                cortex_name = cortex_dir.name
                manifest_path = cortex_dir / CORTEX_MANIFEST_FILE
                
                # 1. 加载 manifest.json
                if not manifest_path.exists():
                    print(f"[CortexManager] 警告: Cortex '{cortex_name}' 缺少 {CORTEX_MANIFEST_FILE}，跳过加载。")
                    continue
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    main_class_path = manifest.get("main_class_path") # e.g., "src.cortices.qq_chat.cortex.QQChatCortex"
                    if not main_class_path:
                        print(f"[CortexManager] 警告: Cortex '{cortex_name}' 的 {CORTEX_MANIFEST_FILE} 缺少 'main_class_path'，跳过加载。")
                        continue
                    
                    # 提取模块路径和类名
                    module_path, class_name = main_class_path.rsplit('.', 1)

                    # 收集内在驱动力描述
                    impetus_data = manifest.get("impetus")
                    if impetus_data and isinstance(impetus_data, dict):
                        impetus_descriptions = impetus_data.get("impetus_description")
                        if impetus_descriptions and isinstance(impetus_descriptions, list):
                            self._collected_impetus_descriptions.extend(impetus_descriptions)


                except Exception as e:
                    print(f"[CortexManager] 错误: 读取或解析 Cortex '{cortex_name}' 的 {CORTEX_MANIFEST_FILE} 失败: {e}，跳过加载。")
                    continue
                
                # 2. 加载并验证 Cortex 的配置
                validated_cortex_config: Optional[BaseModel] = None
                try:
                    # 动态导入 Cortex 特定的配置 Schema
                    # 遵循统一命名约定：CortexName/config/config_schema.py:CortexConfigSchema
                    config_module_path = f"src.cortices.{cortex_name}.config.config_schema"
                    config_schema_module = importlib.import_module(config_module_path)
                    cortex_config_schema: Type[BaseModel] = getattr(config_schema_module, "CortexConfigSchema")
                    
                    validated_cortex_config = load_cortex_config(cortex_dir, cortex_config_schema)
                    
                    # 检查 Cortex 是否启用 (由配置Schema负责定义enabled字段)
                    if hasattr(validated_cortex_config, 'enabled') and not validated_cortex_config.enabled:
                        print(f"[CortexManager] 信息: Cortex '{cortex_name}' 在配置中被禁用，跳过加载。")
                        continue

                except FileNotFoundError:
                    # 如果没有config.toml，Pydantic模型会使用其字段的默认值
                    print(f"[CortexManager] 信息: Cortex '{cortex_name}' 没有找到 config/config.toml 文件，尝试使用默认配置。")
                    try:
                        config_module_path = f"src.cortices.{cortex_name}.config.config_schema"
                        config_schema_module = importlib.import_module(config_module_path)
                        cortex_config_schema: Type[BaseModel] = getattr(config_schema_module, "CortexConfigSchema")
                        validated_cortex_config = cortex_config_schema() # 实例化一个默认配置
                        if hasattr(validated_cortex_config, 'enabled') and not validated_cortex_config.enabled:
                            print(f"[CortexManager] 信息: Cortex '{cortex_name}' 默认配置中被禁用，跳过加载。")
                            continue
                    except (ModuleNotFoundError, AttributeError) as e:
                        print(f"[CortexManager] 警告: Cortex '{cortex_name}' 未找到 config/schema.py 或其中缺少 CortexConfigSchema: {e}，跳过配置验证。")
                        continue
                except (ModuleNotFoundError, AttributeError) as e:
                    print(f"[CortexManager] 警告: Cortex '{cortex_name}' 未找到 config/schema.py 或其中缺少 CortexConfigSchema: {e}，跳过配置验证。")
                    continue
                except ValidationError as e:
                    print(f"[CortexManager] 错误: Cortex '{cortex_name}' 配置验证失败: {e}，跳过加载。")
                    continue
                except Exception as e:
                    print(f"[CortexManager] 错误: 加载或验证 Cortex '{cortex_name}' 配置时发生未知错误: {e}，跳过加载。")
                    continue

                try:
                    # 动态导入 Cortex 类
                    cortex_module = importlib.import_module(module_path)
                    cortex_class: Type[BaseCortex] = getattr(cortex_module, class_name)

                    if inspect.isclass(cortex_class) and issubclass(cortex_class, BaseCortex) and cortex_class is not BaseCortex:
                        cortex_instance = cortex_class()
                        self._cortices[cortex_name] = cortex_instance
                        
                        print(f"[CortexManager] 发现并实例化 Cortex: '{cortex_name}'")
                        # 将WorldModel和验证后的配置对象传递给Cortex的setup方法
                        await cortex_instance.setup(world_model, validated_cortex_config) 
                        print(f"[CortexManager] Cortex '{cortex_name}' 启动成功。")
                    else:
                        print(f"[CortexManager] 警告: '{main_class_path}' 不是一个有效的 Cortex 子类，跳过加载。")

                except Exception as e:
                    print(f"[CortexManager] 错误: 加载 Cortex '{cortex_name}' 失败: {e}。")
                    import traceback
                    traceback.print_exc() # 打印完整的堆栈跟踪以进行调试
        print("[CortexManager] 所有 Cortex 加载完成。")

    async def shutdown_all_cortices(self):
        """
        调用所有已加载 Cortex 的 teardown 方法。
        """
        print("[CortexManager] 开始关闭所有 Cortex...")
        # 按照反向加载顺序关闭，或者只是遍历
        for cortex_name, cortex_instance in list(self._cortices.items()): # 遍历拷贝以避免在迭代时修改字典
            try:
                await cortex_instance.teardown()
                print(f"[CortexManager] Cortex '{cortex_name}' 关闭成功。")
                del self._cortices[cortex_name]
            except Exception as e:
                print(f"[CortexManager] 错误: 关闭 Cortex '{cortex_name}' 失败: {e}")
        print("[CortexManager] 所有 Cortex 关闭完成。")

    def register_subplanner(self, cortex_id: str, subplanner_class: Type[Any]):
        """
        由 Cortex 调用，注册其内部的 SubPlanner。
        """
        print(f"[CortexManager] Cortex '{cortex_id}' 注册 SubPlanner: {subplanner_class.__name__}")
        # TODO: 实际的 SubPlanner 存储和检索逻辑
        pass