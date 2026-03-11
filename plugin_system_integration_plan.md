# 插件系统与Cortex集成方案 (v3 - 最终版)

本文档记录了 `Shiraha_bot` 项目中，将外部 `plugin_system` 与核心 `cortex` 体系集成的最终设计方案。该方案基于项目开发者的反馈进行了多次修正，以确保其设计思想的精确性。

## 1. 核心准则 (The Final Rule)

为了实现插件工具在不同场景下的可用性控制，我们确立了以下核心准则：

1.  **`global` 是一个通行身份**：任何在 `scopes` 列表中包含 `"global"` 的工具，对于系统中的 **任何一个规划器（Planner）** 都是可见且可用的。无论这个规划器是在 `MainPlanner` 中，还是在 `QQChatCortex` 里，或是在任何其他 `Cortex` 中。它是一个无差别的、全局的工具集。

2.  **特定 `scope` 是上下文门票**：形如 `"cortex:qq_chat"` 这样的 `scope` 是一个“上下文门票”。只有当规划器**恰好处于**这个上下文环境中时，它才能看到并使用被这个 `scope` 标记的工具。

3.  **最终可用工具集 = 通行工具 + 上下文门票工具**：任何一个规划器，在它运行时，最终可用的工具列表 = **系统中所有被标记为 `global` 的工具** + **所有与它当前上下文匹配的特定 `scope` 工具**。

## 2. 实现方案

### 2.1 `_manifest.json` 结构：`scopes` 列表

为了支持工具的多作用域可见性，我们将插件清单文件 `_manifest.json` 中的 `scope` 字段升级为 `scopes` **字符串列表**。

**示例:**
```jsonc
// in src/plugins/your_plugin/_manifest.json
{
  "name": "your_plugin",
  // ...
  "tools": [
    {
      "name": "get_weather",
      "description": "获取天气。这是一个通用的基础功能。",
      "scopes": ["global"] 
    },
    {
      "name": "analyze_chat_sentiment",
      "description": "分析聊天消息的情感。在通用场景和QQ聊天场景下都很有用。",
      "scopes": ["global", "cortex:qq_chat"]
    },
    {
      "name": "generate_group_summary",
      "description": "为群聊生成每日摘要，只适用于QQ聊天场景。",
      "scopes": ["cortex:qq_chat"]
    }
  ]
}
```
- **默认规则**: 如果 `scopes` 字段在 `tool` 定义中缺失或为空列表，系统应默认其为 `["global"]`。

### 2.2 `PluginManager` 职责：忠实加载

`PluginManager` 的职责保持纯粹，它只需在加载插件时，完整地读取每个工具的 `scopes` 列表并存储在 `ToolInfo` 对象中。它不负责执行任何过滤逻辑。它需要提供一个接口，供外部获取所有工具的完整信息。

**建议方法**:
```python
# In src/plugin_system/core/plugin_manager.py
class PluginManager:
    # ...
    def get_all_tools_with_info(self) -> List[Tuple[Type[BaseTool], BasePlugin, ToolInfo]]:
        """返回所有已注册的工具及其完整的元信息（包含scopes）"""
        return list(self._tools.values())
```

### 2.3 调用方实现过滤与拼接逻辑

工具的过滤和拼接逻辑由**工具的消费者**（即各个规划器）根据“核心准则”来动态执行。

#### 场景一：在 `BaseCortex` 中（上下文规划）

当一个 `Cortex` 需要进行规划时，它能看到的工具集是 **它自己的原生工具** + **全局插件工具** + **为它定制的插件工具**。

**实现**:
```python
# In src/cortices/base_cortex.py
class BaseCortex(ABC):
    # ... (假设已通过依赖注入获得 plugin_manager) ...

    def get_contextual_tools(self) -> List[Dict[str, Any]]:
        """获取当前 Cortex 上下文中所有可用的工具定义。"""
        # 1. 获取 Cortex 自己的原生工具
        native_tool_definitions = [build_tool_definition(t) for t in self.get_tools()]

        # 2. 定义当前上下文的作用域
        current_cortex_scope = f"cortex:{self.cortex_name}" # e.g., "cortex:qq_chat"

        # 3. 从 PluginManager 获取所有插件工具并进行过滤
        all_plugin_tools_info = self.plugin_manager.get_all_tools_with_info()
        
        contextual_plugin_tools = []
        for _tool_cls, _plugin, tool_info in all_plugin_tools_info:
            tool_scopes = tool_info.scopes or ["global"]
            
            # 核心准则应用：如果工具是全局的(有通行身份)，或它是为当前Cortex定制的(有正确的上下文门票)，就加入列表
            if "global" in tool_scopes or current_cortex_scope in tool_scopes:
                contextual_plugin_tools.append(build_tool_definition(tool_info))

        # 4. 返回拼接后的最终列表
        return native_tool_definitions + contextual_plugin_tools
```

#### 场景二：在 `MainPlanner` 中（全局规划）

`MainPlanner` 的特殊性在于它不隶属于任何特定 `Cortex`。因此，它的上下文是“空的”，它只能使用那些拥有“通行身份” (`global`) 的工具。

**原理分析**:
`MainPlanner` 同样遵循“核心准则”。假设它的上下文为 `current_context = "main_planner"`。在过滤时，它的条件是 `if "global" in tool.scopes or "main_planner" in tool.scopes:`。由于几乎不会有插件将自己标记到 `"main_planner"` 这个特定范围，该条件实际上退化为 `if "global" in tool.scopes:`。

因此，`MainPlanner` 最终只能看到全局工具，这**不是因为它有特权，而是因为它没有特定的上下文门票，是“核心准则”应用下的自然结果**。

## 3. 结论

该方案通过 `scopes` 列表和调用方过滤的模式，优雅地实现了插件工具的作用域控制。它具备以下优点：

- **极致的灵活性**：一个工具可以轻松地被部署到一个或多个作用域。
- **清晰的职责划分**：`PluginManager` 负责加载，消费者（Planner）负责根据自身环境进行过滤和使用。
- **准确反映设计思想**：完美实现了“`global`为通行证，`scope`为上下文门票”的叠加逻辑。
