# WorldModel 新架构设计

## 职责定位

WorldModel 在 Kernel 驱动的自主 Agent 系统中，负责管理 **Agent 自身的基本身份和全局状态**。

### 核心职责

1. **Agent 身份信息**（静态）
   - 从 identity.md 加载
   - 名称、昵称、身份描述
   - 性格、兴趣、表达风格

2. **全局动态状态**（动态）
   - 当前心情
   - 当前能量值
   - 更新时间戳

3. **任务状态观察**（接口）
   - 获取当前焦点任务
   - 获取活跃任务列表
   - 供 MainPlanner 使用

### 不再负责

- ❌ Cortex 状态管理 → Cortex 各自通过 `get_cortex_summary()` 提供
- ❌ 记忆存储 → UnifiedMemory
- ❌ cortex_data → 已废弃
- ❌ 通知管理 → InterruptHandler
- ❌ 短期记忆 → UnifiedMemory

## 与其他组件的关系

```
┌─────────────────────────────────────────────────────────────┐
│                      Mind (主脑提示词拼接器)                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 1. identity.md → WorldModel                    │  │
│  │ 2. 状态注入 (时间/心情/能量) → WorldModel        │  │
│  │ 3. Cortex 摘要 → CortexManager.collect_summaries() │  │
│  │ 4. UnifiedMemory 检索                      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    WorldModel (Agent 自身状态)                │
│  - 静态：name, personality, identity, interests            │
│  - 动态：mood, energy, last_update                       │
│  - 任务观察：focus_task, active_tasks 接口                 │
└─────────────────────────────────────────────────────────────┘
```

## 接口设计

### 核心接口

```python
class WorldModel:
    # --- 静态身份 ---
    @property
    def name(self) -> str: ...
    @property
    def personality(self) -> str: ...
    @property
    def identity(self) -> str: ...

    # --- 动态状态 ---
    @property
    def mood(self) -> str: ...
    @mood.setter
    def mood(self, value: str): ...

    @property
    def energy(self) -> int: ...
    @energy.setter
    def energy(self, value: int): ...

    # --- 任务观察（便捷接口） ---
    async def get_focus_task() -> Optional[str]: ...
    async def get_active_tasks() -> List[TaskSummary]: ...
    async def get_observation() -> str: ...

    # --- 向前兼容（可选） ---
    async def get_full_system_state() -> Dict[str, Any]: ...
```

## 使用场景

### Mind 系统使用

```python
# Mind.build_full_context()
mood = world_model.mood
energy = world_model.energy
# 注入到主脑提示词的"状态注入"部分
```

### MainPlanner 使用

```python
# MainPlanner.plan()
tasks = await world_model.get_active_tasks()
focus = await world_model.get_focus_task()
# 用于指导任务调度决策
```

### 动态状态更新

```python
# Action 执行后
world_model.energy -= 5
world_model.mood = "疲惫"
```

## 迁移建议

**保持接口兼容**：当前 WorldModel 已有完整实现，建议：

1. **逐步简化**
   - 先保持不变，确保系统稳定运行
   - 逐步移除不必要的功能

2. **向前兼容**：保留 `get_full_system_state()` 等接口

3. **定义边界**：
   - 身份信息 → WorldModel
   - 任务状态 → TaskStore + Scheduler
   - Cortex 状态 → CortexManager
   - 记忆 → UnifiedMemory
