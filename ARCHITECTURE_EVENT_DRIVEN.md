# Shiraha Bot 事件驱动架构

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interaction Layer                    │
│                   (QQ/微信/终端 等所有感知接口)                    │
└────────────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Cortex Layer (感知/执行层)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ QQ Chat  │  │ Reading  │  │ Calendar │  │ Browser  │  ...   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│         仅提供基础工具，无复杂逻辑链，纯粹信号上报                         │
│   emit_signal() → signal_callback → InterruptHandler              │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Interrupt Handler (信号入口)                  │
│           接收所有 Cortex 信号，进行优先级筛选                                │
│   handle_cortex_signal() → 提取元数据 → 优先级过滤                          │
|   ↓ 提交到 EventLoop队列                                                       │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EventLoop (事件驱动核心循环)                  │
│  ┌──────────────────────┐  ┌──────────────────────┐           │
│  │ 中断信号队列          │  │ 空闲调度器            │           │
│  │ (Interrupt Queue)    │  │ (Idle Scheduler)     │           │
│  │ 处理外部信号          │  │ 只有无任务时调用Motive│          │
│  └──────────────────────┘  └──────────────────────┘           │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Kernel Layer                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ Scheduler        │  │ Interpreter      │  │ TaskManager  │  │
│  │ (注意力调度)      │  │ (指令解释器)      │  │ (任务管理器)  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│          指令: create/exec/suspend/block/kill/adjust_prio             │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Main Planner (主脑规划器)                      │
│            产出 Shell 指令，控制整个系统的行为                            │
│   Mind.build_full_context():                                         │
│   1. identity.md (人设)                                                 │
│   2. 状态注入 (时间/心情/能量)                                            │
│   3. Shell 指令说明
│   4. SKILL 文件                                                      │
│   5. UnifiedMemory 历史记忆                                           │
│   6. 当前工作上下文                                                          │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Components                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ UnifiedMem   │  │ ActionStack  │  │ Executor     │          │
│  │ (统一记忆)    │  │ (行为栈)       │  │ (执行器)      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## 核心流程

### 1. 事件驱动流程（有外部信号）

```
Cortex 接收事件
    ↓
emit_signal(信号)
    ↓
InterruptHandler.handle_cortex_signal()
    ↓
优先级判断
    ├─ 低于屏蔽等级 → 忽略
    └─ 高于屏蔽等级 → 继续
    ↓
EventLoop.submit_interrupt()
    ├─ 放入中断队列
    ├─ TaskManager.create_task()
    └─ Scheduler.schedule() 选择焦点任务
    ↓
[任务执行...]
```

### 2. 空闲调度流程（无任务时）

```
IdleScheduler 检查
    ↓
focus任务? 否
    ↓
ready/background任务? 否
    ↓
MotiveEngine.generate_motive()
    ↓
WorldModel.refresh_task_snapshots()
    ↓
MainPlanner.plan(Mind.build_full_context())
    └─ 1. identity.md
    └─ 2. 状态注入
    └─ 3. Shell 指令说明
    └─ 4. SKILL 文件
    └─ 5. UnifiedMemory 检索
    └─ 6. 当前工作上下文
    ↓
产出 Shell 指令
    ↓
KernelInterpreter.execute_batch()
    └─ task create
    └─ action push
    └─ memory store
    └─ etc.
    ↓
触发任务执行
```

## 关键改进点

### 1. 从固定循环到事件驱动

**旧代码** (已废弃):
```python
async def _run():
    while True:
        await _run_once()
        await sleep(5)  # 固定5秒循环
```

**新代码**:
```python
async def _run_event_loop():
    while True:
        signal = await interrupt_queue.get()  # 阻塞等待信号
        await process_signal(signal)  # 有信号才处理

async def _idle_scheduler():
    while True:
        if not has_tasks():  # 只有无任务时
            await motive_generate()   # 才调用 Motive
        await sleep(1)
```

### 2. UnifiedContext 标准格式

所有上下文统一使用 `{role, content}` 格式:
- `system`: 系统指令、人设
- `user`: 用户输入
- `assistant`: Agent 输出
- `memory`: 记忆检索结果
- `observation`: 观察和感知

### 3. Mind 主脑提示词拼接器

`src/core/mind/mind_core.py` 负责按固定顺序拼接提示词:

```python
await mind.build_full_context(
    motive="当前动机",
    mood="当前心情",
    energy=100,
    memory_limit=5
)
```

输出完整 messages 列表，供 LLM 使用。

## 文件结构

```
src/
├── core/
│   ├── kernel/
│   │   ├── event_loop.py          # 事件驱动核心循环
│   │   ├── interrupt_handler.py   # 中断处理器
│   │   ├── scheduler.py           # 调度器
│   │   └── interpreter.py         # 指令解释器
│   ├── mind/
│   │   └── mind_core.py           # 主脑提示词拼接器
│   ├── memory/
│   │   ├── unified_memory.py      # 统一记忆系统
│   │   ├── long_term_memory.py    # 长期记忆
│   │   └── memory_retriever.py   # 记忆检索器
│   └── context/
│       └── unified_context.py     # 统一上下文
├── agent/
│   ├── planner/
│   │   └── main_planner.py        # 主规划器（使用 Mind）
│   └── world_model.py             # 世界模型
└── cortices/
    └── qq_chat/
        └── cortex.py             # QQ Cortex（纯感知执行）

data/
└── identity.md                   # 人设文件
```

## Shell 指令示例

```bash
# 创建任务
task create --cortex qq --target user_123 --pri high --motive "回复问候"

# 执行任务
task exec --id task_001 --entry chat_reply

# 推入行为
action push --task_id task_001 --action_id act_001 --action_type sequential --skill chat --steps ["send_message", "wait_reply"]

# 存储记忆
memory store --content "用户喜欢看动漫" --tags ["interest", "anime"] --importance 0.8 --cortex qq --target user_123

# 跨域请求
cross_domain request --from task_001 --to task_002 --payload "需要查询的信息"
```

## 启动流程

```
main.py
    └→ MainSystem.initialize()
         ├→ 注册核心组件
         ├→ 初始化 UnifiedMemory
         ├→ 初始化 Mind 主脑系统
         ├→ 初始化 EventLoop
         ├→ 加载 Cortex
         └──> agent_loop.start()
              └──> EventLoop.start()
                   ├→ 启动事件循环任务 (处理中断)
                   └→ 启动空闲调度器 (无任务时调用 Motive)
```

## 关键特性

1. **真正的事件驱动**: 只在有信号时处理，闲置时通过 idle_scheduler 维持基本活动
2. **单注意力头**: Scheduler 选择一个焦点任务执行
3. **统一上下文**: Mind 系统拼接完整的主脑提示词
4. **层级优先级**: critical > high > medium > low
5. **兼容性**: 保留旧的 AgentLoop 接口，内部使用新的 EventLoop
