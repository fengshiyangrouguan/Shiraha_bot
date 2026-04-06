# Shiraha Bot 单 Planner 原子动作架构

## 整体架构图

```text
┌──────────────────────────────────────────────────────────────────────┐
│                         User Interaction Layer                       │
│                  QQ / 终端调试台 / 定时器 / 其他外部刺激源             │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Cortex Layer（感知/执行层）                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ QQ Chat      │  │ Reading      │  │ Schedule...  │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
│  职责：                                                               │
│  1. 监听外部世界                                                       │
│  2. 发出标准信号                                                       │
│  3. 暴露原子动作与静态控制面板                                         │
│                                                                      │
│  禁止承担主链编排，不再持有复杂的内部回复流程或桥接层。                 │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ emit_signal()
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  MainSystem / InterruptHandler                        │
│                                                                      │
│  MainSystem 负责装配系统边界：                                         │
│  - 注册容器依赖                                                        │
│  - 加载 Cortex / 插件                                                  │
│  - 把 Cortex 信号转交 InterruptHandler                                 │
│                                                                      │
│  InterruptHandler 负责：                                               │
│  - 接收标准化外部事件                                                  │
│  - 按优先级转成 EventLoop 可消费的中断信号                              │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ submit_interrupt()
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        EventLoop（系统主循环）                         │
│                                                                      │
│  ┌──────────────────────┐  ┌──────────────────────┐                 │
│  │ Interrupt Queue      │  │ Debug Queue          │                 │
│  │ 中断信号队列          │  │ 调试输入队列          │                 │
│  └──────────────────────┘  └──────────────────────┘                 │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Idle Scheduler                                                │  │
│  │ 仅在无 FOCUS / 无 READY 任务时调用 Motive 生成动机             │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  EventLoop 是唯一正式主循环，统一处理：                               │
│  - interrupt_input                                                  │
│  - debug_input                                                      │
│  - idle_input                                                       │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
┌───────────────────────────────────┐  ┌───────────────────────────────┐
│    ReplyRuntime（监听型回复器）    │  │    MainPlanner（唯一规划器）   │
│                                   │  │                               │
│  高频社交消息优先由它处理：         │  │  接收统一上下文，输出 JSON：    │
│  - 快速判断是否需要回应             │  │  {                             │
│  - 读取局部会话上下文               │  │    "thought": "...",          │
│  - 调用 ReplyPlanner 生成回复        │  │    "shell_commands": [...]    │
│  - 失败时再升级给主 Planner         │  │  }                             │
└───────────────────────────────────┘  └───────────────┬───────────────┘
                                                        │
                                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  ContextBuilder（统一上下文构建器）                   │
│                                                                      │
│  固定拼接：                                                           │
│  1. 身份信息 identity.md                                              │
│  2. 当前运行态                                                        │
│  3. Kernel Shell 协议说明                                             │
│  4. Cortex 动态摘要                                                   │
│  5. UnifiedMemory 检索结果                                            │
│  6. 当前工作上下文（输入来源/活跃任务/上次观察/最新信号）              │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  KernelInterpreter + Task/Scheduler                   │
│                                                                      │
│  KernelInterpreter 负责解释执行 Planner JSON 中的 shell_commands：     │
│  - task create --mode ...                                            │
│  - task exec --id ...                                                │
│  - task run --cortex ... --action ...                                │
│  - task view --cortex ... --panel ...                                │
│                                                                      │
│  TaskManager / TaskStore / Scheduler 负责：                           │
│  - 任务创建与生命周期                                                 │
│  - 单焦点调度                                                         │
│  - 任务上下文窗口维护                                                 │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       UnifiedMemory / WorldModel                      │
│                                                                      │
│  UnifiedMemory：统一记忆的存储与检索                                  │
│  WorldModel：系统状态快照、最近观察、Cortex 摘要、任务快照             │
└──────────────────────────────────────────────────────────────────────┘
```

## 设计目标

### 1. 只有一个总 Planner

- 系统不再以多态 `Action` 对象树作为正式运行模型。
- 主 Planner 是唯一跨域决策入口。
- Planner 可以接受三类输入：
  - `idle_input`：空闲时由 MotiveEngine 提供的动机输入
  - `debug_input`：开发者调试台直接输入
  - `interrupt_input`：由 Cortex 信号转换而来的中断输入

### 2. Task 只作为运行容器

- Task 不再维护复杂行为栈。
- Task 的职责缩减为：
  - 生命周期管理
  - 优先级管理
  - 运行模式声明
  - 当前任务窗口上下文
  - 最近信号、最近结果、最近观察
  - 已显式查看过的控制面板缓存

### 3. Cortex 只负责领域原子能力

- Cortex 只暴露两类能力：
  - 原子动作
  - 静态控制面板
- Cortex 不再承担主链编排职责。
- 例如：
  - `qq_chat`：发消息、发表情、免打扰会话、查看会话列表
  - `reading`：读一小段、标记暂不阅读、查看书架

### 4. 高频回复走监听型回复器

- 高频社交消息优先由 `ReplyRuntime` 处理。
- 回复器作为监听型任务运行，不频繁唤醒主 Planner。
- 只有遇到无法自行处理的问题，才升级为主链中断式规划。

### 5. 上下文统一但按需展开

- 上下文格式统一为 `{role, content}`。
- 但不会把所有信息一次性全量注入。
- 系统只向主 Planner 注入高价值、当前确实需要的上下文。
- 控制面板内容必须显式 `task view` 后才进入任务窗口，避免 Planner 幻觉式编造。

## 核心组件说明

### 1. MainSystem

文件：
- `main.py`
- `src/main_system.py`

职责：
- 作为系统总装配器
- 初始化数据库、配置、统一记忆、世界模型
- 注册 `TaskStore`、`TaskManager`、`Scheduler`、`InterruptHandler`、`KernelInterpreter`、`EventLoop`
- 加载插件
- 加载全部 Cortex
- 启动 EventLoop

关键边界：
- MainSystem 是装配器，不是行为决策器。
- 它只负责把各组件按正确依赖顺序连起来。
- Cortex 发出的信号也先进入 MainSystem，再转发给 InterruptHandler，保证感知层和内核层边界稳定。

### 2. CortexManager

文件：
- `src/cortex_system/manager.py`

职责：
- 扫描 `src/cortices/` 下的 Cortex 实现
- 读取 manifest 与配置
- 实例化 Cortex
- 调用 `setup(config, signal_callback, skill_manager)` 完成装配
- 收集 Cortex 能力摘要
- 向统一工具表注册原生工具

新增能力：
- `get_cortex(cortex_name)`：读取已加载 Cortex
- `execute_atomic_action(cortex_name, action_name, **kwargs)`：执行原子动作
- `execute_panel_view(cortex_name, panel_name, **kwargs)`：执行控制面板查看

### 3. InterruptHandler

文件：
- `src/core/kernel/interrupt_handler.py`

职责：
- 把外部事件整理为内核信号
- 统一信号入口，避免 Cortex 直接操作调度器或任务系统
- 把标准化信号提交给 EventLoop

设计意义：
- 让 Cortex 只管“感知到什么”，而不管“系统应该怎么调度”
- 让信号优先级与调度规则统一留在内核层

### 4. EventLoop

文件：
- `src/core/kernel/event_loop.py`

职责：
- 作为系统唯一正式主循环
- 统一处理三类输入：
  - 中断信号队列
  - 调试输入队列
  - 空闲状态下的动机生成
- 调用主 Planner，并交给解释器执行其输出
- 在高频 QQ 消息场景下，优先让 ReplyRuntime 接管

EventLoop 内部包含三条子循环：

#### 4.1 中断处理循环

- 从 `_interrupt_queue` 取出信号
- 规范化 cortex 名称
- 先创建或更新对应 task
- 把中断内容写入该 task 的 `task_window`
- 优先交给 `ReplyRuntime`
- 若回复器未处理或升级，则调用主 Planner

#### 4.2 调试输入循环

- 从 `_debug_queue` 获取开发者输入
- 直接作为 `debug_input` 交给主 Planner
- 不依赖外部平台，不需要经过 Cortex

#### 4.3 空闲调度循环

- 当系统中没有 `FOCUS` 且没有 `READY` 任务时，判定为“当前可空闲规划”
- 调用 `MotiveEngine.generate_motive()`
- 若拿到动机，则刷新 WorldModel 快照并调用主 Planner

### 5. MainPlanner

文件：
- `src/agent/planner/main_planner.py`
- `src/agent/planner/planner_result.py`

职责：
- 作为唯一总 Planner
- 接收统一输入
- 调用 ContextBuilder 构建主链上下文
- 请求 LLM
- 把结果解析为结构化 `PlannerResult`

Planner 的正式输出协议：

```json
{
  "thought": "面对当前输入时的想法、判断或动机",
  "shell_commands": [
    "task exec --id reply_listener_qq_main"
  ]
}
```

其中：
- `thought` 用来显式表达系统当前的判断
- `shell_commands` 才是后续可执行动作

### 6. ContextBuilder

文件：
- `src/core/context/context_builder.py`

职责：
- 作为新的唯一主链上下文拼接器
- 直接替代旧双轨结构中“Mind 拼 prompt + ContextBuilder”并存的问题
- 固定输出标准消息列表

上下文固定由以下部分组成：

#### 6.1 身份提示

- 来自 `data/identity.md`
- 若文件缺失则使用默认身份

#### 6.2 当前运行态

- 当前时间
- 当前架构模式
- 注意力模型
- 输出协议
- 可选的 mood / energy

#### 6.3 Kernel Shell 协议说明

明确告诉 Planner 当前只允许使用的新命令族：
- `task create --mode ...`
- `task exec ...`
- `task run ...`
- `task view ...`
- `memory store / retrieve ...`

#### 6.4 Cortex 摘要

- 来自 CortexManager 的动态汇总
- 用于告诉 Planner 当前有哪些领域能力可用

#### 6.5 统一记忆结果

- 从 UnifiedMemory 检索近期重要任务、对话和观察
- 使用 `memory` 角色注入

#### 6.6 当前工作上下文

包括：
- 当前输入来源
- 当前动机
- 活跃任务摘要
- 最新中断信号
- 调试输入
- 上一次观察

### 7. Task / TaskManager / TaskStore / Scheduler

文件：
- `src/core/task/models.py`
- `src/core/task/task_manager.py`
- `src/core/task/task_store.py`
- `src/core/kernel/scheduler.py`

#### 7.1 Task 数据结构

当前 Task 是一个“运行容器”，主要字段包括：
- `task_id`
- `target_id`
- `cortex`
- `priority`
- `status`
- `mode`
- `motive`
- `task_window`
- `task_config`
- `last_signal`
- `last_observation`
- `last_result`
- `view_cache`
- `execution_count`

#### 7.2 四种 Task Mode

- `once`
  - 一次性长链任务
  - 适合查询、跨域处理、一次性流程执行
- `listen`
  - 监听型任务
  - 适合长期守候某类信号，例如回复器
- `loop`
  - 低优先级循环任务
  - 适合持续读书、缓慢整理等行为
- `cron`
  - 定时触发任务
  - 用于时间驱动场景

#### 7.3 调度语义

- `Scheduler` 维持单焦点调度
- 同一时刻主系统仍按“一个焦点任务”推进
- 其他任务可以处于 `READY / BACKGROUND / SUSPENDED / BLOCKED / MUTED`

#### 7.4 任务窗口

- `task_window` 是当前任务的局部上下文窗口
- 中断、面板查看结果、原子动作结果都会写入这里
- 这代表“当前任务真正看过、做过、得到过什么”

### 8. KernelInterpreter

文件：
- `src/core/kernel/interpreter.py`

职责：
- 执行 Planner 输出的 `shell_commands`
- 同时兼容：
  - `PlannerResult`
  - dict
  - JSON 字符串
  - 纯文本 shell

当前主协议：

#### 8.1 任务生命周期命令

```bash
task create --cortex qq_chat --target group_1 --mode listen --pri high --motive "监听群消息"
task exec --id task_xxx
task suspend --id task_xxx
task resume --id task_xxx
task block --id task_xxx
task kill --id task_xxx
task adjust_prio --id task_xxx --pri high
```

#### 8.2 原子动作命令

```bash
task run --cortex qq_chat --action send_message --conversation_id group_1 --content "你好"
task run --cortex reading --action read_book_chunk --book_id novel_a
```

#### 8.3 控制面板命令

```bash
task view --cortex qq_chat --panel view_conversation_list --task_id task_xxx
task view --cortex reading --panel view_bookshelf --task_id task_xxx
```

#### 8.4 记忆命令

```bash
memory store --content "用户喜欢科幻小说" --type long_term --cortex qq_chat --target group_1
memory retrieve --query "最近对话和偏好" --limit 5
```

兼容说明：
- 旧 `action push / action complete` 已不再是主链协议
- 解释器只保留降级提示，不再把它们当成核心执行模型

### 9. ReplyRuntime / ReplyPlanner

文件：
- `src/core/reply/reply_runtime.py`
- `src/core/reply/reply_planner.py`

#### 9.1 设计目的

- 把高频社交回复与主 Planner 解耦
- 让系统对普通聊天具备“局部自治处理”能力
- 避免每条消息都打断整个主脑

#### 9.2 ReplyRuntime 的职责

- 接收 QQ 相关信号
- 为对应会话创建或复用 `listen` 任务
- 把消息写入该任务窗口
- 先做启发式快速判断：
  - 是否 @ 我
  - 是否提问
  - 是否明显点名请求
- 如果快速判断不需要回复，则直接沉默，不升级给主 Planner
- 如果需要回复，则读取会话上下文，调用 ReplyPlanner 生成回复方案
- 最后调用 QQ Cortex 原子能力发送回复
- 如果任何步骤失败，则升级为主 Planner 中断处理

#### 9.3 ReplyPlanner 的职责

- 在局部会话上下文中规划“要不要回复、回复什么、用什么风格”
- 支持输出如：
  - `should_reply`
  - `reply_content`
  - `style`
  - `use_emoji`

这层代表“回复策略层”，不属于 Cortex 的原子动作层。

### 10. UnifiedMemory / WorldModel

文件：
- `src/core/memory/unified_memory.py`
- `src/core/memory/long_term_memory.py`
- `src/core/memory/memory_retriever.py`
- `src/agent/world_model.py`

#### 10.1 UnifiedMemory

职责：
- 统一短期/长期记忆接口
- 为 Planner、回复器、Cortex 提供统一检索入口
- 支持存储、检索、遗忘

#### 10.2 WorldModel

职责：
- 记录最近观察
- 记录任务快照
- 记录 Cortex 摘要
- 为 Planner 提供当前系统全景状态

WorldModel 不是行动者，而是主链的“状态镜子”。

## 当前 Cortex 设计

### 1. QQ Chat Cortex

文件：
- `src/cortices/qq_chat/cortex.py`
- `src/cortices/qq_chat/tools/basic_tools.py`

当前角色：
- 负责接入 QQ 平台消息
- 缓存最近消息
- 为回复器提供 `get_conversation_context()`
- 暴露原子动作和面板

已公开的典型能力：
- 原子动作
  - `send_message`
  - `send_emoji`
  - `mute_conversation`
  - `get_messages`
  - `get_conversation_info`
  - `casual_reply_bundle`
- 控制面板
  - `view_conversation_list`

### 2. Reading Cortex

文件：
- `src/cortices/reading/cortex.py`
- `src/cortices/reading/tools/atomic_tools.py`

当前角色：
- 提供阅读领域的原子能力
- 不再承担复杂的阅读流程编排

已公开的典型能力：
- 原子动作
  - `read_book_chunk`
  - `mark_book_dormant`
- 控制面板
  - `view_bookshelf`

## 核心运行流程

### 1. 启动流程

```text
main.py
    ↓
MainSystem.initialize()
    ├─ 初始化配置
    ├─ 初始化数据库 / ToolRegistry / PlatformManager
    ├─ 初始化 UnifiedMemory
    ├─ 初始化 WorldModel
    ├─ 注册 TaskStore / TaskManager / Scheduler / InterruptHandler
    ├─ 注册 CortexManager / KernelInterpreter / EventLoop
    ├─ 绑定 InterruptHandler -> EventLoop
    ├─ 加载插件
    ├─ 加载全部 Cortex
    └─ 启动 EventLoop
```

### 2. 外部消息进入后的流程

```text
QQ Cortex 收到消息
    ↓
emit_signal()
    ↓
MainSystem._handle_cortex_signal()
    ↓
InterruptHandler.handle_external_event()
    ↓
EventLoop.submit_interrupt()
    ↓
EventLoop._handle_interrupt_signal()
    ├─ 创建或更新对应 Task
    ├─ 写入 task_window
    ├─ 优先交给 ReplyRuntime
    │    ├─ 若已自行处理完成 → 结束
    │    └─ 若无法处理 → 升级给 MainPlanner
    ↓
MainPlanner.plan()
    ↓
返回 {thought, shell_commands}
    ↓
KernelInterpreter.execute_batch()
    ↓
Task / Memory / Cortex 执行结果回写
    ↓
WorldModel.set_last_observation()
```

### 3. 空闲时的自主流程

```text
Idle Scheduler 周期检查
    ↓
当前无 FOCUS 任务
当前无 READY 任务
    ↓
MotiveEngine.generate_motive()
    ↓
WorldModel.refresh_task_snapshots()
    ↓
MainPlanner.plan(input_source="idle_input")
    ↓
返回 JSON 规划结果
    ↓
KernelInterpreter.execute_batch()
```

### 4. 调试台输入流程

```text
开发者调用 submit_debug_request()
    ↓
EventLoop.submit_debug_input()
    ↓
EventLoop._run_debug_loop()
    ↓
MainPlanner.plan(input_source="debug_input")
    ↓
返回 JSON 规划结果
    ↓
KernelInterpreter.execute_batch()
```

## 协议设计

### 1. Planner 输入协议

主 Planner 当前接收的关键输入字段包括：
- `motive`
- `previous_observation`
- `input_source`
- `latest_signal`
- `debug_request`

这些字段被 ContextBuilder 转换为统一上下文。

### 2. Planner 输出协议

正式结构：

```json
{
  "thought": "当前面对最新消息/信号/动机时的判断",
  "shell_commands": [
    "task create --cortex qq_chat --target group_1001 --mode listen --pri medium --motive \"监听群消息\"",
    "task exec --id task_123"
  ]
}
```

规则：
- 必须优先输出 JSON
- `shell_commands` 必须是字符串数组
- 如果 LLM 没按要求输出，系统会尝试做降级解析

### 3. Task 协议

#### 3.1 create

```bash
task create --cortex <领域> --target <目标ID> --mode <once|listen|loop|cron> --pri <优先级> --motive "<动机>"
```

#### 3.2 exec

```bash
task exec --id <任务ID>
```

语义：
- 让这个任务进入当前执行窗口
- 当前版本主要用于调度和执行次数更新

#### 3.3 run

```bash
task run --cortex <领域> --action <原子动作> [参数...]
```

语义：
- 直接执行一个原子动作
- 若给定 `task_id`，结果会被回写到该任务窗口

#### 3.4 view

```bash
task view --cortex <领域> --panel <控制面板> [参数...]
```

语义：
- 显式读取控制面板
- 若给定 `task_id`，结果会被缓存到 `view_cache`

## 与旧架构的差异

### 1. 旧主链的问题

- 过度依赖 `agent_loop`
- 行为建模分散在 motive / plan / action / observation 多个阶段
- 多态 Action 栈越叠越重
- Cortex 内部常常混入复杂策略逻辑
- QQ 等高频社交场景容易频繁打断主脑

### 2. 新主链的收敛方向

- EventLoop 成为唯一正式主循环
- 主 Planner 成为唯一跨域规划器
- Task 只保留运行与调度语义
- Cortex 只保留原子动作与面板
- 高频回复交给监听型回复器
- 主链上下文统一由 ContextBuilder 构建

## 文件结构

```text
src/
├── main_system.py                     # 主系统装配器
├── agent/
│   ├── motive/
│   │   └── motive_engine.py          # 空闲动机生成
│   ├── planner/
│   │   ├── main_planner.py           # 唯一总 Planner
│   │   └── planner_result.py         # Planner 结构化结果
│   └── world_model.py                # 系统状态镜子
├── core/
│   ├── context/
│   │   ├── context_builder.py        # 统一上下文构建器
│   │   └── unified_context.py        # 上下文消息容器
│   ├── kernel/
│   │   ├── event_loop.py             # 系统主循环
│   │   ├── interrupt_handler.py      # 中断入口
│   │   ├── interpreter.py            # Shell 指令解释器
│   │   └── scheduler.py              # 单焦点调度器
│   ├── memory/
│   │   ├── unified_memory.py         # 统一记忆
│   │   ├── long_term_memory.py       # 长期记忆
│   │   └── memory_retriever.py       # 记忆检索器
│   ├── reply/
│   │   ├── reply_runtime.py          # 监听型回复器运行时
│   │   └── reply_planner.py          # 高级回复规划器
│   └── task/
│       ├── models.py                 # Task / TaskMode / TaskStatus
│       ├── task_manager.py           # 任务管理
│       └── task_store.py             # 任务存储
├── cortex_system/
│   └── manager.py                    # Cortex 加载与原子动作调度
└── cortices/
    ├── qq_chat/
    │   ├── cortex.py                 # QQ 领域 Cortex
    │   └── tools/basic_tools.py      # QQ 原子动作与面板
    └── reading/
        ├── cortex.py                 # 阅读领域 Cortex
        └── tools/atomic_tools.py     # 阅读原子动作与面板
```

## Shell 指令示例

```bash
# 创建一个监听型 QQ 会话任务
task create --cortex qq_chat --target group_123 --mode listen --pri medium --motive "监听会话消息"

# 直接执行一个 QQ 原子动作
task run --cortex qq_chat --action send_message --conversation_id group_123 --content "收到，我在。"

# 查看 QQ 会话列表面板
task view --cortex qq_chat --panel view_conversation_list --task_id task_001

# 创建一个低优先级阅读循环任务
task create --cortex reading --target bookshelf_main --mode loop --pri low --motive "随意读一会儿书"

# 执行一次阅读原子动作
task run --cortex reading --action read_book_chunk --book_id book_demo --task_id task_002

# 存储记忆
memory store --content "某群会话近期在讨论科幻小说" --type long_term --cortex qq_chat --target group_123
```

## 当前实现状态

### 已落地

- EventLoop 成为唯一正式主循环
- MainPlanner 输出 `thought + shell_commands` JSON
- ContextBuilder 成为唯一主链提示词构建器
- Task 新增 `once/listen/loop/cron`
- `task run / task view` 进入主链协议
- QQ 高频消息优先交由 ReplyRuntime
- QQ 与 Reading 已开始原子化改造

### 尚未完全展开

- `cron` 的时间驱动细节仍需继续补强
- 更多 Cortex 还需要按同样模式改造
- 旧 `action` 体系虽然已退出主链，但兼容代码仍在仓库中
- Skill 的“按领域 + 片段加载”目前仍是接口方向，尚需继续细化执行层

## 关键特性

1. 真正的单主循环：主链不再依赖旧 `agent_loop`
2. 单 Planner 决策：跨域规划只有一个入口
3. Task 语义收缩：任务是运行容器，不是行为栈
4. Cortex 原子化：领域层只负责感知、原子动作和静态面板
5. 回复自治：高频消息先由监听型回复器局部处理
6. 上下文可控：统一格式，但按需展开，减少幻觉和噪音
7. 协议清晰：Planner 输出 JSON，Interpreter 执行标准 shell 命令族

