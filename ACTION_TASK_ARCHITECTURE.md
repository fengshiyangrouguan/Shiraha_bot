# Action / Task 架构参考

本文档用于沉淀当前对 `Task`、`Action`、信号接收、状态流转和规划执行链路的最终讨论结果，作为后续开发时的统一框架参考。

---

## 1. 核心结论

这套系统里，`Task` 负责持有状态，`Action` 负责解释状态。

- `Task` 是任务容器，持有：
  - `status`
  - `priority`
  - `context_ref`
  - `actions`
  - `anchors`
- `Action` 是任务当前阶段的行为定义，决定：
  - 当前任务为什么活着
  - 需要什么上下文
  - 对什么信号有反应
  - 执行一轮后应该进入什么状态

`Task.status` 不应该被理解为“任务类型”，而应该被理解为“当前栈顶 Action 对调度器提出的资源请求状态”。

因此：

- `Task` 不主动思考调度语义。
- `Action` 不直接随手修改 `Task.status`。
- `TaskManager` 是唯一的状态落地点。
- `Scheduler` 只负责注意力分配与抢占。
- `InterruptHandler` 只负责接收外部事件并转成内核信号。

---

## 2. 最终的 Action 定义

### 2.1 Action 的职责

一个 `Action` 表示某个 `Task` 当前阶段正在做的事情。

例如：

- 阅读一本书
- 处理一次 QQ 对话
- 等待某个人回复
- 向别的 Task 请求帮助
- 整理一个工具调用结果

`Action` 不是小型调度器，也不是小型任务管理器。  
它只负责两件事：

1. 当前拿到焦点时，如何执行一轮
2. 当前不在焦点时，如何感知外部信号

因此，一个 `Action` 最终只保留两个核心入口：

- `execute(...)`
- `on_perception(...)`

不建议再拆出第三套状态判定方法，否则状态逻辑会散掉。

### 2.2 Action 的推荐字段

一个 `Action` 至少应该可存储一个skill.md的索引，用于传递给子planner
### 2.3 Action 的推荐运行范式

从运行语义上，Action 可以粗分为四类：

- 推进型
  - 例如阅读、写作、规划、整理（常态下ready和focus切换）
  - 常态：`FOCUS -> READY`
- 监听型
  - 例如聊天监听、群消息嗅探、等某类触发词（没有消息/主动聊天注入的信号就退回到background，）
  - 常态：`BACKGROUND`
- 等待型
  - 例如等待工具结果、等待异步回调
  - 常态：`BLOCKED` 或 `BACKGROUND`
- 一次性执行型
  - 例如执行一次回复、执行一次写入、执行一次确认
  - 常态：执行结束后 `FINISH`

注意：这是“行为模式”的划分，不是 `Task` 类型划分。

---

## 3. Action 与 Planner 的关系

### 3.1 统一结论

规划器不应该在 `Action.execute()` 结束后由外部补调。  
更合理的设计是：

`execute()` 本身就是一次完整执行周期的入口，而规划器调用发生在这个执行周期内部。

也就是说：

- `Scheduler` 只决定“谁获得焦点”
- `Action.execute()` 决定“这一轮怎么推进”
- 真正的计划生成由统一 Planner 完成
- `TaskManager` 负责解释执行结果并改状态

### 3.2 BaseAction 中的通用计划骨架

`plan` 这部分应该尽量收敛到 `BaseAction` 的通用方法里。

推荐使用模板方法模式：

- `BaseAction.execute()` 提供统一骨架：
  - 准备标准上下文
  - 注入 Action 对应的 skill / prompt spec
  - 调统一 Planner
  - 解析 Planner 输出
  - 返回统一的 `ActionSignal`

- 子类 Action 只提供差异化内容：
  - 本 Action 绑定哪个 skill
  - 额外注入什么上下文
  - 如何解释本 Action 的完成条件
  - 是否需要特殊的 `on_perception()`

### 3.3 Action 是否应该绑定自己的 SKILL.md

是，推荐绑定。

一个 `Action` 应有自己对应的 `SKILL.md` 或等价的行为说明，用于向统一 Planner 注入：

- 该行为的目标
- 该行为该如何规划
- 该行为允许使用哪些工具
- 该行为的注意事项

但要注意：

- `SKILL.md` 适合描述“如何规划”
- 不适合描述“如何调度”

例如：

- “阅读时先概括章节，再抽取问题，再记录笔记” 适合写在 skill 中
- “执行完是回 READY 还是 BACKGROUND” 不适合写在 skill 中，应当由 `ActionSignal` 和 `TaskManager` 决定

---

## 4. 最终的 ActionSignal 设计

### 4.1 为什么需要 ActionSignal

`Action` 不应直接修改 `Task.status`。  
否则状态修改会分散在各处，后续维护会迅速失控。

因此推荐统一引入 `ActionSignal`：

- `Action` 负责返回意图
- `TaskManager` 负责把意图翻译为 `Task` 的正式状态变更

### 4.2 推荐信号集合

建议至少覆盖以下信号：

- `NOOP`
  - 什么都不做
- `YIELD_READY`
  - 让出焦点，但还想继续推进
- `YIELD_BACKGROUND`
  - 让出焦点，转入后台监听
- `BLOCK`
  - 当前推进不了，等待外部结果
- `WAKE_READY`
  - 在后台或阻塞态中命中信号，请求进入 `READY`
- `FINISH`
  - 当前 Action 完成
- `PUSH_ACTION`
  - 向当前 Task 压入新 Action
- `UPDATE_ANCHOR`
  - 回填锚点结果

### 4.3 信号与状态不是一回事

不要让 `Action` 直接返回裸 `TaskStatus` 作为行为语义。

因为：

- `TaskStatus` 是资源状态
- `ActionSignal` 是行为意图

例如：

- 阅读 Action 一轮执行后，不是简单地“返回 READY”
- 它真正表达的是“我暂时让出焦点，但后续还要继续”，这应是 `YIELD_READY`

同理：

- Chat Action 一轮执行后，不是简单“写 BACKGROUND”
- 它表达的是“我说完了，进入后台监听”，即 `YIELD_BACKGROUND`

---

## 5. 信号由谁接收

### 5.1 分层答案

系统中的信号接收分为三层：

1. `Cortex`
2. `InterruptHandler`
3. `Action.on_perception()`

### 5.2 各层职责

#### Cortex

各个外部适配器负责接收原始世界事件，例如：

- QQ 消息
- Webhook
- 终端输入
- 定时器
- 工具回调

它们只负责把原始事件送进系统，不负责内核调度。

#### InterruptHandler

`InterruptHandler` 是统一的内核信号入口。

它负责：

- 接收来自各个 Cortex 的原始事件
- 做轻量级元数据提取
- 做优先级初筛和中断过滤
- 找出可能相关的 Task
- 触发后续的感知分发或重调度

因此，如果问“谁负责接收外部信号”，统一答案是：

`InterruptHandler` 负责接收进入内核的信号。

#### Action.on_perception()

`on_perception()` 是 Action 的后台感知钩子。  
它负责处理：

“即使当前不在栈顶、没有拿到焦点，但我仍然关心的外部刺激”

这正适合：

- `ask for help` 在栈底时偷听答案
- QQ chat / 等回复 Action 在后台等相关消息
- 等待某类通知或匹配词

---

## 6. on_perception() 什么时候被调用

`on_perception()` 不是定时轮询函数，而是事件驱动函数。

它的最合理调用时机是：

当系统收到外部事件，而某个 Action 当前不在执行态时，对它做一次轻量感知分发。

即：

1. 外部事件到来
2. 进入 `InterruptHandler`
3. 找到相关 `Task`
4. 对这些 Task 中非执行态的 Action 调用 `on_perception(event)`
5. `on_perception()` 返回 `ActionSignal`
6. `TaskManager` 处理该信号

因此：

- 栈顶 Action：通常通过 `execute()` 推进
- 非栈顶 Action：通过 `on_perception()` 做后台感知

`on_perception()` 的设计边界必须严格：

- 允许：
  - 关键词匹配
  - target_id 匹配
  - 简单规则判断
  - embedding / 正则等低成本检测
- 不允许：
  - 大模型推理
  - 大段上下文拼接
  - 直接发消息
  - 复杂调度逻辑

一句话总结：

`on_perception()` 是“外部事件到来时，对非执行态 Action 的低成本后台感知入口”。

---

## 7. Task.status 应该如何修改

### 7.1 最重要的原则

`Task.status` 的修改只应来自两类来源：

1. `ActionSignal`
2. `Kernel` 的资源调度行为

具体地说：

- `Action` 返回行为意图
- `TaskManager` 依据意图修改状态
- `Scheduler` 在焦点切换、抢占时修改资源相关状态

不要让业务层到处直接写 `task.status = ...`。

### 7.2 状态修改的归属

#### Action

不直接写状态，只返回 `ActionSignal`

#### TaskManager

唯一状态落地者，负责：

- 根据 `ActionSignal` 修改 `Task.status`
- 处理 Action 栈变化
- 回填 `anchors`
- 更新优先级

#### Scheduler

只负责：

- `FOCUS`
- `READY`
- `SUSPENDED`

也就是注意力资源的切换与抢占。

#### InterruptHandler

只负责唤醒和转发，不负责复杂状态机落地。

---

## 8. 各类 Action 的典型状态语义

### 8.1 阅读 Action

阅读 Action 属于推进型任务。

特点：

- 核心驱动力不是外部消息，而是继续获得执行时间片
- 失去焦点后通常回 `READY`
- 被更高优先级打断后变成 `SUSPENDED`
- 一般不依赖 `on_perception()`

因此阅读 Action 的 `on_perception()` 默认可以是空实现：

```python
def on_perception(self, data):
    return ActionSignal.noop()
```

### 8.2 Chat Action

Chat Action 更像监听型 / 交互型行为。

特点：

- 说完一轮后，通常不需要一直占着 `READY`
- 更适合退到 `BACKGROUND`
- 只有在对话相关消息到来时，再通过 `on_perception()` 唤醒

因此：

- `execute()` 结束后通常返回 `YIELD_BACKGROUND`
- `on_perception()` 命中相关消息后返回 `WAKE_READY`

### 8.3 AskForHelp Action

这是 `on_perception()` 的典型适用对象。

它即使在栈底，也仍然关心外部是否出现了自己要的答案。

因此：

- 平时处于挂起或后台
- 外部消息进入后执行 `on_perception()`
- 一旦命中，就可以：
  - `FINISH`
  - `UPDATE_ANCHOR`
  - 或请求唤醒原任务

---

## 9. “READY 后又退回 BACKGROUND” 的逻辑该写在哪里

如果某个 Action 的要求是：

“即使自己被设成 READY，只要发现 Task 中并没有真正需要的信号，就应该退回 BACKGROUND”

这段逻辑不应该写在：

- `Scheduler`
- `TaskManager` 的硬编码分支
- `SKILL.md`

最扁平的设计是：

让它写在 `execute()` 的开头。

原因：

- `READY` 只表示调度器给了它一次前台检查的机会
- 真正能不能推进，应由该 Action 自己在 `execute()` 里判断
- 如果条件不成立，`execute()` 直接返回 `YIELD_BACKGROUND`

这样无需额外拆出第三个判定方法，链路最短。

---

## 10. 最终的最扁平化工作链路

推荐把整条链路收敛为下面这套最小模型：

### 10.1 对外事件链路

```text
外部事件
-> Cortex
-> InterruptHandler.handle_external_event(...)
-> 找到相关 Task
-> 对非执行态 Action 调用 on_perception(...)
-> Action 返回 ActionSignal
-> TaskManager.apply_signal(...)
-> Scheduler.schedule()
```

### 10.2 前台执行链路

```text
Scheduler 选中 Task
-> Task 进入 FOCUS
-> 调用栈顶 Action.execute(...)
-> execute 内部调用统一 Planner
-> Planner 产出计划 / 工具调用 / 行为意图
-> execute 返回 ActionSignal
-> TaskManager.apply_signal(...)
-> Scheduler.schedule()
```

### 10.3 统一原则

整套系统最终只保留三个关键抽象：

- `Action`
- `ActionSignal`
- `TaskManager.apply_signal(...)`

其中：

- `Action` 负责执行和感知
- `ActionSignal` 负责表达意图
- `TaskManager.apply_signal(...)` 负责状态落地

这样链路最短，职责边界最稳定。

---

## 11. 推荐的最终分工

### Cortex

- 接收原始外部输入
- 不负责调度

### InterruptHandler

- 统一接收进入内核的信号
- 做轻量判断和唤醒触发

### Scheduler

- 管理焦点与抢占
- 不理解具体业务

### TaskManager

- 统一解释 `ActionSignal`
- 唯一修改 `Task.status`
- 管理 Action 栈和锚点

### BaseAction

- 提供统一 `execute()` 骨架
- 在内部调用标准 Planner

### 子类 Action

- 提供 skill
- 提供少量差异化上下文
- 实现必要的 `on_perception()`

---

## 12. 最终框架原则

为了避免架构继续变重，后续开发遵守以下原则：

1. 不让 `Action` 直接写 `Task.status`
2. 不让 `Scheduler` 理解具体业务 Action
3. 不让 `TaskManager` 写满特例分支
4. 不为状态判定再增加第三套专用方法
5. 所有状态变化尽量统一收口到 `ActionSignal -> TaskManager.apply_signal(...)`
6. `on_perception()` 只做轻量感知，不做重执行
7. 统一 Planner 只保留一套，不为每个 Action 各造一个子脑
8. `SKILL.md` 用于规划提示，不用于资源调度

---

## 13. 一句话总结

最终架构应当是：

`Action` 负责“执行”和“感知”，  
`ActionSignal` 负责“表达意图”，  
`TaskManager` 负责“落状态”，  
`InterruptHandler` 负责“接收信号”，  
`Scheduler` 负责“分配焦点”。

这是当前讨论下最扁平、最稳定、最适合继续演进的方案。
