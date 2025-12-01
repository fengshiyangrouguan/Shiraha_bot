# 自主代理统一架构设计文档 (v3)

## 1. 核心愿景与目标

本文档旨在定义一个高级自主代理（Autonomous Agent）的统一软件架构。该代理将超越传统“问答式”或“指令式”的聊天机器人（Chatbot），进化为一个拥有内在动机、能够主动感知外部世界、自主规划并执行复杂任务的“数字生命体”。

- **从被动到主动**：代理的核心驱动力源于其内部的“动机引擎”，而非等待外部用户的具体指令。
- **从对话到行动**：代理不仅能生成对话，更能通过使用工具集（Tools）与外部世界（如B站、贴吧、QQ等平台）进行实际的交互和操作。
- **从孤立到情境感知**：代理拥有一个“世界模型”（World Model），能够持续吸收和理解外部信息，形成对世界的认知，并基于这种认知做出决策。

## 2. 核心设计原则

我们确立了以下几点作为架构的基石：

1.  **自主循环 (Autonomous Loop)**：代理的生命由一个内在的“心跳”（`Heartbeat`）驱动。在每个心跳周期，代理会主动“感知世界”->“生成动机”->“规划行动”，形成一个完整的认知与行动闭环。

2.  **分层规划 (Hierarchical Planning)**：Agent 的规划能力是分层的。
    - **主规划器 (Main Planner)**：负责宏观的战略决策，决定“做什么”。
    - **子规划器 (Sub-Planners)**：负责特定场景下的战术执行，决定“怎么做”。

3.  **插件化特性 (Plugin-based Features)**：所有复杂的功能（如QQ聊天、网页浏览）都被封装成独立的、自包含的“特性包”（Features）。这使得系统高度模块化、易于扩展。

4.  **作用域工具集 (Scoped Toolsets)**：每个规划器只能访问其“作用域”内的工具。主规划器使用宏观的“入口工具”（如`enter_chat_mode`），而子规划器使用微观的执行工具（如`send_message`）。

5.  **抢占式中断 (Preemptive Interruption)**：为应对需要立即处理的紧急外部事件（如被@），系统设有一个中断机制，可以强制中断当前任务，让代理立即重新进入“感知-动机-规划”循环以处理更高优先级的事务。

## 3. 高层架构图

```mermaid
graph TD
    subgraph Agent Core (src/agent)
        A[AgentLoop] -- 驱动 --> B(MotiveEngine);
        B -- 生成意图 --> C(MainPlanner);
        C -- 使用工具 --> D[CortexManager];
        A -- 更新/读取 --> E[WorldModel];
        B -- 读取 --> E;
        C -- 读取 --> E;
    end

    subgraph Feature Modules (src/features)
        F1[QQ Chat Feature]
        F2[Web Browsing Feature]
        F3[...]
    end

    D -- 注册/管理 --> F1;
    D -- 注册/管理 --> F2;
    D -- 注册/管理 --> F3;

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#9c9,stroke:#333,stroke-width:4px
```

## 4. 核心模块详解

### 4.1. `src/agent/` - 代理核心心智

这是 Agent 的大脑和灵魂所在。

-   **`agent_loop.py`**: `AgentLoop` 类是代理的“身体”，是整个系统的总协调者和主循环的承载者。它由一个外部调度器（Scheduler）按“心跳”节奏唤醒，并负责监听紧急“中断”信号。
-   **`motive_engine.py`**: `MotiveEngine` 类是代理的“灵魂”。它在每个心跳周期开始时，根据 `WorldModel` 的最新状态、上次行动的总结和自身的核心身份（人格、长期目标），生成一个最高阶的、模糊的意图（Intent）。
-   **`world_model.py`**: `WorldModel` 类是代理的“记忆中枢”。它负责存储和管理代理对世界的所有认知，包括短期记忆（草稿纸）、长期记忆（知识库）、以及从外部感知到的实时状态（如QQ未读消息数）。
-   **`planners/`**: 所有规划器的家。
    -   `base_planner.py`: `BasePlanner` 抽象基类，定义了所有规划器共享的 ReAct 循环逻辑。
    -   `main_planner.py`: `MainPlanner` 类，继承自 `BasePlanner`，是唯一的、负责战略决策的主规划器。

### 4.2. `src/features/` - 可插拔的特性模块

这是 Agent 能力的来源。每个子目录都是一个高内聚的、自包含的功能单元。

-   **`qq_chat/` (示例)**:
    -   `manifest.json`: 特性清单文件。用于被 `CortexManager` 自动发现和注册。它声明了特性名称、入口工具、子规划器路径、工具作用域等元数据。
    -   `planner.py`: 包含 `QQChatSubPlanner` 类，它继承自 `agent.planners.base_planner`，但重写了部分方法以优化聊天场景。
    -   `tools.py`: 定义并实现 `send_message`, `send_emoji` 等仅在 `qq_chat` 作用域下可用的微观工具，和在main_planner下可用的工具。

### 4.3. `CortexManager` - 特性与工具的管理者

`CortexManager` 是一个在 Agent 启动时初始化的核心服务，它取代了传统的 `ToolManager`。

-   **自动发现与注册**: 启动时，它会扫描 `src/cortices/` 目录，解析每个特性包的 `manifest.json`。
-   **作用域管理**: 它根据清单文件，将不同特性的微观工具注册到各自的作用域下（如 `'qq_chat'`, `'web_browsing'`）。
-   **入口工具生成**: 它会为每个特性动态创建一个“入口工具”（如 `enter_qq_chat_mode`），并将其注册到主规划器的 `'main'` 作用域下。
-   **委派执行**: 当主规划器调用某个入口工具时，`CortexManager` 负责实例化对应的子规划器、切换工具作用域、执行子规划器的任务，并在结束后将作用域切回。

## 5. AgentLoop 详细工作流程图

此流程图展示了我们设计的“周期性感知与抢占式中断模型”。

```mermaid
graph TD
    subgraph "持续运行的后台服务"
        P1[MessageRelayService] -->|接收到普通消息| DB[(WorldModel 数据库)];
        P1 -->|接收到@、紧急等重要消息| I{中断信号 Event};
    end

    subgraph "AgentLoop 主循环"
        S(Start) --> A{等待心跳/中断};
        
        T[外部调度器<br>Scheduler] -- 每N秒 --> A;
        I -- 立即触发 --> A;

        A --> B["1. 感知<br>world_model.get_summary()"];
        B --> C["2. 动机<br>motive_engine.generate_intent()"];
        C --> D["3. 规划与执行<br>main_planner.execute_plan()"];
        
        subgraph "MainPlanner 内部 ReAct 循环"
            D --> D1{构建宏观上下文};
            D1 --> D2{LLM 决策};
            D2 --> D3{选择宏观工具};
        end

        D3 -- 调用普通工具<br>如 get_weather() --> D4[执行并获取结果];
        D4 --> D1;

        D3 -- 调用入口工具<br>如 enter_chat_mode() --> E[CortexManager 委派];
        
        subgraph "SubPlanner 任务"
            E --> E1[1. 切换工具作用域];
            E1 --> E2[2. 实例化子规划器];
            E2 --> E3[3. 执行子规划器 ReAct 循环];
            E3 --> E4[4. 任务完成，返回总结];
            E4 --> E5[5. 恢复主作用域];
        end

        E5 --> D4;
        
        D -- 意图完成 --> F(休眠);
        F --> A;
    end

    subgraph "中断处理"
        I -.-> D;
        I -.-> E3;
        style I fill:#f00,color:#fff
    end
    
    linkStyle 10 stroke:#f00,stroke-width:2px,color:red;
    linkStyle 11 stroke:#f00,stroke-width:2px,color:red;

    note right of I
      中断信号会立即打断
      MainPlanner (D) 或
      SubPlanner (E3) 的
      当前任务，强制循环
      返回到 A 重新开始。
    end
```

---
**流程解释**:
1.  **等待唤醒**: `AgentLoop` 平时处于休眠状态，等待“心跳”或“中断”信号将其唤醒。
2.  **感知**: 唤醒后，做的第一件事是调用 `world_model.get_summary()`，全面了解外部世界的最新变化和自己的内部状态（“睁眼看世界”）。
3.  **动机**: 将最新的世界摘要交给 `motive_engine`，生成一个当前最重要的高阶意图。
4.  **主规划**: `main_planner` 开始执行这个意图。在它的 ReAct 循环中，它会调用 `'main'` 作用域下的宏观工具。
5.  **委派/执行**: 
    - 如果是普通工具，直接执行并获取结果，继续主规划循环。
    - 如果是“入口工具”，则由 `CortexManager` 委派给相应的**子规划器**。子规划器在自己的作用域和工具集下完成一个复杂的子任务，然后返回一个总结。
6.  **完成/休眠**: 当主规划器判断高阶意图已经完成后，`AgentLoop` 进入休眠，等待下一次唤醒。
7.  **中断**: 如果在任何时候（无论是在主规划还是子规划中）收到了“中断信号”，当前正在执行的任务会被立即取消，`AgentLoop` 会被强制拉回到步骤 `A`，以最高优先级处理新的世界状态。



#         logger.info(f"""
# --------------------------------
# 全部系统初始化完成，{self.global_config.bot.nickname}已成功唤醒
# --------------------------------
# 如果想要自定义{self.global_config.bot.nickname}的功能,请查阅：https://docs.mai-mai.org/manual/usage/
# 或者遇到了问题，请访问我们的文档:https://docs.mai-mai.org/
# --------------------------------
# 如果你想要编写或了解插件相关内容，请访问开发文档https://docs.mai-mai.org/develop/
# --------------------------------
# 如果你需要查阅模型的消耗以及麦麦的统计数据，请访问根目录的maibot_statistics.html文件
# """)