from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional, List
import time
import uuid

class TaskStatus(Enum):
    """
    任务生命周期状态机。
    定义了任务在调度循环(Event Loop)中的活跃程度及资源占用策略。
    """
    READY = "ready"           # 【就绪】任务已具备执行条件，进入 FIFO 或 优先级调度队列，等待分配 CPU/Token 资源。
    FOCUS = "focus"           # 【焦点】当前主脑正在处理/关注的任务。系统内同时只能有一个任务处于此状态。
    SUSPENDED = "suspended"   # 【挂起】任务被主动或策略性中断。现场存储，不消耗轮询资源。
    BLOCKED = "blocked"       # 【阻塞】任务因外部 IO（如：工具调用、网络搜索）被动停滞。内核持续监控其 Wait_Handle 信号。
    MUTED = "muted"           # 【静默】任务仍活跃，但主脑主动屏蔽它的消息。系统屏蔽该任务及其关联 Cortex 的所有主动上报请求。
    BACKGROUND = "background" # 【后台】感知模式。任务不占焦点，但内核保持其语义嗅探。当触发特定关键字或事件时，自动转为 READY。
    TERMINATED = "terminated" # 【终结】任务生命周期结束。触发垃圾回收(GC)，清理 Context_Ref 关联的临时内存。

    # 笔记：
        # 1. READY 与 FOCUS 的状态切换（空闲调度）：
        #    - 在系统“空闲”时，调度器遵循【公平轮询】原则。从 READY 队列中按进入顺序（FIFO）提取任务。
        #    - 切换逻辑：当前 FOCUS 任务执行完一个动作步长后，转为 READY；
        #    - 调度器从 READY 队列首部取出下一个任务，将其状态设为 FOCUS，实现注意力的平滑轮询。

        # 2. 优先级层级下的 Ready/Focus 轮询：
        #    - 调度器维护多个【优先级分桶】（CRITICAL > HIGH > MEDIUM > LOW）。
        #    - 只有当高优先级桶为空时，才会执行低优先级桶中的任务。
        #    - 在同级桶内，任务以 FOCUS <-> READY 的形式循环切换，确保同级别的社交请求或指令都能得到响应。

        # 3. 高优先级抢占与 SUSPENDED 的“优先复活权”：
        #    - 当 HIGH 指令（如用户直接 @）进入时，当前正在 FOCUS 的 MEDIUM 任务被强制转为 SUSPENDED。
        #    - 【核心逻辑】：当所有高优先级任务处理完毕（TERMINATED）后，调度器在降级寻找下一个任务时，
        #      会优先扫描状态为 SUSPENDED 的任务而非 READY 任务。
        #    - 理由：SUSPENDED 任务代表了“被中断的上下文”，其恢复优先级高于从未启动过的新任务，
        #      这保证了 Agent 能够“续接”之前的思考，而不是不断开启新话题。

        # 4. BLOCKED 与 SUSPENDED 的本质区别：
        #    - BLOCKED 是被阻塞，一般是因为AGENT在子规划中主动决定去别的地方办一件事再回来，并携带上下文，自动设置为阻塞。
        #    - SUSPENDED 是主规划器主动要求挂起，因为有更重要的事情去处理。

        # 5. MUTED 的过滤机制：
        #    - 用于解决“信息过载”。例如群聊中无关消息，任务仍在记录（context_ref 更新），
        #    - 但其产生的任何事件（Event）都不会触发系统级的抢占或 READY 提拔，直到其解除静默。

class Priority(Enum):
    """
    任务调度优先级定义。
    用于内核(Kernel)判断任务的抢占行为、轮询频率以及注意力分配权重。
    """

    # 低优先级：闲暇打发时间的任务级别或后台任务 (如：阅读学习、无关群聊消息存储、长周期知识库索引)
    # 调度表现：仅在系统无 MEDIUM 及以上任务时执行。极易被任何新事件抢占。
    LOW = "low"

    # 中优先级：常规社交与间接交互 (例：私聊信息、提及关键字、普通任务提醒)
    # 调度策略：在同级队列中轮询。确保 Agent 在处理长任务的间隙，能优雅地处理社交请求。
    MEDIUM = "medium"

    # 高优先级：直接指令与显性中断 (例：@艾特本人、用户从终端直接下达的即时指令)
    # 调度策略：立即抢占(Preempt)当前所有 MEDIUM/LOW 任务，触发‘挂起-恢复’流程。
    HIGH = "high"

    # 紧急优先级：系统预警或核心安全 (如：硬件故障报警、核心服务连接断开、安全防御触发、实时电话呼叫)
    # 调度表现：拥有最高执行特权，无条件切断当前一切非 CRITICAL 流程，直至任务解决，该级别不会暴露给主规划器，专属于内核的紧急处理机制，自动设置该优先级。
    CRITICAL = "critical"

class GoalStatus(Enum):
    """
    GoalStatus: 模拟人脑目标的生命周期状态。
    决定了目标在 Task 内部的显隐性、优先级以及是否触发回调。
    """

    # --- 1. 潜意识阶段 (Subconscious Phase) ---
    PENDING = auto()      
    """
    [屏蔽态/挂起]：目标已存在，但处于“潜意识”边缘。
    - 表现：不进入 LLM 上下文，不占用 Focus。
    - 触发：等待小模型检定(check_arousal)命中关键词或语义。
    """

    # --- 2. 显意识阶段 (Conscious Phase) ---
    ACTIVE = auto()       
    """
    [激活态/聚焦]：目标被唤醒，进入“显意识”中心。
    - 表现：目标描述被注入 Context Window，Task 申请 READY 状态请求执行。
    - 触发：检定命中或由其他 Task 强制激活。
    """

    # --- 3. 阻塞执行阶段 (Blocking Phase) ---
    BLOCKING = auto()     
    """
    [专注态/阻塞]：该目标要求 Task 屏蔽低优先级干扰。
    - 表现：Task 进入高阈值监听模式，除了本目标的回传或其他特高优先级信号，不响应任何刺激。
    """

    # --- 4. 终结阶段 (Terminal Phase) ---
    COMPLETED = auto()    
    """
    [达成/交割]：任务逻辑已闭环。
    - 表现：最后一次将结果载荷(Payload)注入上下文，随后准备销毁并回传。
    """

    FAILED = auto()       
    """
    [逻辑失败]：子 Planner 判定该目标在当前环境下无法达成。
    - 表现：打包失败原因，回传给原请求者。
    """

    TIMEOUT = auto()      
    """
    [遗忘/失去耐心]：超过 TTL 设置的时间。
    - 表现：目标被系统强制回收，模拟人脑的“由于等太久而放弃”。
    """

    DISCARDED = auto()    
    """
    [主动放弃]：由于更高优先级的目标冲突，该目标被强制终止。
    - 表现：模拟人脑在紧急情况下丢弃次要念头。
    """

class Goal:
    def __init__(
        self, 
        description: str, 
        priority: Priority = Priority.LOW, 
        timeout: int = 300, 
        origin_task_id: str = None,
        metadata: Dict[str, Any] = None
    ):
        """
        模拟人脑的“目标”或“动机单元”。
        它是 Task 内部的动力源，具备自我检定能力，决定了 Task 何时从“漫无目的”转向“精准聚焦”。
        """

        # 1. 基础身份与动力学属性
        self.goal_id = str(uuid.uuid4())[:8]
        self.description = description
        self.priority = Priority(priority)  # 决定该目标在 Task 内部的排序
        
        # 2. 生命周期与超时管理 (TTL)
        self.created_at = time.time()
        self.timeout = timeout
        self.status = GoalStatus.PENDING
        
        # 3. 异步回传锚点 (Return Pointer)
        # 记录是谁发起的请求，完成后结果该飞回哪里
        self.origin_task_id = origin_task_id
        self.result_payload: Any = None
        
        # 4. 语义检定属性 (用于小模型检定)
        # 存储目标的核心语义特征，用于在“屏蔽态”下匹配外部输入
        self.keywords = self._extract_keywords(description)
        self.embedding_vector = None  # 可选：预留给语义向量空间
        
        # 5. 状态记录
        self.is_blocked = False  # 该目标是否要求所属 Task 进入专注(BLOCK)态
        self.metadata = metadata or {}

    def _extract_keywords(self, text: str):
        # 简单的启发式提取，或者调用极小模型生成关键词
        # 用于 L1/L2 级的快速检定
        return set(text.split()) 

    def check_timeout(self) -> bool:
        """检查目标是否超时"""
        if time.time() - self.created_at > self.timeout:
            self.status = GoalStatus.TIMEOUT
            return True
        return False

    def is_relevant(self, incoming_text: str) -> bool:
        """
        小模型检定逻辑 (L2 Gatekeeper)
        决定当前外部输入是否足以“拉回”注意力
        """
        # 这里可以接入你的 Embedding 相似度计算
        # 或者简单的关键词碰撞逻辑
        for word in self.keywords:
            if word in incoming_text:
                return True
        return False

    def finalize(self, result: Any, success: bool = True):
        """完成目标并打包结果"""
        self.status = GoalStatus.COMPLETED if success else GoalStatus.FAILED
        self.result_payload = result
        self.end_time = time.time()

    def __repr__(self):
        return f"<Goal {self.goal_id}: {self.description[:20]}... [Prio:{self.priority}]>"
    
@dataclass
class Task:
    
    # 指向当前特定的任务实例（如：一次具体的查资料过程）。
    # 作用：用于内核调度、异步回调（Callback）精准归位、以及追踪该任务的生命周期。
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")

    # 【目标逻辑锚点】 (Cross-Platform Anchor)
    # 指向任务服务的一个抽象的最终主体（如：QQ/微信/终端背后的同一用户本身映射出来的代号，而不是qq号，微信号本身）。
    # 核心逻辑笔记：
    # 1. 任务唯一性判定：只有当 (target_id + cortex_domain) 均相同时，视为重复/更新任务。
    # 2. 语义一致性：即使在不同 Cortex（如 QQ 和 微信）下 task_id 不同，
    #    只要 target_id 一致，内核将通过 Prompt 注入告知 主Planner 它们服务于同一个“对象”。
    # 3. 隔离与同步：
    #    - 上下文隔离：不同 task 拥有独立的 context_ref 列表，防止平台间上下文互相污染。
    #    - 缓冲区共享：基于 target_id 绑定一个共享缓冲区 (Shared Buffer)。
    #    - 状态镜像：当旧任务发生注意力转移（Focus Switch）时，内核将当前任务的上下文总结、意图快照更新至 Shared Buffer。
    #    - 状态转移：新目标任务进入 FOCUS 时，优先读取该 target_id 的 Buffer 载入上下文，实现跨平台记忆续接而无需全量加载历史。
    target_id: str = "" 

    # 负责该任务的认知域（Cortex Domain），如：qq、微信、终端等。用于区分不同平台或服务的任务。
    cortex: str = ""   

    # 优先级
    priority: Priority = Priority.LOW
    
    # 状态控制
    status: TaskStatus = TaskStatus.BACKGROUND
    
    # 上下文指针 (Context Ref)
    # 对应数据库中的 session_id 或 内存中的句柄

    #笔记：
    # 无论任务是“查资料”还是“闲聊”，其 context_ref 指向的内存块（或数据库表）都应该遵循一个统一的 Schema。
    # 这样主脑（Planner）在切换 exec 时，不需要适配不同的数据格式。
    # 而且这样可以直接复用子planner

    # 上下文直接用一个类管理，里面有一个dict，key是context_ref，value是一个标准上下文列表。
    # 列表里每个元素通常是一个 Dict，包含以下字段：
    # Role: system, user, assistant, tool (注意：必须包含工具层)。
    # Content: 文本内容。

    context_ref: str = "" # 任务执行上下文的引用标识 (如：ctx_12345)，用于关联对话历史、总结存储等
    
    # 任务元数据 (用于 Summary 或 调试)
    motive: str = ""           # 初始意图
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    goal: List[Goal] = field(default_factory=list)  # 任务内的目标列表，支持多目标管理

    # 空槽位容器：{ "source_goal_id": "result" }
    anchors: Dict[str, str] = field(default_factory=dict) # 用于存放异步回传的锚点数据，如工具调用结果、外部事件等

    def to_dict(self):
        return {k: (v.value if isinstance(v, Enum) else v) for k, v in self.__dict__.items()}
    

    # --- 调度执行策略：高优先级快线 (High-Priority Fast Track) ---
    
    # 笔记 1：身份确权与路径短路 (Identity Trust & Short-circuiting)
    # - 逻辑：一旦 Planner 经过语义识别将某个 Task 提升为 HIGH 优先级，该任务即进入“受信任列表”。
    # - 优化：对于该 target_id 产生的后续增量消息（如补充指令、即时反馈、甚至普通表情包），
    #   内核将【绕过】复杂的语义重估流程。
    # - 目的：消除大模型预判带来的延迟，确保高优先级交互的“零卡顿”响应。

    # 笔记 2：优先级继承 (Priority Inheritance)
    # - 表现：同一 target_id 在当前活跃任务（Active Task）生命周期内的所有输入，默认继承该任务的最高优先级。
    # - 场景：用户在处理紧急生产故障（HIGH）时，随后的每一句回复都必须维持 HIGH 状态，
    #   防止因单句语义较轻（如“收到”、“好的”）而被误判为 LOW，从而导致注意力被其他 MEDIUM 任务抢占。

    # 笔记 3：断点执行与异步沉降 (Breakpoint & Silent Sink)
    # - 抢占发生时：若 LOW 任务正在请求 LLM，不强行掐断连接，而是允许其在后台运行。
    # - 静默存储：后台返回的数据直接进入该 Task 的 context_ref/mailbox 缓冲区。
    # - 恢复即所得：当 HIGH 任务结束，焦点（Focus）切回原任务时，直接从缓冲区提取已完成的结果，
    #   实现“无缝续接”，避免用户感知到二次等待。

    # 笔记 4：衰减与结案机制 (Attenuation & Closure)
    # - 防御：为防止 target_id 长期霸占高优先级权限，引入“动态衰减”。
    # - 触发：若 HIGH 任务在指定时间内（如 5min）无新事件，或 Planner 显式完成任务/降低优先级，
    #   系统将撤销其特权，后续消息回归基础优先级判定流程。


    # --- 调度策略：阻塞态下的优先级退避 (Blocking-Aware Priority) ---
    
    # 笔记 1：优先级有效性 (Priority Validity)
    # - 原则：优先级仅在任务处于 READY 态时具有“竞争意义”。
    # - 逻辑：当任务进入 BLOCK 时，其高优先级进入【休眠期】，不参与当前注意力的分配博弈。
    # - 效果：防止高优先级阻塞任务造成的“系统空转”，释放 CPU/Token 给其他低优先级任务。

    # 笔记 2：事件驱动的瞬时抢占 (Event-Driven Preemption)
    # - 流程：信号到达 -> Task 从 BLOCK 迁至 READY -> 恢复 HIGH 竞争权重。
    # - 调度表现：由于恢复了 HIGH 权重，该任务会在下一个微周期内【强制抢占】当前的 FOCUS 任务。
    # - 结论：这实现了极致的响应速度——“不干活时彻底让位，活到了瞬间抢回”。

    # 笔记 3：状态机状态缩减 (Status Reduction)
    # - 核心三态：
    #   * FOCUS: 正在执行。
    #   * READY: 身份已定，排队等头。
    #   * BLOCK: 身份挂起，等待信号。
    # - 被动结果：SUSPEND（因别人抢占而产生的 READY 态镜像）。




    # --- 执行策略：事件驱动的主动通信 (Event-Driven Proactive Execution) ---
    
    # 笔记 1：反向唤醒机制 (Reverse Wake-up)
    # - 逻辑：exec 一般是由于【外部刺激】，让planner主动决定（如读书的时候想去分享）或者直接调用（如 Webhook, Timer, Context_Match）唤醒。
    # - 状态转换：刺激事件 -> Task.status 从 SUSPEND/BLOCKED 迁至 READY -> 内核分配 FOCUS。
    
    # 笔记 2：消灭定时轮询 (Eliminating Polling)
    # - 优势：取代老旧的 Motive 定时循环逻辑。AI 不再“为了说话而说话”，
    #   而是仅在满足特定逻辑条件（刺激）时，才通过 exec 申请一轮对话权。
    #   这样只会出现一轮申请和一轮回复，然后又回到了BACKGROUND状态单纯等待消息，彻底杜绝了“逻辑死循环”式的连续发言问题。
    



    # --- 架构收敛：主脑原语与职责分离 (Kernel Convergence & Responsibility Layering) ---
    
    # 笔记 1：主脑工具集的语义收敛 (Semantic Convergence)
    # - 原则：主脑仅保留【资源管理】与【意图激活】权。
    # - 核心指令：create, kill, exec, mute, adjust_prio。
    # - 优势：极大地降低了 Kernel 的逻辑复杂度，提高了调度循环的稳定性。

    # 笔记 2：被动挂起的“物理化” (Passive Suspend as Physical Law)
    # - 定义：被动 SUSPEND 不再是一个显式工具，而是【进程管理器】在处理高优先级抢占时的“自然结果”。
    # - 流程：New_High_Prio -> adjust_prio -> Old_Task 自动进入封存状态。

    # 笔记 3：主动挂起的“意图化” (Active Suspend as Intent)
    # - 定义：子 Planner 通过可选携带载荷 (Payload) 的主动调用实现状态变更。
    # - 关键：允许子 Planner 带载调用 `kill`,`suspend`,`block`。
    # - 载荷包含：上下文索引 (Context_Index)、任务遗产 (Legacy_Summary)、建议目标 (Next_Target)。

    # 笔记 4：职责分离的价值 (Value of Separation)
    # - 主脑：只管“谁在跑”，不管“怎么跑”。
    # - 子 Planner：只管“怎么跑”和“跑多久”，不管“别人什么时候跑”。



    # --- 架构精髓：Yield 语义与任务接力 (Yield Semantics & Handover) ---
    
    # 笔记 1：Yield 的本质是“分发”而非“停滞”
    # - 逻辑：子 Planner 调用 yield_focus(payload) 后，立即释放唯一注意力头。
    # - 上下文：Payload 充当单向传输载体，将当前的【中间产物】或【新意图】传递给主脑。
    # - 原任务状态：直接从 FOCUS 回归 READY。

    # 笔记 2：去中心化的任务回归 (Decentralized Resume)
    # - 关键：原任务（A）不依赖新任务（B）的结束信号。
    # - 表现：A 回到 READY 列表后，就像一个新创建的任务一样，公平地参与优先级竞争。
    # - 意义：解决了“分享完必须回来”的刚性逻辑，实现了“分享完可以去干任何事”的灵活性。

    # 笔记 3：主动挂起的消亡 (The Death of Active Suspend)
    # - 结论：在唯一注意力模型下，主动让出焦点即视为 yield。
    # - 区别：
    #   * 想换个地方干活 -> yield(target_id, payload)
    #   * 没活了，休息 -> yield()
    #   * 等零件，干不动 -> block(event_id)

    # 笔记 1：主动权的收敛 (Active Primitives)
    # - 子 Planner 仅需掌握两个核心动作：
    #   1. yield_focus() -> 任务阶段性结束，释放资源，回归 READY 轮询态。
    #   2. block_self(event) -> 逻辑中断，进入 BLOCK 态等待数据。
    # - 结论：主动 SUSPEND 指令被删除，其语义被 yield_focus 完全覆盖。主 Planner 只保留 create, kill, exec, mute, adjust_prio 五个指令，极大简化了主脑的决策树。

    # 笔记 2：被动挂起的物理化 (Passive Suspend as Resource Event)
    # - 逻辑：SUSPEND 状态由【进程管理器】在发生“资源抢占”时自动标记。
    # - 触发：当 HIGH 任务强制 exec 时，原 FOCUS 任务被动进入 SUSPEND。
    # - 特性：这是一种系统级的“现场封存”，对子 Planner 透明（不可见）。

    # 笔记 3：READY 态的灵活性 (The Flexibility of READY)
    # - 状态：READY 意味着“万事俱备，只欠注意力”。
    # - 优势：分享任务结束后，读书任务作为 READY 成员，通过正常的优先级博弈自然回归。
    # - 结果：实现了“分享完不一定马上回”的柔性调度，因为 READY 本身就是排队。

    # 笔记 4：阻塞的特权 (The Privilege of BLOCK)
    # - 逻辑：只有 BLOCK 具有“唤醒锁定”。一旦数据到达，它会强制从列表中跳出来向内核申请 READY 或直接抢占。

# --- 架构演进：目标驱动与实体扁平化 (Goal-Driven Flattening) ---

    # 笔记 2：目标作为动力源 (Goal as Motive Power)
    # - 每一个 Task 的运行不再是随机的，而是由 Goal 列表驱动。
    # - 空目标列表 = 纯粹的无目的回应消息。
    # - 外部注入 = 强制向 Task 写入一个高权重 Goal。

    # 笔记 3：上下文请求的本质 (Context Request)
    # - 跨 Task 请求不再是“寻求帮助的任务”，而是一个【待完成的目标包】。
    # - 逻辑：A 任务 -> 生成 Goal_Object -> 投递给 B 任务。
    # - B 任务在下一次 Focus 时，优先消费优先级最高的 Goal。

    # 笔记 4：优势 - 动态覆盖 (Dynamic Overriding)
    # - 解决了“不知为何而战”的问题。AI 在群聊里的表现完全取决于当前 Goal。
    # - 实现了无缝的思维切换：不需要重启进程，只需要重写当前目标。

# --- 架构细节：多级目标检定 (Multi-level Goal Verification) ---
    
    # 笔记 1：检定解耦 (Verification Decoupling)
    # - 原则：不要让 LLM 负责“判断任务是否结束”的轮询，这太贵了。
    # - 方案：在 Goal 类中内置轻量级检测器（基于关键词或 Embedding）。

    # 笔记 2：观察态的恢复逻辑 (Natural Regression to Observation)
    # - 逻辑：当 Goal 堆栈为空时，Task 自动进入【无固定目标状态】。
    # - 表现：此时的 Planner 仅执行低频的“环境监控”，不再主动发起高能耗思考。

    # 笔记 3：强制检定的时机 (Check Timing)
    # - 被动检定：每一条外部消息进来时，触发小模型过滤。
    # - 主动检定：每次 exec 任务执行完毕，子 Planner 必须显式报告当前 Goal 状态。

    # 笔记 4：小模型防抖 (Small-Model Debouncing)
    # - 策略：如果小模型认为 Goal 结束了，但 LLM 认为没结束，以 LLM 为准。
    # - 目的：防止语义误判导致的“半途而废”。

# --- 架构逻辑：Goal 的潜意识过滤与上下文动态加载 ---
    
    # 笔记 1：上下文的延迟加载 (Lazy Context Loading)
    # - 语义：Goal 在未被激活时，其相关的上下文碎片不进入显意识（LLM Window）。
    # - 目的：节省 Token，防止无关信息干扰主线思维。

    # 笔记 2：检定作为“唤醒钩子” (Verification as Wake-up Hook)
    # - 流程：Input -> Small_Model_Check(Goal_Criteria) -> If Match: Focus_Task().
    # - 特性：这是一种逻辑上的“中断”。只有命中目标的刺激，才能把 Task 从 READY 拽入 FOCUS。

    # 笔记 3：结案清理 (The Wrap-up & Destruction)
    # - 核心：销毁前必须完成【最后一次上下文合并】。
    # - 意义：确保回复给用户或原目标的结果是带有“目标达成确认”的，而非没头没脑的截断。

    # 笔记 4：超时管理 (TTL - Time To Live)
    # - 机制：每一个 Goal 自带 TTL。
    # - 表现：模拟人脑的“遗忘”或“放弃”。如果这件事太久没回音，AI 就不再惦记了。

# --- 架构细节：异步回传锚点与策略化阻塞 ---
    
    # 笔记 1：回传位置指针 (Return Pointer / Callback Anchor)
    # - 逻辑：在 Task 内部建立一个【空槽位】，专门等待特定 target_id 的结果注入。
    # - 效果：任务不需要死等，它可以继续处理其他低优先级的微目标（Micro-goals）。

    # 笔记 2：自主阻塞权限 (Autonomous Blocking Power)
    # - AI 决策：子 Planner 根据目标的【急迫度】决定是否调用 block_self()。
    # - 价值：模拟了人脑的“心不在焉”与“全神贯注”。

    # 笔记 3：状态切换的平滑性 (Seamless Transition)
    # - 过程：B 回传结果 -> 激活 A 的回传锚点 -> A 优先级瞬间拉满。
    # - 体验：就像你正在洗碗（闲聊），突然想起刚才查的公式（结果回传），瞬间擦手去写代码。

    # 笔记 4：防止逻辑孤岛 (Orphan Goal Prevention)
    # - 机制：回传锚点必须带 TTL。如果 B 挂了，A 必须能通过超时机制自我唤醒，
    #   并报错“对方没理我”，而不是永久等下去。

# --- 架构细节：优先级共享与继承逻辑 ---
    
    # 笔记 1：优先级继承 (Priority Inheritance)
    # - 逻辑：Task 的优先级 = max(自身基础分, 内部所有活动 Goal 的最高分)。
    # - 效果：解决了“小任务承载大目标”时的动力不足问题。

    # 笔记 2：动态衰减 (Dynamic Decay)
    # - 模拟：人脑的兴奋感会随时间降低。
    # - 逻辑：某些 Goal 如果长时间未达成，其优先级可以从 HIGH 逐渐降级到 MEDIUM，
    #   避免某个死循环目标永久霸占注意力。

    # 笔记 3：阈值判定 (Threshold Check)
    # - 逻辑：在 BLOCK 期间，只有 Goal.priority > Task.interrupt_threshold 的信号，
    #   才能修改 Task 的状态。