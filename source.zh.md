# LangGraph 多 Agent + Supervisor 系统实现教程

## 目录

1. [引言](#引言)
2. [LangGraph 核心概念](#langgraph-核心概念)
3. [多 Agent 系统架构设计](#多-agent-系统架构设计)
4. [Supervisor 模式详解](#supervisor-模式详解)
5. [完整实现示例](#完整实现示例)
6. [高级特性与优化](#高级特性与优化)
7. [常见问题与最佳实践](#常见问题与最佳实践)

---

## 引言

### 什么是 LangGraph？

LangGraph 是一个用于构建有状态、多参与者应用程序的框架，基于 LangChain 生态系统。它使用图（Graph）的概念来组织复杂的 AI 工作流，特别适合构建需要多个 Agent 协作的复杂系统。

### 为什么使用多 Agent + Supervisor 模式？

在复杂任务场景中，单一 Agent 往往难以高效处理所有问题。多 Agent + Supervisor 模式具有以下优势：

- **专业化分工**：每个 Agent 专注于特定领域（如研究、编码、写作）
- **并行处理**：多个 Agent 可同时处理不同子任务
- **灵活扩展**：易于添加新的专业 Agent
- **集中协调**：Supervisor 负责任务分配和结果整合

### 适用场景

- 复杂的研究与分析任务
- 多步骤的内容创作流程
- 需要多种专业技能的项目
- 企业级自动化工作流

---

## LangGraph 核心概念

### 1. 状态（State）

状态是在图中流转的数据结构，记录了整个工作流的上下文信息。

```python
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """定义 Agent 系统的状态结构"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str  # 下一个要执行的节点
    task: str  # 当前任务描述
    results: dict  # 各 Agent 的执行结果
```

**关键点**：
- `Annotated` 用于定义状态更新策略
- `operator.add` 表示消息列表会累加而非覆盖
- 状态在节点间传递和更新

### 2. 节点（Node）

节点是图中的执行单元，每个节点是一个函数，接收状态并返回更新后的状态。

```python
def researcher_node(state: AgentState) -> AgentState:
    """研究员 Agent 节点"""
    # 执行研究任务
    result = perform_research(state["task"])
    
    return {
        "messages": [HumanMessage(content=f"研究完成: {result}")],
        "results": {**state.get("results", {}), "research": result}
    }
```

### 3. 边（Edge）

边定义了节点之间的连接关系，分为两种：

**普通边（Normal Edge）**：固定的流转路径
```python
graph.add_edge("node_a", "node_b")
```

**条件边（Conditional Edge）**：根据状态动态决定下一个节点
```python
def route_function(state: AgentState) -> str:
    """根据状态决定路由"""
    if state["next"] == "researcher":
        return "researcher_node"
    elif state["next"] == "coder":
        return "coder_node"
    return "end"

graph.add_conditional_edges(
    "supervisor",
    route_function,
    {
        "researcher_node": "researcher_node",
        "coder_node": "coder_node",
        "end": END
    }
)
```

### 4. 图（Graph）

图是整个工作流的容器，组织所有节点和边。

```python
from langgraph.graph import StateGraph, END

# 创建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)

# 设置入口点
workflow.set_entry_point("supervisor")

# 编译图
app = workflow.compile()
```

---

## 多 Agent 系统架构设计

### 系统架构图

```
┌─────────────────────────────────────────┐
│           用户输入任务                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Supervisor (监督者)              │
│  - 分析任务                              │
│  - 选择合适的 Agent                      │
│  - 协调执行流程                          │
└──────┬──────────┬──────────┬────────────┘
       │          │          │
       ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│Researcher│ │  Coder   │ │  Writer  │
│  Agent   │ │  Agent   │ │  Agent   │
│          │ │          │ │          │
│- 信息检索│ │- 代码生成│ │- 内容创作│
│- 数据分析│ │- 代码审查│ │- 文档编写│
└──────┬───┘ └────┬─────┘ └────┬─────┘
       │          │          │
       └──────────┼──────────┘
                  │
                  ▼
         ┌────────────────┐
         │  结果整合与输出 │
         └────────────────┘
```

### Agent 角色定义

#### 1. Supervisor Agent（监督者）

**职责**：
- 理解用户任务
- 将复杂任务分解为子任务
- 选择合适的专业 Agent
- 整合各 Agent 的结果
- 判断任务是否完成

**核心能力**：
- 任务规划能力
- 决策能力
- 协调能力

#### 2. Researcher Agent（研究员）

**职责**：
- 信息检索与收集
- 数据分析
- 背景调研
- 事实核查

**工具**：
- 搜索引擎 API
- 数据库查询
- 文档解析工具

#### 3. Coder Agent（程序员）

**职责**：
- 代码编写
- 代码审查
- 技术方案设计
- 调试与优化

**工具**：
- 代码执行环境
- 代码分析工具
- 文档生成工具

#### 4. Writer Agent（写作者）

**职责**：
- 内容创作
- 文档编写
- 报告生成
- 内容润色

**工具**：
- 模板引擎
- 格式转换工具
- 语法检查工具

---

## Supervisor 模式详解

### Supervisor 的工作流程

```
1. 接收任务
   ↓
2. 分析任务需求
   ↓
3. 选择合适的 Agent
   ↓
4. 分配子任务
   ↓
5. 监控执行进度
   ↓
6. 收集 Agent 结果
   ↓
7. 判断是否需要继续
   ├─ 是 → 返回步骤 3
   └─ 否 → 整合结果并输出
```

### Supervisor 实现要点

#### 1. 任务分析与规划

```python
def analyze_task(task: str) -> dict:
    """分析任务并生成执行计划"""
    prompt = f"""
    分析以下任务，确定需要哪些专业 Agent 参与：
    
    任务：{task}
    
    可用 Agent：
    - researcher: 信息检索、数据分析
    - coder: 代码编写、技术实现
    - writer: 内容创作、文档编写
    
    请返回 JSON 格式的执行计划：
    {{
        "agents_needed": ["agent1", "agent2"],
        "execution_order": ["step1", "step2"],
        "expected_outcome": "预期结果描述"
    }}
    """
    
    # 调用 LLM 进行分析
    response = llm.invoke(prompt)
    return parse_json(response)
```

#### 2. Agent 选择逻辑

```python
def select_next_agent(state: AgentState) -> str:
    """根据当前状态选择下一个 Agent"""
    
    # 检查任务是否完成
    if is_task_complete(state):
        return "FINISH"
    
    # 获取已执行的 Agent
    executed = state.get("executed_agents", [])
    
    # 根据任务计划选择下一个 Agent
    plan = state.get("plan", {})
    remaining_agents = [
        agent for agent in plan["agents_needed"] 
        if agent not in executed
    ]
    
    if remaining_agents:
        return remaining_agents[0]
    
    # 如果需要迭代优化
    if needs_refinement(state):
        return determine_refinement_agent(state)
    
    return "FINISH"
```

#### 3. 结果整合

```python
def integrate_results(state: AgentState) -> str:
    """整合各 Agent 的执行结果"""
    results = state.get("results", {})
    
    prompt = f"""
    请整合以下各专业 Agent 的工作成果，生成最终输出：
    
    研究结果：{results.get('researcher', 'N/A')}
    代码实现：{results.get('coder', 'N/A')}
    文档内容：{results.get('writer', 'N/A')}
    
    原始任务：{state['task']}
    
    请生成完整、连贯的最终结果。
    """
    
    final_output = llm.invoke(prompt)
    return final_output
```

---

## 完整实现示例

### 环境准备

```bash
# 安装依赖
pip install langgraph langchain langchain-openai langchain-community
```

### 完整代码实现

```python
import operator
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json

# ============================================
# 1. 定义状态结构
# ============================================

class AgentState(TypedDict):
    """多 Agent 系统的状态"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str  # 原始任务
    next_agent: str  # 下一个要执行的 Agent
    results: dict  # 各 Agent 的结果
    executed_agents: list  # 已执行的 Agent 列表
    iteration: int  # 迭代次数
    final_output: str  # 最终输出

# ============================================
# 2. 初始化 LLM
# ============================================

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ============================================
# 3. 定义 Supervisor Agent
# ============================================

def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor 负责任务分析和 Agent 调度"""
    
    messages = state["messages"]
    task = state["task"]
    results = state.get("results", {})
    executed = state.get("executed_agents", [])
    
    # 构建 Supervisor 的提示词
    system_prompt = """
    你是一个任务协调者（Supervisor），负责管理多个专业 Agent。
    
    可用的 Agent：
    - researcher: 擅长信息检索、数据分析、背景调研
    - coder: 擅长代码编写、技术实现、算法设计
    - writer: 擅长内容创作、文档编写、报告生成
    
    你的职责：
    1. 分析任务需求
    2. 选择合适的 Agent 执行子任务
    3. 判断任务是否完成
    
    请根据当前情况，返回 JSON 格式的决策：
    {
        "next_agent": "researcher/coder/writer/FINISH",
        "reason": "选择理由",
        "subtask": "分配给该 Agent 的具体子任务"
    }
    """
    
    context = f"""
    原始任务：{task}
    
    已执行的 Agent：{executed}
    
    当前结果：
    {json.dumps(results, ensure_ascii=False, indent=2)}
    
    请决定下一步行动。
    """
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context}
    ])
    
    # 解析决策
    try:
        decision = json.loads(response.content)
    except:
        # 如果解析失败，默认结束
        decision = {"next_agent": "FINISH", "reason": "解析错误", "subtask": ""}
    
    return {
        "messages": [AIMessage(content=f"Supervisor 决策: {decision['reason']}")],
        "next_agent": decision["next_agent"],
        "results": {**results, "supervisor_decision": decision}
    }

# ============================================
# 4. 定义专业 Agent 节点
# ============================================

def researcher_node(state: AgentState) -> AgentState:
    """研究员 Agent - 负责信息检索和分析"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    
    prompt = f"""
    你是一个专业的研究员，擅长信息检索和数据分析。
    
    任务：{subtask}
    
    请进行深入研究，提供详细的分析结果。
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Researcher 完成: {result[:100]}...")],
        "results": {**state["results"], "researcher": result},
        "executed_agents": executed + ["researcher"]
    }

def coder_node(state: AgentState) -> AgentState:
    """程序员 Agent - 负责代码编写"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    research_result = state["results"].get("researcher", "")
    
    prompt = f"""
    你是一个专业的程序员，擅长代码编写和技术实现。
    
    任务：{subtask}
    
    参考信息：{research_result}
    
    请提供完整的代码实现，包含注释和说明。
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Coder 完成: 代码已生成")],
        "results": {**state["results"], "coder": result},
        "executed_agents": executed + ["coder"]
    }

def writer_node(state: AgentState) -> AgentState:
    """写作者 Agent - 负责内容创作"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    research_result = state["results"].get("researcher", "")
    code_result = state["results"].get("coder", "")
    
    prompt = f"""
    你是一个专业的写作者，擅长内容创作和文档编写。
    
    任务：{subtask}
    
    研究资料：{research_result}
    代码实现：{code_result}
    
    请创作高质量的内容或文档。
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Writer 完成: 内容已创作")],
        "results": {**state["results"], "writer": result},
        "executed_agents": executed + ["writer"]
    }

# ============================================
# 5. 定义路由逻辑
# ============================================

def route_agent(state: AgentState) -> Literal["researcher", "coder", "writer", "end"]:
    """根据 Supervisor 的决策路由到相应的 Agent"""
    
    next_agent = state.get("next_agent", "FINISH")
    
    if next_agent == "FINISH":
        return "end"
    elif next_agent == "researcher":
        return "researcher"
    elif next_agent == "coder":
        return "coder"
    elif next_agent == "writer":
        return "writer"
    else:
        return "end"

# ============================================
# 6. 构建工作流图
# ============================================

def create_multi_agent_graph():
    """创建多 Agent + Supervisor 工作流图"""
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("writer", writer_node)
    
    # 设置入口点
    workflow.set_entry_point("supervisor")
    
    # 添加条件边：从 supervisor 根据决策路由到不同 Agent
    workflow.add_conditional_edges(
        "supervisor",
        route_agent,
        {
            "researcher": "researcher",
            "coder": "coder",
            "writer": "writer",
            "end": END
        }
    )
    
    # 添加普通边：各 Agent 完成后返回 supervisor
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("coder", "supervisor")
    workflow.add_edge("writer", "supervisor")
    
    # 编译图
    app = workflow.compile()
    
    return app

# ============================================
# 7. 运行示例
# ============================================

def run_multi_agent_system(task: str):
    """运行多 Agent 系统"""
    
    # 创建工作流
    app = create_multi_agent_graph()
    
    # 初始化状态
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "next_agent": "",
        "results": {},
        "executed_agents": [],
        "iteration": 0,
        "final_output": ""
    }
    
    # 执行工作流
    print(f"🚀 开始执行任务: {task}\n")
    print("=" * 60)
    
    final_state = app.invoke(initial_state)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("📊 执行结果：\n")
    
    for agent, result in final_state["results"].items():
        if agent != "supervisor_decision":
            print(f"\n【{agent.upper()}】")
            print("-" * 60)
            print(result[:500] + "..." if len(result) > 500 else result)
    
    return final_state

# ============================================
# 8. 测试示例
# ============================================

if __name__ == "__main__":
    # 示例任务
    task = """
    创建一个 Python 程序，实现一个简单的待办事项管理系统。
    要求：
    1. 支持添加、删除、查看待办事项
    2. 使用 SQLite 数据库存储
    3. 提供命令行界面
    4. 编写完整的使用文档
    """
    
    result = run_multi_agent_system(task)
```

### 执行流程说明

1. **任务输入**：用户提交复杂任务
2. **Supervisor 分析**：分析任务，决定需要哪些 Agent
3. **Researcher 执行**：收集相关技术资料和最佳实践
4. **返回 Supervisor**：提交研究结果
5. **Supervisor 再决策**：基于研究结果，分配编码任务
6. **Coder 执行**：编写完整代码实现
7. **返回 Supervisor**：提交代码
8. **Supervisor 再决策**：分配文档编写任务
9. **Writer 执行**：编写使用文档
10. **返回 Supervisor**：提交文档
11. **Supervisor 判断完成**：整合所有结果，输出最终成果

---

## 高级特性与优化

### 1. 添加记忆功能

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 创建检查点保存器
memory = SqliteSaver.from_conn_string(":memory:")

# 编译时添加检查点
app = workflow.compile(checkpointer=memory)

# 运行时指定线程 ID
config = {"configurable": {"thread_id": "conversation_1"}}
result = app.invoke(initial_state, config=config)
```

### 2. 并行执行 Agent

```python
from langgraph.graph import START

def create_parallel_graph():
    """创建支持并行执行的图"""
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("aggregator", aggregator_node)
    
    # 从 supervisor 并行分发到多个 Agent
    workflow.add_edge("supervisor", "researcher")
    workflow.add_edge("supervisor", "coder")
    
    # 多个 Agent 完成后汇聚到 aggregator
    workflow.add_edge("researcher", "aggregator")
    workflow.add_edge("coder", "aggregator")
    
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()
```

### 3. 人工审核节点

```python
from langgraph.checkpoint.sqlite import SqliteSaver

def human_review_node(state: AgentState) -> AgentState:
    """人工审核节点"""
    
    # 暂停执行，等待人工输入
    print("\n🔍 需要人工审核：")
    print(f"当前结果：{state['results']}")
    
    approval = input("\n是否批准继续？(yes/no): ")
    
    if approval.lower() == "yes":
        return {
            "messages": [HumanMessage(content="人工审核通过")],
            "next_agent": "continue"
        }
    else:
        feedback = input("请提供修改意见: ")
        return {
            "messages": [HumanMessage(content=f"需要修改: {feedback}")],
            "next_agent": "revise"
        }

# 在图中添加人工审核节点
workflow.add_node("human_review", human_review_node)
workflow.add_edge("coder", "human_review")
```

### 4. 错误处理与重试

```python
def safe_agent_wrapper(agent_func, max_retries=3):
    """为 Agent 添加错误处理和重试机制"""
    
    def wrapped_agent(state: AgentState) -> AgentState:
        for attempt in range(max_retries):
            try:
                return agent_func(state)
            except Exception as e:
                print(f"⚠️ Agent 执行失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    # 最后一次尝试失败，返回错误状态
                    return {
                        "messages": [AIMessage(content=f"执行失败: {str(e)}")],
                        "results": {
                            **state.get("results", {}),
                            "error": str(e)
                        },
                        "next_agent": "FINISH"
                    }
                
                # 等待后重试
                import time
                time.sleep(2 ** attempt)
        
        return state
    
    return wrapped_agent

# 使用包装器
workflow.add_node("researcher", safe_agent_wrapper(researcher_node))
```

### 5. 动态 Agent 注册

```python
class AgentRegistry:
    """Agent 注册中心"""
    
    def __init__(self):
        self.agents = {}
    
    def register(self, name: str, agent_func, description: str):
        """注册新的 Agent"""
        self.agents[name] = {
            "function": agent_func,
            "description": description
        }
    
    def get_agent(self, name: str):
        """获取 Agent"""
        return self.agents.get(name, {}).get("function")
    
    def list_agents(self) -> str:
        """列出所有可用 Agent"""
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.agents.items()
        ])

# 使用示例
registry = AgentRegistry()
registry.register("researcher", researcher_node, "信息检索和数据分析")
registry.register("coder", coder_node, "代码编写和技术实现")
registry.register("writer", writer_node, "内容创作和文档编写")

# 在 Supervisor 中使用
def dynamic_supervisor(state: AgentState, registry: AgentRegistry):
    """支持动态 Agent 的 Supervisor"""
    
    available_agents = registry.list_agents()
    
    prompt = f"""
    可用的 Agent：
    {available_agents}
    
    任务：{state['task']}
    
    请选择合适的 Agent。
    """
    
    # ... 决策逻辑
```

### 6. 流式输出

```python
async def stream_multi_agent_system(task: str):
    """支持流式输出的多 Agent 系统"""
    
    app = create_multi_agent_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "results": {},
        "executed_agents": []
    }
    
    # 流式执行
    async for event in app.astream(initial_state):
        for node_name, node_state in event.items():
            print(f"\n📍 节点: {node_name}")
            
            if "messages" in node_state:
                latest_message = node_state["messages"][-1]
                print(f"💬 {latest_message.content}")
            
            # 实时显示进度
            if "executed_agents" in node_state:
                print(f"✅ 已完成: {', '.join(node_state['executed_agents'])}")

# 运行流式系统
import asyncio
asyncio.run(stream_multi_agent_system("创建一个 Web 应用"))
```

---

## 常见问题与最佳实践

### 常见问题

#### Q1: Supervisor 如何避免无限循环？

**解决方案**：
```python
class AgentState(TypedDict):
    # ... 其他字段
    iteration: int
    max_iterations: int

def supervisor_node(state: AgentState) -> AgentState:
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)
    
    if iteration >= max_iter:
        return {
            "messages": [AIMessage(content="达到最大迭代次数")],
            "next_agent": "FINISH"
        }
    
    # ... 正常决策逻辑
    
    return {
        "iteration": iteration + 1,
        # ... 其他返回值
    }
```

#### Q2: 如何处理 Agent 之间的依赖关系？

**解决方案**：
```python
def supervisor_node(state: AgentState) -> AgentState:
    executed = state.get("executed_agents", [])
    results = state.get("results", {})
    
    # 定义依赖关系
    dependencies = {
        "coder": ["researcher"],  # coder 依赖 researcher
        "writer": ["researcher", "coder"]  # writer 依赖两者
    }
    
    # 选择下一个 Agent
    for agent, deps in dependencies.items():
        if agent not in executed:
            # 检查依赖是否满足
            if all(dep in executed for dep in deps):
                return {"next_agent": agent}
    
    return {"next_agent": "FINISH"}
```

#### Q3: 如何优化 LLM 调用成本？

**解决方案**：
1. **缓存重复查询**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_llm_call(prompt: str) -> str:
    return llm.invoke(prompt).content
```

2. **使用不同模型**
```python
# Supervisor 使用强模型
supervisor_llm = ChatOpenAI(model="gpt-4")

# 简单 Agent 使用弱模型
agent_llm = ChatOpenAI(model="gpt-3.5-turbo")
```

3. **批量处理**
```python
def batch_process_agents(tasks: list) -> list:
    """批量处理多个任务"""
    prompts = [create_prompt(task) for task in tasks]
    responses = llm.batch(prompts)
    return responses
```

### 最佳实践

#### 1. 清晰的状态设计

```python
class AgentState(TypedDict):
    # 核心字段
    task: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # 控制流字段
    next_agent: str
    iteration: int
    
    # 数据字段
    results: dict
    intermediate_steps: list
    
    # 元数据字段
    executed_agents: list
    start_time: float
    metadata: dict
```

#### 2. 模块化 Agent 设计

```python
class BaseAgent:
    """Agent 基类"""
    
    def __init__(self, name: str, llm, tools: list = None):
        self.name = name
        self.llm = llm
        self.tools = tools or []
    
    def execute(self, state: AgentState) -> AgentState:
        """执行 Agent 任务"""
        raise NotImplementedError
    
    def create_prompt(self, state: AgentState) -> str:
        """创建提示词"""
        raise NotImplementedError

class ResearcherAgent(BaseAgent):
    """研究员 Agent"""
    
    def create_prompt(self, state: AgentState) -> str:
        return f"研究任务: {state['task']}"
    
    def execute(self, state: AgentState) -> AgentState:
        prompt = self.create_prompt(state)
        result = self.llm.invoke(prompt).content
        
        return {
            "results": {**state["results"], self.name: result},
            "executed_agents": state["executed_agents"] + [self.name]
        }
```

#### 3. 完善的日志记录

```python
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_agent(agent_func):
    """为 Agent 添加日志装饰器"""
    
    def wrapper(state: AgentState) -> AgentState:
        agent_name = agent_func.__name__
        
        logger.info(f"[{datetime.now()}] {agent_name} 开始执行")
        logger.debug(f"输入状态: {state}")
        
        try:
            result = agent_func(state)
            logger.info(f"[{datetime.now()}] {agent_name} 执行成功")
            logger.debug(f"输出状态: {result}")
            return result
        except Exception as e:
            logger.error(f"[{datetime.now()}] {agent_name} 执行失败: {e}")
            raise
    
    return wrapper

@logged_agent
def researcher_node(state: AgentState) -> AgentState:
    # ... 实现
    pass
```

#### 4. 测试策略

```python
import unittest

class TestMultiAgentSystem(unittest.TestCase):
    
    def setUp(self):
        """测试前准备"""
        self.app = create_multi_agent_graph()
    
    def test_simple_task(self):
        """测试简单任务"""
        state = {
            "task": "研究 Python 最佳实践",
            "messages": [],
            "results": {},
            "executed_agents": []
        }
        
        result = self.app.invoke(state)
        
        # 验证结果
        self.assertIn("researcher", result["executed_agents"])
        self.assertIn("researcher", result["results"])
    
    def test_complex_workflow(self):
        """测试复杂工作流"""
        state = {
            "task": "创建一个 Web 应用并编写文档",
            "messages": [],
            "results": {},
            "executed_agents": []
        }
        
        result = self.app.invoke(state)
        
        # 验证所有必要的 Agent 都执行了
        expected_agents = ["researcher", "coder", "writer"]
        for agent in expected_agents:
            self.assertIn(agent, result["executed_agents"])
    
    def test_error_handling(self):
        """测试错误处理"""
        # 模拟错误场景
        pass

if __name__ == "__main__":
    unittest.main()
```

#### 5. 性能监控

```python
import time
from functools import wraps

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
    
    def track(self, agent_name: str):
        """跟踪 Agent 性能的装饰器"""
        
        def decorator(func):
            @wraps(func)
            def wrapper(state: AgentState) -> AgentState:
                start_time = time.time()
                
                result = func(state)
                
                elapsed = time.time() - start_time
                
                if agent_name not in self.metrics:
                    self.metrics[agent_name] = []
                
                self.metrics[agent_name].append({
                    "duration": elapsed,
                    "timestamp": time.time()
                })
                
                return result
            
            return wrapper
        return decorator
    
    def get_report(self) -> str:
        """生成性能报告"""
        report = "性能报告\n" + "=" * 50 + "\n"
        
        for agent, metrics in self.metrics.items():
            durations = [m["duration"] for m in metrics]
            avg_duration = sum(durations) / len(durations)
            
            report += f"\n{agent}:\n"
            report += f"  调用次数: {len(metrics)}\n"
            report += f"  平均耗时: {avg_duration:.2f}s\n"
            report += f"  总耗时: {sum(durations):.2f}s\n"
        
        return report

# 使用示例
monitor = PerformanceMonitor()

@monitor.track("researcher")
def researcher_node(state: AgentState) -> AgentState:
    # ... 实现
    pass

# 执行后查看报告
print(monitor.get_report())
```

---

## 总结

本教程详细介绍了如何使用 LangGraph 构建多 Agent + Supervisor 系统：

### 核心要点

1. **状态管理**：使用 TypedDict 定义清晰的状态结构
2. **节点设计**：每个 Agent 是独立的节点，职责明确
3. **Supervisor 模式**：中央协调者负责任务分配和结果整合
4. **条件路由**：根据状态动态决定执行流程
5. **可扩展性**：易于添加新的 Agent 和功能

### 适用场景

- 复杂的研究与分析任务
- 多步骤的内容创作
- 需要多种专业技能的项目
- 企业级自动化工作流

### 进阶方向

- 集成外部工具（搜索、数据库、API）
- 实现更复杂的协作模式（层级、网状）
- 添加人机协作功能
- 优化性能和成本
- 部署到生产环境

---

## 参考资源

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangChain 文档](https://python.langchain.com/)
- [多 Agent 系统设计模式](https://arxiv.org/abs/2308.10848)
- [LangGraph GitHub 仓库](https://github.com/langchain-ai/langgraph)

---

**文档版本**: 1.0  
**最后更新**: 2026年2月  
**作者**: AI 助手  
**许可**: MIT License
