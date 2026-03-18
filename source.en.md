# LangGraph Multi-Agent + Supervisor System Implementation Tutorial

## Table of Contents

1. [Introduction](#introduction)
2. [LangGraph Core Concepts](#langgraph-core-concepts)
3. [Multi-Agent System Architecture Design](#multi-agent-system-architecture-design)
4. [Supervisor Pattern Deep Dive](#supervisor-pattern-deep-dive)
5. [Complete Implementation Example](#complete-implementation-example)
6. [Advanced Features and Optimization](#advanced-features-and-optimization)
7. [Common Questions and Best Practices](#common-questions-and-best-practices)

---

## Introduction

### What is LangGraph?

LangGraph is a framework for building stateful, multi-participant applications based on the LangChain ecosystem. It uses the concept of graphs to organize complex AI workflows, making it particularly suitable for building complex systems that require collaboration between multiple Agents.

### Why Use Multi-Agent + Supervisor Pattern?

In complex task scenarios, a single Agent often struggles to efficiently handle all problems. The Multi-Agent + Supervisor pattern offers the following advantages:

- **Specialized Division of Labor**: Each Agent focuses on a specific domain (research, coding, writing)
- **Parallel Processing**: Multiple Agents can handle different sub-tasks simultaneously
- **Flexible Expansion**: Easy to add new specialized Agents
- **Centralized Coordination**: Supervisor is responsible for task distribution and result integration

### Applicable Scenarios

- Complex research and analysis tasks
- Multi-step content creation workflows
- Projects requiring multiple specialized skills
- Enterprise-level automation workflows

---

## LangGraph Core Concepts

### 1. State

State is the data structure that flows through the graph, recording the context information of the entire workflow.

```python
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """Define the state structure for the Agent system"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str  # The next node to execute
    task: str  # Current task description
    results: dict  # Execution results from each Agent
```

**Key Points**:
- `Annotated` is used to define state update strategies
- `operator.add` indicates that message lists are accumulated rather than overwritten
- State is passed and updated between nodes

### 2. Node

Nodes are execution units in the graph. Each node is a function that receives state and returns updated state.

```python
def researcher_node(state: AgentState) -> AgentState:
    """Researcher Agent node"""
    # Execute research task
    result = perform_research(state["task"])
    
    return {
        "messages": [HumanMessage(content=f"Research completed: {result}")],
        "results": {**state.get("results", {}), "research": result}
    }
```

### 3. Edge

Edges define the connection relationships between nodes and are divided into two types:

**Normal Edge**: Fixed flow path
```python
graph.add_edge("node_a", "node_b")
```

**Conditional Edge**: Dynamically determines the next node based on state
```python
def route_function(state: AgentState) -> str:
    """Determine routing based on state"""
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

### 4. Graph

The graph is the container for the entire workflow, organizing all nodes and edges.

```python
from langgraph.graph import StateGraph, END

# Create graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)

# Set entry point
workflow.set_entry_point("supervisor")

# Compile graph
app = workflow.compile()
```

---

## Multi-Agent System Architecture Design

### System Architecture Diagram

```
┌─────────────────────────────────────────┐
│           User Input Task               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Supervisor                      │
│  - Analyze task                         │
│  - Select appropriate Agent             │
│  - Coordinate execution flow            │
└──────┬──────────┬──────────┬────────────┘
       │          │          │
       ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│Researcher│ │  Coder   │ │  Writer  │
│  Agent   │ │  Agent   │ │  Agent   │
│          │ │          │ │          │
│- Info    │ │- Code    │ │- Content │
│  Retrieval│ │  Generation│ │  Creation│
│- Data    │ │- Code    │ │- Document│
│  Analysis│ │  Review  │ │  Writing │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     └────────────┼────────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Result Integration & Output │
         └────────────────┘
```

### Agent Role Definitions

#### 1. Supervisor Agent

**Responsibilities**:
- Understand user tasks
- Decompose complex tasks into sub-tasks
- Select appropriate specialized Agents
- Integrate results from each Agent
- Determine if the task is complete

**Core Capabilities**:
- Task planning ability
- Decision-making ability
- Coordination ability

#### 2. Researcher Agent

**Responsibilities**:
- Information retrieval and collection
- Data analysis
- Background research
- Fact-checking

**Tools**:
- Search engine API
- Database queries
- Document parsing tools

#### 3. Coder Agent

**Responsibilities**:
- Code writing
- Code review
- Technical solution design
- Debugging and optimization

**Tools**:
- Code execution environment
- Code analysis tools
- Documentation generation tools

#### 4. Writer Agent

**Responsibilities**:
- Content creation
- Document writing
- Report generation
- Content polishing

**Tools**:
- Template engine
- Format conversion tools
- Grammar checking tools

---

## Supervisor Pattern Deep Dive

### Supervisor Workflow

```
1. Receive task
   ↓
2. Analyze task requirements
   ↓
3. Select appropriate Agent
   ↓
4. Assign sub-task
   ↓
5. Monitor execution progress
   ↓
6. Collect Agent results
   ↓
7. Determine if continuation is needed
   ├─ Yes → Return to step 3
   └─ No → Integrate results and output
```

### Supervisor Implementation Key Points

#### 1. Task Analysis and Planning

```python
def analyze_task(task: str) -> dict:
    """Analyze task and generate execution plan"""
    prompt = f"""
    Analyze the following task and determine which specialized Agents are needed:
    
    Task: {task}
    
    Available Agents:
    - researcher: Information retrieval, data analysis
    - coder: Code writing, technical implementation
    - writer: Content creation, document writing
    
    Please return the execution plan in JSON format:
    {{
        "agents_needed": ["agent1", "agent2"],
        "execution_order": ["step1", "step2"],
        "expected_outcome": "Expected result description"
    }}
    """
    
    # Call LLM for analysis
    response = llm.invoke(prompt)
    return parse_json(response)
```

#### 2. Agent Selection Logic

```python
def select_next_agent(state: AgentState) -> str:
    """Select the next Agent based on current state"""
    
    # Check if task is complete
    if is_task_complete(state):
        return "FINISH"
    
    # Get executed Agents
    executed = state.get("executed_agents", [])
    
    # Select next Agent based on task plan
    plan = state.get("plan", {})
    remaining_agents = [
        agent for agent in plan["agents_needed"] 
        if agent not in executed
    ]
    
    if remaining_agents:
        return remaining_agents[0]
    
    # If iteration refinement is needed
    if needs_refinement(state):
        return determine_refinement_agent(state)
    
    return "FINISH"
```

#### 3. Result Integration

```python
def integrate_results(state: AgentState) -> str:
    """Integrate execution results from each Agent"""
    results = state.get("results", {})
    
    prompt = f"""
    Please integrate the work results from each specialized Agent and generate the final output:
    
    Research results: {results.get('researcher', 'N/A')}
    Code implementation: {results.get('coder', 'N/A')}
    Document content: {results.get('writer', 'N/A')}
    
    Original task: {state['task']}
    
    Please generate a complete and coherent final result.
    """
    
    final_output = llm.invoke(prompt)
    return final_output
```

---

## Complete Implementation Example

### Environment Setup

```bash
# Install dependencies
pip install langgraph langchain langchain-openai langchain-community
```

### Complete Code Implementation

```python
import operator
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json

# ============================================
# 1. Define State Structure
# ============================================

class AgentState(TypedDict):
    """State for multi-Agent system"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str  # Original task
    next_agent: str  # Next Agent to execute
    results: dict  # Results from each Agent
    executed_agents: list  # List of executed Agents
    iteration: int  # Iteration count
    final_output: str  # Final output

# ============================================
# 2. Initialize LLM
# ============================================

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ============================================
# 3. Define Supervisor Agent
# ============================================

def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor is responsible for task analysis and Agent scheduling"""
    
    messages = state["messages"]
    task = state["task"]
    results = state.get("results", {})
    executed = state.get("executed_agents", [])
    
    # Build Supervisor prompt
    system_prompt = """
    You are a task coordinator (Supervisor) responsible for managing multiple specialized Agents.
    
    Available Agents:
    - researcher: Specializes in information retrieval, data analysis, background research
    - coder: Specializes in code writing, technical implementation, algorithm design
    - writer: Specializes in content creation, document writing, report generation
    
    Your responsibilities:
    1. Analyze task requirements
    2. Select appropriate Agent to execute sub-tasks
    3. Determine if the task is complete
    
    Please return the decision in JSON format based on the current situation:
    {
        "next_agent": "researcher/coder/writer/FINISH",
        "reason": "Reason for selection",
        "subtask": "Specific sub-task assigned to this Agent"
    }
    """
    
    context = f"""
    Original task: {task}
    
    Executed Agents: {executed}
    
    Current results:
    {json.dumps(results, ensure_ascii=False, indent=2)}
    
    Please decide the next action.
    """
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context}
    ])
    
    # Parse decision
    try:
        decision = json.loads(response.content)
    except:
        # Default to finish if parsing fails
        decision = {"next_agent": "FINISH", "reason": "Parse error", "subtask": ""}
    
    return {
        "messages": [AIMessage(content=f"Supervisor decision: {decision['reason']}")],
        "next_agent": decision["next_agent"],
        "results": {**results, "supervisor_decision": decision}
    }

# ============================================
# 4. Define Specialized Agent Nodes
# ============================================

def researcher_node(state: AgentState) -> AgentState:
    """Researcher Agent - responsible for information retrieval and analysis"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    
    prompt = f"""
    You are a professional researcher, specializing in information retrieval and data analysis.
    
    Task: {subtask}
    
    Please conduct in-depth research and provide detailed analysis results.
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Researcher completed: {result[:100]}...")],
        "results": {**state["results"], "researcher": result},
        "executed_agents": executed + ["researcher"]
    }

def coder_node(state: AgentState) -> AgentState:
    """Coder Agent - responsible for code writing"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    research_result = state["results"].get("researcher", "")
    
    prompt = f"""
    You are a professional programmer, specializing in code writing and technical implementation.
    
    Task: {subtask}
    
    Reference information: {research_result}
    
    Please provide complete code implementation with comments and explanations.
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Coder completed: Code generated")],
        "results": {**state["results"], "coder": result},
        "executed_agents": executed + ["coder"]
    }

def writer_node(state: AgentState) -> AgentState:
    """Writer Agent - responsible for content creation"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    research_result = state["results"].get("researcher", "")
    code_result = state["results"].get("coder", "")
    
    prompt = f"""
    You are a professional writer, specializing in content creation and document writing.
    
    Task: {subtask}
    
    Research materials: {research_result}
    Code implementation: {code_result}
    
    Please create high-quality content or documents.
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Writer completed: Content created")],
        "results": {**state["results"], "writer": result},
        "executed_agents": executed + ["writer"]
    }

# ============================================
# 5. Define Routing Logic
# ============================================

def route_agent(state: AgentState) -> Literal["researcher", "coder", "writer", "end"]:
    """Route to the corresponding Agent based on Supervisor's decision"""
    
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
# 6. Build Workflow Graph
# ============================================

def create_multi_agent_graph():
    """Create multi-Agent + Supervisor workflow graph"""
    
    # Create state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("writer", writer_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges: route from supervisor to different Agents based on decision
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
    
    # Add normal edges: return to supervisor after each Agent completes
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("coder", "supervisor")
    workflow.add_edge("writer", "supervisor")
    
    # Compile graph
    app = workflow.compile()
    
    return app

# ============================================
# 7. Run Example
# ============================================

def run_multi_agent_system(task: str):
    """Run multi-Agent system"""
    
    # Create workflow
    app = create_multi_agent_graph()
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "next_agent": "",
        "results": {},
        "executed_agents": [],
        "iteration": 0,
        "final_output": ""
    }
    
    # Execute workflow
    print(f"🚀 Starting task execution: {task}\n")
    print("=" * 60)
    
    final_state = app.invoke(initial_state)
    
    # Output results
    print("\n" + "=" * 60)
    print("📊 Execution Results:\n")
    
    for agent, result in final_state["results"].items():
        if agent != "supervisor_decision":
            print(f"\n【{agent.upper()}】")
            print("-" * 60)
            print(result[:500] + "..." if len(result) > 500 else result)
    
    return final_state

# ============================================
# 8. Test Example
# ============================================

if __name__ == "__main__":
    # Example task
    task = """
    Create a Python program that implements a simple todo list management system.
    Requirements:
    1. Support adding, deleting, and viewing todo items
    2. Use SQLite database for storage
    3. Provide command-line interface
    4. Write complete user documentation
    """
    
    result = run_multi_agent_system(task)
```

### Execution Flow Explanation

1. **Task Input**: User submits a complex task
2. **Supervisor Analysis**: Analyze the task and determine which Agents are needed
3. **Researcher Execution**: Collect relevant technical materials and best practices
4. **Return to Supervisor**: Submit research results
5. **Supervisor Re-decision**: Based on research results, assign coding task
6. **Coder Execution**: Write complete code implementation
7. **Return to Supervisor**: Submit code
8. **Supervisor Re-decision**: Assign document writing task
9. **Writer Execution**: Write user documentation
10. **Return to Supervisor**: Submit documentation
11. **Supervisor Determines Completion**: Integrate all results and output final deliverable

---

## Advanced Features and Optimization

### 1. Adding Memory Function

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Create checkpoint saver
memory = SqliteSaver.from_conn_string(":memory:")

# Add checkpoint when compiling
app = workflow.compile(checkpointer=memory)

# Specify thread ID at runtime
config = {"configurable": {"thread_id": "conversation_1"}}
result = app.invoke(initial_state, config=config)
```

### 2. Parallel Agent Execution

```python
from langgraph.graph import START

def create_parallel_graph():
    """Create graph that supports parallel execution"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("aggregator", aggregator_node)
    
    # Distribute from supervisor to multiple Agents in parallel
    workflow.add_edge("supervisor", "researcher")
    workflow.add_edge("supervisor", "coder")
    
    # After multiple Agents complete, converge to aggregator
    workflow.add_edge("researcher", "aggregator")
    workflow.add_edge("coder", "aggregator")
    
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()
```

### 3. Human Review Node

```python
from langgraph.checkpoint.sqlite import SqliteSaver

def human_review_node(state: AgentState) -> AgentState:
    """Human review node"""
    
    # Pause execution and wait for human input
    print("\n🔍 Human review required:")
    print(f"Current results: {state['results']}")
    
    approval = input("\nApprove to continue? (yes/no): ")
    
    if approval.lower() == "yes":
        return {
            "messages": [HumanMessage(content="Human review approved")],
            "next_agent": "continue"
        }
    else:
        feedback = input("Please provide modification feedback: ")
        return {
            "messages": [HumanMessage(content=f"Needs modification: {feedback}")],
            "next_agent": "revise"
        }

# Add human review node to graph
workflow.add_node("human_review", human_review_node)
workflow.add_edge("coder", "human_review")
```

### 4. Error Handling and Retry

```python
def safe_agent_wrapper(agent_func, max_retries=3):
    """Add error handling and retry mechanism for Agent"""
    
    def wrapped_agent(state: AgentState) -> AgentState:
        for attempt in range(max_retries):
            try:
                return agent_func(state)
            except Exception as e:
                print(f"⚠️ Agent execution failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    # Last attempt failed, return error state
                    return {
                        "messages": [AIMessage(content=f"Execution failed: {str(e)}")],
                        "results": {
                            **state.get("results", {}),
                            "error": str(e)
                        },
                        "next_agent": "FINISH"
                    }
                
                # Wait and retry
                import time
                time.sleep(2 ** attempt)
        
        return state
    
    return wrapped_agent

# Use wrapper
workflow.add_node("researcher", safe_agent_wrapper(researcher_node))
```

### 5. Dynamic Agent Registration

```python
class AgentRegistry:
    """Agent registry center"""
    
    def __init__(self):
        self.agents = {}
    
    def register(self, name: str, agent_func, description: str):
        """Register new Agent"""
        self.agents[name] = {
            "function": agent_func,
            "description": description
        }
    
    def get_agent(self, name: str):
        """Get Agent"""
        return self.agents.get(name, {}).get("function")
    
    def list_agents(self) -> str:
        """List all available Agents"""
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.agents.items()
        ])

# Usage example
registry = AgentRegistry()
registry.register("researcher", researcher_node, "Information retrieval and data analysis")
registry.register("coder", coder_node, "Code writing and technical implementation")
registry.register("writer", writer_node, "Content creation and document writing")

# Use in Supervisor
def dynamic_supervisor(state: AgentState, registry: AgentRegistry):
    """Supervisor that supports dynamic Agents"""
    
    available_agents = registry.list_agents()
    
    prompt = f"""
    Available Agents:
    {available_agents}
    
    Task: {state['task']}
    
    Please select the appropriate Agent.
    """
    
    # ... Decision logic
```

### 6. Streaming Output

```python
async def stream_multi_agent_system(task: str):
    """Multi-Agent system with streaming output support"""
    
    app = create_multi_agent_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "results": {},
        "executed_agents": []
    }
    
    # Stream execution
    async for event in app.astream(initial_state):
        for node_name, node_state in event.items():
            print(f"\n📍 Node: {node_name}")
            
            if "messages" in node_state:
                latest_message = node_state["messages"][-1]
                print(f"💬 {latest_message.content}")
            
            # Display progress in real-time
            if "executed_agents" in node_state:
                print(f"✅ Completed: {', '.join(node_state['executed_agents'])}")

# Run streaming system
import asyncio
asyncio.run(stream_multi_agent_system("Create a Web application"))
```

---

## Common Questions and Best Practices

### Common Questions

#### Q1: How to prevent infinite loops in Supervisor?

**Solution**:
```python
class AgentState(TypedDict):
    # ... other fields
    iteration: int
    max_iterations: int

def supervisor_node(state: AgentState) -> AgentState:
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)
    
    if iteration >= max_iter:
        return {
            "messages": [AIMessage(content="Maximum iterations reached")],
            "next_agent": "FINISH"
        }
    
    # ... Normal decision logic
    
    return {
        "iteration": iteration + 1,
        # ... other return values
    }
```

#### Q2: How to handle dependencies between Agents?

**Solution**:
```python
def supervisor_node(state: AgentState) -> AgentState:
    executed = state.get("executed_agents", [])
    results = state.get("results", {})
    
    # Define dependencies
    dependencies = {
        "coder": ["researcher"],  # coder depends on researcher
        "writer": ["researcher", "coder"]  # writer depends on both
    }
    
    # Select next Agent
    for agent, deps in dependencies.items():
        if agent not in executed:
            # Check if dependencies are satisfied
            if all(dep in executed for dep in deps):
                return {"next_agent": agent}
    
    return {"next_agent": "FINISH"}
```

#### Q3: How to optimize LLM call costs?

**Solution**:
1. **Cache repeated queries**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_llm_call(prompt: str) -> str:
    return llm.invoke(prompt).content
```

2. **Use different models**
```python
# Supervisor uses strong model
supervisor_llm = ChatOpenAI(model="gpt-4")

# Simple Agents use weaker model
agent_llm = ChatOpenAI(model="gpt-3.5-turbo")
```

3. **Batch processing**
```python
def batch_process_agents(tasks: list) -> list:
    """Batch process multiple tasks"""
    prompts = [create_prompt(task) for task in tasks]
    responses = llm.batch(prompts)
    return responses
```

### Best Practices

#### 1. Clear State Design

```python
class AgentState(TypedDict):
    # Core fields
    task: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Control flow fields
    next_agent: str
    iteration: int
    
    # Data fields
    results: dict
    intermediate_steps: list
    
    # Metadata fields
    executed_agents: list
    start_time: float
    metadata: dict
```

#### 2. Modular Agent Design

```python
class BaseAgent:
    """Base Agent class"""
    
    def __init__(self, name: str, llm, tools: list = None):
        self.name = name
        self.llm = llm
        self.tools = tools or []
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute Agent task"""
        raise NotImplementedError
    
    def create_prompt(self, state: AgentState) -> str:
        """Create prompt"""
        raise NotImplementedError

class ResearcherAgent(BaseAgent):
    """Researcher Agent"""
    
    def create_prompt(self, state: AgentState) -> str:
        return f"Research task: {state['task']}"
    
    def execute(self, state: AgentState) -> AgentState:
        prompt = self.create_prompt(state)
        result = self.llm.invoke(prompt).content
        
        return {
            "results": {**state["results"], self.name: result},
            "executed_agents": state["executed_agents"] + [self.name]
        }
```

#### 3. Comprehensive Logging

```python
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_agent(agent_func):
    """Add logging decorator to Agent"""
    
    def wrapper(state: AgentState) -> AgentState:
        agent_name = agent_func.__name__
        
        logger.info(f"[{datetime.now()}] {agent_name} started execution")
        logger.debug(f"Input state: {state}")
        
        try:
            result = agent_func(state)
            logger.info(f"[{datetime.now()}] {agent_name} executed successfully")
            logger.debug(f"Output state: {result}")
            return result
        except Exception as e:
            logger.error(f"[{datetime.now()}] {agent_name} execution failed: {e}")
            raise
    
    return wrapper

@logged_agent
def researcher_node(state: AgentState) -> AgentState:
    # ... implementation
    pass
```

#### 4. Testing Strategy

```python
import unittest

class TestMultiAgentSystem(unittest.TestCase):
    
    def setUp(self):
        """Pre-test setup"""
        self.app = create_multi_agent_graph()
    
    def test_simple_task(self):
        """Test simple task"""
        state = {
            "task": "Research Python best practices",
            "messages": [],
            "results": {},
            "executed_agents": []
        }
        
        result = self.app.invoke(state)
        
        # Verify results
        self.assertIn("researcher", result["executed_agents"])
        self.assertIn("researcher", result["results"])
    
    def test_complex_workflow(self):
        """Test complex workflow"""
        state = {
            "task": "Create a Web application and write documentation",
            "messages": [],
            "results": {},
            "executed_agents": []
        }
        
        result = self.app.invoke(state)
        
        # Verify all necessary Agents executed
        expected_agents = ["researcher", "coder", "writer"]
        for agent in expected_agents:
            self.assertIn(agent, result["executed_agents"])
    
    def test_error_handling(self):
        """Test error handling"""
        # Simulate error scenario
        pass

if __name__ == "__main__":
    unittest.main()
```

#### 5. Performance Monitoring

```python
import time
from functools import wraps

class PerformanceMonitor:
    """Performance monitor"""
    
    def __init__(self):
        self.metrics = {}
    
    def track(self, agent_name: str):
        """Decorator to track Agent performance"""
        
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
        """Generate performance report"""
        report = "Performance Report\n" + "=" * 50 + "\n"
        
        for agent, metrics in self.metrics.items():
            durations = [m["duration"] for m in metrics]
            avg_duration = sum(durations) / len(durations)
            
            report += f"\n{agent}:\n"
            report += f"  Call count: {len(metrics)}\n"
            report += f"  Average duration: {avg_duration:.2f}s\n"
            report += f"  Total duration: {sum(durations):.2f}s\n"
        
        return report

# Usage example
monitor = PerformanceMonitor()

@monitor.track("researcher")
def researcher_node(state: AgentState) -> AgentState:
    # ... implementation
    pass

# View report after execution
print(monitor.get_report())
```

---

## Summary

This tutorial provides a detailed introduction to building a Multi-Agent + Supervisor system using LangGraph:

### Core Points

1. **State Management**: Use TypedDict to define clear state structures
2. **Node Design**: Each Agent is an independent node with clear responsibilities
3. **Supervisor Pattern**: Central coordinator responsible for task distribution and result integration
4. **Conditional Routing**: Dynamically determine execution flow based on state
5. **Scalability**: Easy to add new Agents and features

### Applicable Scenarios

- Complex research and analysis tasks
- Multi-step content creation
- Projects requiring multiple specialized skills
- Enterprise-level automation workflows

### Advanced Directions

- Integrate external tools (search, database, API)
- Implement more complex collaboration patterns (hierarchical, mesh)
- Add human-AI collaboration features
- Optimize performance and costs
- Deploy to production environment

---

## Reference Resources

- [LangGraph Official Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Multi-Agent System Design Patterns](https://arxiv.org/abs/2308.10848)
- [LangGraph GitHub Repository](https://github.com/langchain-ai/langgraph)

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Author**: AI Assistant  
**License**: MIT License