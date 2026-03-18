# LangGraph 다중 에이전트 + Supervisor 시스템 구현 튜토리얼

## 목차

1. [서론](#서론)
2. [LangGraph 핵심 개념](#langgraph-핵심-개념)
3. [다중 에이전트 시스템 아키텍처 설계](#다중-에이전트-시스템-아키텍처-설계)
4. [Supervisor 패턴 상세 설명](#supervisor-패턴-상세-설명)
5. [전체 구현 예제](#전체-구현-예제)
6. [고급 기능과 최적화](#고급-기능과-최적화)
7. [자주 묻는 질문과 모범 사례](#자주-묻는-질문과-모범-사례)

---

## 서론

### LangGraph란?

LangGraph는 LangChain 생태계를 기반으로 상태와 다중 참여자 애플리케이션을 구축하기 위한 프레임워크입니다. 그래프(Graph) 개념을 사용하여 복잡한 AI 워크플로를 구성하며, 여러 에이전트가 협업해야 하는 복잡한 시스템 구축에 특히 적합합니다.

### 다중 에이전트 + Supervisor 패턴을 사용하는 이유?

복잡한 작업 시나리오에서 단일 에이전트는 모든 문제를 효율적으로 처리하기 어려운 경우가 많습니다. 다중 에이전트 + Supervisor 패턴은 다음과 같은 장점을 제공합니다:

- **전문화 분업**: 각 에이전트가 특정 분야(연구, 코딩, 글쓰기)에 집중
- **병렬 처리**: 여러 에이전트가 동시에 다른 하위 작업 처리
- **유연한 확장**: 새로운 전문 에이전트 추가 용이
- **중앙 조정**: Supervisor가 작업 할당 및 결과 통합 담당

### 적용 시나리오

- 복잡한 연구 및 분석 작업
- 다단계 콘텐츠创作 프로세스
- 다양한 전문 기술이 필요한 프로젝트
- 기업급 자동화 워크플로

---

## LangGraph 핵심 개념

### 1. 상태(State)

상태는 그래프에서流转되는 데이터 구조로, 전체 워크플로의 컨텍스트 정보를 기록합니다.

```python
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """에이전트 시스템의 상태 구조 정의"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str  # 다음에 실행할 노드
    task: str  # 현재 작업 설명
    results: dict  # 각 에이전트의 실행 결과
```

**핵심 포인트**:
- `Annotated`는 상태 업데이트 전략을 정의하는 데 사용
- `operator.add`는 메시지 목록이 덮어쓰기가 아닌 누적됨을 의미
- 상태는 노드 간에 전달되고 업데이트됨

### 2. 노드(Node)

노드는 그래프의 실행 단위로, 상태를 수신하고 업데이트된 상태를 반환하는 함수입니다.

```python
def researcher_node(state: AgentState) -> AgentState:
    """연구원 에이전트 노드"""
    # 연구 작업 실행
    result = perform_research(state["task"])
    
    return {
        "messages": [HumanMessage(content=f"연구 완료: {result}")],
        "results": {**state.get("results", {}), "research": result}
    }
```

### 3. 엣지(Edge)

엣지는 노드 간의 연결 관계를 정의하며, 두 가지 유형이 있습니다:

**일반 엣지(Normal Edge)**: 고정된流转 경로
```python
graph.add_edge("node_a", "node_b")
```

**조건부 엣지(Conditional Edge)**: 상태에 따라 동적으로 다음 노드 결정
```python
def route_function(state: AgentState) -> str:
    """상태에 따라 라우팅 결정"""
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

### 4. 그래프(Graph)

그래프는 전체 워크플로의 컨테이너로, 모든 노드와 엣지를 구성합니다.

```python
from langgraph.graph import StateGraph, END

# 그래프 생성
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)

# 진입점 설정
workflow.set_entry_point("supervisor")

# 그래프 컴파일
app = workflow.compile()
```

---

## 다중 에이전트 시스템 아키텍처 설계

### 시스템 아키텍처 다이어그램

```
┌─────────────────────────────────────────┐
│           사용자 입력 작업                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Supervisor (감독자)              │
│  - 작업 분석                              │
│  - 적절한 에이전트 선택                    │
│  - 실행 프로세스 조정                      │
└──────┬──────────┬──────────┬────────────┘
       │          │          │
       ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│Researcher│ │  Coder   │ │  Writer  │
│  Agent   │ │  Agent   │ │  Agent   │
│          │ │          │ │          │
│- 정보 검색│ │- 코드 생성│ │- 콘텐츠创作│
│- 데이터 분석│ │- 코드 리뷰│ │- 문서 작성│
└──────┬───┘ └────┬─────┘ └────┬─────┘
       │          │          │
       └──────────┼──────────┘
                  │
                  ▼
         ┌────────────────┐
         │  결과 통합 및 출력 │
         └────────────────┘
```

### 에이전트 역할 정의

#### 1. Supervisor Agent (감독자)

**职责**:
- 사용자 작업 이해
- 복잡한 작업을 하위 작업으로 분해
- 적절한 전문 에이전트 선택
- 각 에이전트의 결과 통합
- 작업 완료 여부 판단

**핵심 능력**:
- 작업 계획 능력
- 의사결정 능력
- 조정 능력

#### 2. Researcher Agent (연구원)

**职责**:
- 정보 검색 및 수집
- 데이터 분석
- 배경 조사
- 사실 확인

**도구**:
- 검색 엔진 API
- 데이터베이스 쿼리
- 문서 파싱 도구

#### 3. Coder Agent (프로그래머)

**职责**:
- 코드 작성
- 코드 리뷰
- 기술方案 설계
- 디버깅 및 최적화

**도구**:
- 코드 실행 환경
- 코드 분석 도구
- 문서 생성 도구

#### 4. Writer Agent (작가)

**职责**:
- 콘텐츠创作
- 문서 작성
- 보고서 생성
- 콘텐츠润色

**도구**:
- 템플릿 엔진
- 형식 변환 도구
- 문법 검사 도구

---

## Supervisor 패턴 상세 설명

### Supervisor의 작업 프로세스

```
1. 작업 수신
   ↓
2. 작업 요구사항 분석
   ↓
3. 적절한 에이전트 선택
   ↓
4. 하위 작업 할당
   ↓
5. 실행 진행 상황 모니터링
   ↓
6. 에이전트 결과 수집
   ↓
7. 계속 필요한지 판단
   ├─ 예 → 단계 3으로 이동
   └─ 아니오 → 결과 통합 및 출력
```

### Supervisor 구현 핵심 포인트

#### 1. 작업 분석 및 계획

```python
def analyze_task(task: str) -> dict:
    """작업을 분석하고 실행 계획 생성"""
    prompt = f"""
    다음 작업을 분석하여 어떤 전문 에이전트가 참여해야 하는지 결정하세요:
    
    작업: {task}
    
    사용 가능한 에이전트:
    - researcher: 정보 검색, 데이터 분석
    - coder: 코드 작성, 기술 구현
    - writer: 콘텐츠创作, 문서 작성
    
    JSON 형식으로 실행 계획을 반환하세요:
    {{
        "agents_needed": ["agent1", "agent2"],
        "execution_order": ["step1", "step2"],
        "expected_outcome": "예상 결과 설명"
    }}
    """
    
    # 분석을 위해 LLM 호출
    response = llm.invoke(prompt)
    return parse_json(response)
```

#### 2. 에이전트 선택 로직

```python
def select_next_agent(state: AgentState) -> str:
    """현재 상태에 따라 다음 에이전트 선택"""
    
    # 작업 완료 여부 확인
    if is_task_complete(state):
        return "FINISH"
    
    # 실행된 에이전트 가져오기
    executed = state.get("executed_agents", [])
    
    # 작업 계획에 따라 다음 에이전트 선택
    plan = state.get("plan", {})
    remaining_agents = [
        agent for agent in plan["agents_needed"] 
        if agent not in executed
    ]
    
    if remaining_agents:
        return remaining_agents[0]
    
    # 반복 최적화가 필요한 경우
    if needs_refinement(state):
        return determine_refinement_agent(state)
    
    return "FINISH"
```

#### 3. 결과 통합

```python
def integrate_results(state: AgentState) -> str:
    """각 에이전트의 실행 결과 통합"""
    results = state.get("results", {})
    
    prompt = f"""
    다음 각 전문 에이전트의 작업 결과를 통합하여 최종 출력을 생성하세요:
    
    연구 결과: {results.get('researcher', 'N/A')}
    코드 구현: {results.get('coder', 'N/A')}
    문서 내용: {results.get('writer', 'N/A')}
    
    원래 작업: {state['task']}
    
    완전하고 일관된 최종 결과를 생성하세요.
    """
    
    final_output = llm.invoke(prompt)
    return final_output
```

---

## 전체 구현 예제

### 환경 준비

```bash
# 의존성 설치
pip install langgraph langchain langchain-openai langchain-community
```

### 전체 코드 구현

```python
import operator
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json

# ============================================
# 1. 상태 구조 정의
# ============================================

class AgentState(TypedDict):
    """다중 에이전트 시스템의 상태"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str  # 원래 작업
    next_agent: str  # 다음에 실행할 에이전트
    results: dict  # 각 에이전트의 결과
    executed_agents: list  # 실행된 에이전트 목록
    iteration: int  # 반복 횟수
    final_output: str  # 최종 출력

# ============================================
# 2. LLM 초기화
# ============================================

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ============================================
# 3. Supervisor Agent 정의
# ============================================

def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor는 작업 분석 및 에이전트 스케줄링 담당"""
    
    messages = state["messages"]
    task = state["task"]
    results = state.get("results", {})
    executed = state.get("executed_agents", [])
    
    # Supervisor의 프롬프트 구성
    system_prompt = """
    당신은 여러 전문 에이전트를 관리하는 작업 조정자(Supervisor)입니다.
    
    사용 가능한 에이전트:
    - researcher: 정보 검색, 데이터 분석, 배경 조사에 능함
    - coder: 코드 작성, 기술 구현, 알고리즘 설계에 능함
    - writer: 콘텐츠创作, 문서 작성, 보고서 생성에 능함
    
    당신의职责:
    1. 작업 요구사항 분석
    2. 하위 작업을 실행할 적절한 에이전트 선택
    3. 작업 완료 여부 판단
    
    현재 상황에 따라 JSON 형식으로 의사결정을 반환하세요:
    {
        "next_agent": "researcher/coder/writer/FINISH",
        "reason": "선택 이유",
        "subtask": "해당 에이전트에 할당한 구체적인 하위 작업"
    }
    """
    
    context = f"""
    원래 작업: {task}
    
    실행된 에이전트: {executed}
    
    현재 결과:
    {json.dumps(results, ensure_ascii=False, indent=2)}
    
    다음 행동을 결정하세요.
    """
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context}
    ])
    
    # 의사결정 파싱
    try:
        decision = json.loads(response.content)
    except:
        # 파싱 실패 시 기본 종료
        decision = {"next_agent": "FINISH", "reason": "파싱 오류", "subtask": ""}
    
    return {
        "messages": [AIMessage(content=f"Supervisor 의사결정: {decision['reason']}")],
        "next_agent": decision["next_agent"],
        "results": {**results, "supervisor_decision": decision}
    }

# ============================================
# 4. 전문 에이전트 노드 정의
# ============================================

def researcher_node(state: AgentState) -> AgentState:
    """연구원 에이전트 - 정보 검색 및 분석 담당"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    
    prompt = f"""
    당신은 정보 검색 및 데이터 분석에 전문적인 연구원입니다.
    
    작업: {subtask}
    
    심층 연구를 진행하여 상세한 분석 결과를 제공하세요.
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Researcher 완료: {result[:100]}...")],
        "results": {**state["results"], "researcher": result},
        "executed_agents": executed + ["researcher"]
    }

def coder_node(state: AgentState) -> AgentState:
    """프로그래머 에이전트 - 코드 작성 담당"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    research_result = state["results"].get("researcher", "")
    
    prompt = f"""
    당신은 코드 작성 및 기술 구현에 전문적인 프로그래머입니다.
    
    작업: {subtask}
    
    참고 정보: {research_result}
    
    주석과 설명이 포함된 완전한 코드 구현을 제공하세요.
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Coder 완료: 코드 생성됨")],
        "results": {**state["results"], "coder": result},
        "executed_agents": executed + ["coder"]
    }

def writer_node(state: AgentState) -> AgentState:
    """작가 에이전트 - 콘텐츠创作 담당"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    research_result = state["results"].get("researcher", "")
    code_result = state["results"].get("coder", "")
    
    prompt = f"""
    당신은 콘텐츠创作 및 문서 작성에 전문적인 작가입니다.
    
    작업: {subtask}
    
    연구 자료: {research_result}
    코드 구현: {code_result}
    
    고품질의 콘텐츠나 문서를创作하세요.
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Writer 완료: 콘텐츠 생성됨")],
        "results": {**state["results"], "writer": result},
        "executed_agents": executed + ["writer"]
    }

# ============================================
# 5. 라우팅 로직 정의
# ============================================

def route_agent(state: AgentState) -> Literal["researcher", "coder", "writer", "end"]:
    """Supervisor의 의사결정에 따라 해당 에이전트로 라우팅"""
    
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
# 6. 워크플로 그래프 구축
# ============================================

def create_multi_agent_graph():
    """다중 에이전트 + Supervisor 워크플로 그래프 생성"""
    
    # 상태 그래프 생성
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("writer", writer_node)
    
    # 진입점 설정
    workflow.set_entry_point("supervisor")
    
    # 조건부 엣지 추가: supervisor에서 의사결정에 따라 다른 에이전트로 라우팅
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
    
    # 일반 엣지 추가: 각 에이전트 완료 후 supervisor로 복귀
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("coder", "supervisor")
    workflow.add_edge("writer", "supervisor")
    
    # 그래프 컴파일
    app = workflow.compile()
    
    return app

# ============================================
# 7. 실행 예제
# ============================================

def run_multi_agent_system(task: str):
    """다중 에이전트 시스템 실행"""
    
    # 워크플로 생성
    app = create_multi_agent_graph()
    
    # 상태 초기화
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "next_agent": "",
        "results": {},
        "executed_agents": [],
        "iteration": 0,
        "final_output": ""
    }
    
    # 워크플로 실행
    print(f"🚀 작업 실행 시작: {task}\n")
    print("=" * 60)
    
    final_state = app.invoke(initial_state)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 실행 결과:\n")
    
    for agent, result in final_state["results"].items():
        if agent != "supervisor_decision":
            print(f"\n【{agent.upper()}】")
            print("-" * 60)
            print(result[:500] + "..." if len(result) > 500 else result)
    
    return final_state

# ============================================
# 8. 테스트 예제
# ============================================

if __name__ == "__main__":
    # 예제 작업
    task = """
    Python으로 간단한 할 일 관리 시스템을 만드는 프로그램을 작성하세요.
    요구사항:
    1. 할 일 추가, 삭제, 조회 지원
    2. SQLite 데이터베이스 사용
    3.命令行 인터페이스 제공
    4. 완전한 사용 문서 작성
    """
    
    result = run_multi_agent_system(task)
```

### 실행 프로세스 설명

1. **작업 입력**: 사용자가 복잡한 작업 제출
2. **Supervisor 분석**: 작업 분석, 필요한 에이전트 결정
3. **Researcher 실행**: 관련 기술 자료 및 모범 사례 수집
4. **Supervisor로 복귀**: 연구 결과 제출
5. **Supervisor 재의사결정**: 연구 결과를 기반으로 코딩 작업 할당
6. **Coder 실행**: 완전한 코드 구현 작성
7. **Supervisor로 복귀**: 코드 제출
8. **Supervisor 재의사결정**: 문서 작성 작업 할당
9. **Writer 실행**: 사용 문서 작성
10. **Supervisor로 복귀**: 문서 제출
11. **Supervisor 완료 판단**: 모든 결과 통합, 최종 산출물 출력

---

## 고급 기능과 최적화

### 1. 기억 기능 추가

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 체크포인트 저장소 생성
memory = SqliteSaver.from_conn_string(":memory:")

# 컴파일 시 체크포인트 추가
app = workflow.compile(checkpointer=memory)

# 실행 시 스레드 ID 지정
config = {"configurable": {"thread_id": "conversation_1"}}
result = app.invoke(initial_state, config=config)
```

### 2. 에이전트 병렬 실행

```python
from langgraph.graph import START

def create_parallel_graph():
    """병렬 실행을 지원하는 그래프 생성"""
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("aggregator", aggregator_node)
    
    # supervisor에서 여러 에이전트로 병렬分发
    workflow.add_edge("supervisor", "researcher")
    workflow.add_edge("supervisor", "coder")
    
    # 여러 에이전트 완료 후 aggregator로汇聚
    workflow.add_edge("researcher", "aggregator")
    workflow.add_edge("coder", "aggregator")
    
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()
```

### 3. 인간 검토 노드

```python
from langgraph.checkpoint.sqlite import SqliteSaver

def human_review_node(state: AgentState) -> AgentState:
    """인간 검토 노드"""
    
    # 실행 일시 중지, 인간 입력 대기
    print("\n🔍 인간 검토 필요:")
    print(f"현재 결과: {state['results']}")
    
    approval = input("\n계속 승인하시겠습니까? (yes/no): ")
    
    if approval.lower() == "yes":
        return {
            "messages": [HumanMessage(content="인간 검토 통과")],
            "next_agent": "continue"
        }
    else:
        feedback = input("수정 의견 제공: ")
        return {
            "messages": [HumanMessage(content=f"수정 필요: {feedback}")],
            "next_agent": "revise"
        }

# 그래프에 인간 검토 노드 추가
workflow.add_node("human_review", human_review_node)
workflow.add_edge("coder", "human_review")
```

### 4. 오류 처리 및 재시도

```python
def safe_agent_wrapper(agent_func, max_retries=3):
    """에이전트에 오류 처리 및 재시도 메커니즘 추가"""
    
    def wrapped_agent(state: AgentState) -> AgentState:
        for attempt in range(max_retries):
            try:
                return agent_func(state)
            except Exception as e:
                print(f"⚠️ 에이전트 실행 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    # 마지막 시도 실패 시 오류 상태 반환
                    return {
                        "messages": [AIMessage(content=f"실패: {str(e)}")],
                        "results": {
                            **state.get("results", {}),
                            "error": str(e)
                        },
                        "next_agent": "FINISH"
                    }
                
                # 대기 후 재시도
                import time
                time.sleep(2 ** attempt)
        
        return state
    
    return wrapped_agent

# 래퍼 사용
workflow.add_node("researcher", safe_agent_wrapper(researcher_node))
```

### 5. 동적 에이전트 등록

```python
class AgentRegistry:
    """에이전트 레지스트리"""
    
    def __init__(self):
        self.agents = {}
    
    def register(self, name: str, agent_func, description: str):
        """새 에이전트 등록"""
        self.agents[name] = {
            "function": agent_func,
            "description": description
        }
    
    def get_agent(self, name: str):
        """에이전트 가져오기"""
        return self.agents.get(name, {}).get("function")
    
    def list_agents(self) -> str:
        """사용 가능한 모든 에이전트 나열"""
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.agents.items()
        ])

# 사용 예제
registry = AgentRegistry()
registry.register("researcher", researcher_node, "정보 검색 및 데이터 분석")
registry.register("coder", coder_node, "코드 작성 및 기술 구현")
registry.register("writer", writer_node, "콘텐츠创作 및 문서 작성")

# Supervisor에서 사용
def dynamic_supervisor(state: AgentState, registry: AgentRegistry):
    """동적 에이전트를 지원하는 Supervisor"""
    
    available_agents = registry.list_agents()
    
    prompt = f"""
    사용 가능한 에이전트:
    {available_agents}
    
    작업: {state['task']}
    
    적절한 에이전트를 선택하세요.
    """
    
    # ... 의사결정 로직
```

### 6. 스트리밍 출력

```python
async def stream_multi_agent_system(task: str):
    """스트리밍 출력을 지원하는 다중 에이전트 시스템"""
    
    app = create_multi_agent_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "results": {},
        "executed_agents": []
    }
    
    # 스트리밍 실행
    async for event in app.astream(initial_state):
        for node_name, node_state in event.items():
            print(f"\n📍 노드: {node_name}")
            
            if "messages" in node_state:
                latest_message = node_state["messages"][-1]
                print(f"💬 {latest_message.content}")
            
            # 실시간 진행 상황 표시
            if "executed_agents" in node_state:
                print(f"✅ 완료: {', '.join(node_state['executed_agents'])}")

# 스트리밍 시스템 실행
import asyncio
asyncio.run(stream_multi_agent_system("Web 애플리케이션 생성"))
```

---

## 자주 묻는 질문과 모범 사례

### 자주 묻는 질문

#### Q1: Supervisor가 무한 루프를 피하려면 어떻게 해야 합니까?

**해결책**:
```python
class AgentState(TypedDict):
    # ... 다른 필드
    iteration: int
    max_iterations: int

def supervisor_node(state: AgentState) -> AgentState:
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)
    
    if iteration >= max_iter:
        return {
            "messages": [AIMessage(content="최대 반복 횟수 도달")],
            "next_agent": "FINISH"
        }
    
    # ... 정상 의사결정 로직
    
    return {
        "iteration": iteration + 1,
        # ... 다른 반환값
    }
```

#### Q2: 에이전트 간의 의존 관계를 어떻게 처리합니까?

**해결책**:
```python
def supervisor_node(state: AgentState) -> AgentState:
    executed = state.get("executed_agents", [])
    results = state.get("results", {})
    
    # 의존 관계 정의
    dependencies = {
        "coder": ["researcher"],  # coder는 researcher에 의존
        "writer": ["researcher", "coder"]  # writer는 둘 다 의존
    }
    
    # 다음 에이전트 선택
    for agent, deps in dependencies.items():
        if agent not in executed:
            # 의존 조건 충족 확인
            if all(dep in executed for dep in deps):
                return {"next_agent": agent}
    
    return {"next_agent": "FINISH"}
```

#### Q3: LLM 호출 비용을 최적화하려면 어떻게 해야 합니까?

**해결책**:
1. **반복 쿼리 캐싱**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_llm_call(prompt: str) -> str:
    return llm.invoke(prompt).content
```

2. **다른 모델 사용**
```python
# Supervisor는 강력한 모델 사용
supervisor_llm = ChatOpenAI(model="gpt-4")

# 간단한 에이전트는 약한 모델 사용
agent_llm = ChatOpenAI(model="gpt-3.5-turbo")
```

3. **배치 처리**
```python
def batch_process_agents(tasks: list) -> list:
    """여러 작업 배치 처리"""
    prompts = [create_prompt(task) for task in tasks]
    responses = llm.batch(prompts)
    return responses
```

### 모범 사례

#### 1. 명확한 상태 설계

```python
class AgentState(TypedDict):
    # 핵심 필드
    task: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # 제어 흐름 필드
    next_agent: str
    iteration: int
    
    # 데이터 필드
    results: dict
    intermediate_steps: list
    
    # 메타데이터 필드
    executed_agents: list
    start_time: float
    metadata: dict
```

#### 2. 모듈식 에이전트 설계

```python
class BaseAgent:
    """에이전트 기본 클래스"""
    
    def __init__(self, name: str, llm, tools: list = None):
        self.name = name
        self.llm = llm
        self.tools = tools or []
    
    def execute(self, state: AgentState) -> AgentState:
        """에이전트 작업 실행"""
        raise NotImplementedError
    
    def create_prompt(self, state: AgentState) -> str:
        """프롬프트 생성"""
        raise NotImplementedError

class ResearcherAgent(BaseAgent):
    """연구원 에이전트"""
    
    def create_prompt(self, state: AgentState) -> str:
        return f"연구 작업: {state['task']}"
    
    def execute(self, state: AgentState) -> AgentState:
        prompt = self.create_prompt(state)
        result = self.llm.invoke(prompt).content
        
        return {
            "results": {**state["results"], self.name: result},
            "executed_agents": state["executed_agents"] + [self.name]
        }
```

#### 3. 완전한 로깅 기록

```python
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_agent(agent_func):
    """에이전트에 로깅 데코레이터 추가"""
    
    def wrapper(state: AgentState) -> AgentState:
        agent_name = agent_func.__name__
        
        logger.info(f"[{datetime.now()}] {agent_name} 실행 시작")
        logger.debug(f"입력 상태: {state}")
        
        try:
            result = agent_func(state)
            logger.info(f"[{datetime.now()}] {agent_name} 실행 성공")
            logger.debug(f"출력 상태: {result}")
            return result
        except Exception as e:
            logger.error(f"[{datetime.now()}] {agent_name} 실행 실패: {e}")
            raise
    
    return wrapper

@logged_agent
def researcher_node(state: AgentState) -> AgentState:
    # ... 구현
    pass
```

#### 4. 테스트 전략

```python
import unittest

class TestMultiAgentSystem(unittest.TestCase):
    
    def setUp(self):
        """테스트 전 준비"""
        self.app = create_multi_agent_graph()
    
    def test_simple_task(self):
        """간단한 작업 테스트"""
        state = {
            "task": "Python 모범 사례 연구",
            "messages": [],
            "results": {},
            "executed_agents": []
        }
        
        result = self.app.invoke(state)
        
        # 결과 검증
        self.assertIn("researcher", result["executed_agents"])
        self.assertIn("researcher", result["results"])
    
    def test_complex_workflow(self):
        """복잡한 워크플로 테스트"""
        state = {
            "task": "Web 애플리케이션 생성 및 문서 작성",
            "messages": [],
            "results": {},
            "executed_agents": []
        }
        
        result = self.app.invoke(state)
        
        # 필요한 모든 에이전트가 실행되었는지 검증
        expected_agents = ["researcher", "coder", "writer"]
        for agent in expected_agents:
            self.assertIn(agent, result["executed_agents"])
    
    def test_error_handling(self):
        """오류 처리 테스트"""
        # 오류 시나리오 시뮬레이션
        pass

if __name__ == "__main__":
    unittest.main()
```

#### 5. 성능 모니터링

```python
import time
from functools import wraps

class PerformanceMonitor:
    """성능 모니터"""
    
    def __init__(self):
        self.metrics = {}
    
    def track(self, agent_name: str):
        """에이전트 성능 추적 데코레이터"""
        
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
        """성능 보고서 생성"""
        report = "성능 보고서\n" + "=" * 50 + "\n"
        
        for agent, metrics in self.metrics.items():
            durations = [m["duration"] for m in metrics]
            avg_duration = sum(durations) / len(durations)
            
            report += f"\n{agent}:\n"
            report += f"  호출 횟수: {len(metrics)}\n"
            report += f"  평균 소요 시간: {avg_duration:.2f}초\n"
            report += f"  총 소요 시간: {sum(durations):.2f}초\n"
        
        return report

# 사용 예제
monitor = PerformanceMonitor()

@monitor.track("researcher")
def researcher_node(state: AgentState) -> AgentState:
    # ... 구현
    pass

# 실행 후 보고서 확인
print(monitor.get_report())
```

---

## 요약

이 튜토리얼에서는 LangGraph를 사용하여 다중 에이전트 + Supervisor 시스템을 구축하는 방법을 자세히 설명했습니다:

### 핵심 포인트

1. **상태 관리**: TypedDict를 사용하여 명확한 상태 구조 정의
2. **노드 설계**: 각 에이전트는 독립적인 노드로,职责이 명확함
3. **Supervisor 패턴**: 중앙 조정자가 작업 할당 및 결과 통합 담당
4. **조건부 라우팅**: 상태에 따라 실행 프로세스 동적 결정
5. **확장성**: 새로운 에이전트 및 기능 추가 용이

### 적용 시나리오

- 복잡한 연구 및 분석 작업
- 다단계 콘텐츠创作
- 다양한 전문 기술이 필요한 프로젝트
- 기업급 자동화 워크플로

### 고급 방향

- 외부 도구 통합 (검색, 데이터베이스, API)
- 더 복잡한 협업 패턴 구현 (계층적, 메시)
- 인간-에이전트 협업 기능 추가
- 성능 및 비용 최적화
- 프로덕션 환경 배포

---

## 참고 자료

- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangChain 문서](https://python.langchain.com/)
- [다중 에이전트 시스템 설계 패턴](https://arxiv.org/abs/2308.10848)
- [LangGraph GitHub 저장소](https://github.com/langchain-ai/langgraph)

---

**문서 버전**: 1.0  
**최종 업데이트**: 2026년 2월  
**저자**: AI 어시스턴트  
**라이선스**: MIT License