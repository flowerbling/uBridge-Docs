# Руководство по реализации мультиагентной системы LangGraph с Supervisor

## Содержание

1. [Введение](#введение)
2. [Основные концепции LangGraph](#основные-концепции-langgraph)
3. [Проектирование архитектуры мультиагентной системы](#проектирование-архитектуры-мультиагентной-системы)
4. [Детальный разбор паттерна Supervisor](#детальный-разбор-паттерна-supervisor)
5. [Полный пример реализации](#полный-пример-реализации)
6. [Расширенные функции и оптимизация](#расширенные-функции-и-оптимизация)
7. [Частые вопросы и лучшие практики](#частые-вопросы-и-лучшие-практики)

---

## Введение

### Что такое LangGraph?

LangGraph — это фреймворк для построения состоятельных многопользовательских приложений на основе экосистемы LangChain. Он использует концепцию графа (Graph) для организации сложных рабочих процессов ИИ, что особенно удобно для построения сложных систем, требующих взаимодействия нескольких агентов.

### Почему стоит использовать паттерн мультиагентной системы с Supervisor?

В сложных сценариях выполнения задач один агент часто не может эффективно обработать все проблемы. Паттерн мультиагентной системы с Supervisor обладает следующими преимуществами:

- **Специализация**: каждый агент сосредоточен на конкретной области (исследования, программирование, написание текстов)
- **Параллельная обработка**: несколько агентов могут одновременно обрабатывать различные подзадачи
- **Гибкое масштабирование**: легко добавлять новых специализированных агентов
- **Централизованная координация**: Supervisor отвечает за распределение задач и интеграцию результатов

### Области применения

- Сложные исследовательские и аналитические задачи
- Многоэтапные процессы создания контента
- Проекты, требующие различных профессиональных навыков
- Корпоративные автоматизированные рабочие процессы

---

## Основные концепции LangGraph

### 1. Состояние (State)

Состояние — это структура данных, передаваемая через граф, которая записывает контекстную информацию всего рабочего процесса.

```python
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """Определение структуры состояния агентной системы"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str  # Следующий узел для выполнения
    task: str  # Описание текущей задачи
    results: dict  # Результаты выполнения агентов
```

**Ключевые моменты**:
- `Annotated` используется для определения стратегии обновления состояния
- `operator.add` означает, что список сообщений накапливается, а не перезаписывается
- Состояние передаётся и обновляется между узлами

### 2. Узел (Node)

Узел — это единица выполнения в графе, каждый узел представляет собой функцию, которая принимает состояние и возвращает обновлённое состояние.

```python
def researcher_node(state: AgentState) -> AgentState:
    """Узел агента-исследователя"""
    # Выполнение исследовательской задачи
    result = perform_research(state["task"])
    
    return {
        "messages": [HumanMessage(content=f"Исследование завершено: {result}")],
        "results": {**state.get("results", {}), "research": result}
    }
```

### 3. Ребро (Edge)

Ребра определяют связи между узлами и бывают двух типов:

**Обычное ребро (Normal Edge)**: фиксированный путь выполнения
```python
graph.add_edge("node_a", "node_b")
```

**Условное ребро (Conditional Edge)**: динамическое определение следующего узла на основе состояния
```python
def route_function(state: AgentState) -> str:
    """Определение маршрута на основе состояния"""
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

### 4. Граф (Graph)

Граф — это контейнер всего рабочего процесса, организующий все узлы и рёбра.

```python
from langgraph.graph import StateGraph, END

# Создание графа
workflow = StateGraph(AgentState)

# Добавление узлов
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)

# Установка точки входа
workflow.set_entry_point("supervisor")

# Компиляция графа
app = workflow.compile()
```

---

## Проектирование архитектуры мультиагентной системы

### Архитектурная схема системы

```
┌─────────────────────────────────────────┐
│           Входные данные пользователя    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Supervisor (Координатор)        │
│  - Анализ задачи                        │
│  - Выбор подходящего агента             │
│  - Координация процесса выполнения      │
└──────┬──────────┬──────────┬────────────┘
       │          │          │
       ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│Researcher│ │  Coder   │ │  Writer  │
│  Agent   │ │  Agent   │ │  Agent   │
│          │ │          │ │          │
│- Поиск   │ │- Генерация│ │- Создание│
│  информ. │ │  кода    │ │  контента│
│- Анализ  │ │- Реценз. │ │- Написание│
│  данных  │ │  кода    │ │  документ.│
└──────┬───┘ └────┬─────┘ └────┬─────┘
       │          │          │
       └──────────┼──────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Интеграция    │
         │  и вывод       │
         └────────────────┘
```

### Определение ролей агентов

#### 1. Агент-Supervisor (Координатор)

**Обязанности**:
- Понимание задачи пользователя
- Разложение сложной задачи на подзадачи
- Выбор подходящего специализированного агента
- Интеграция результатов от агентов
- Определение завершённости задачи

**Ключевые способности**:
- Способность к планированию задач
- Способность к принятию решений
- Способность к координации

#### 2. Агент-Researcher (Исследователь)

**Обязанности**:
- Поиск и сбор информации
- Анализ данных
- Фоновое исследование
- Проверка фактов

**Инструменты**:
- API поисковых систем
- Запросы к базам данных
- Инструменты парсинга документов

#### 3. Агент-Coder (Программист)

**Обязанности**:
- Написание кода
- Рецензирование кода
- Проектирование технических решений
- Отладка и оптимизация

**Инструменты**:
- Среда выполнения кода
- Инструменты анализа кода
- Инструменты генерации документации

#### 4. Агент-Writer (Писатель)

**Обязанности**:
- Создание контента
- Написание документации
- Генерация отчётов
- Редактирование контента

**Инструменты**:
- Шаблонизаторы
- Инструменты конвертации форматов
- Инструменты проверки грамматики

---

## Детальный разбор паттерна Supervisor

### Рабочий процесс Supervisor

```
1. Получение задачи
   ↓
2. Анализ требований задачи
   ↓
3. Выбор подходящего агента
   ↓
4. Распределение подзадач
   ↓
5. Мониторинг прогресса выполнения
   ↓
6. Сбор результатов от агентов
   ↓
7. Определение необходимости продолжения
   ├─ Да → Возврат к шагу 3
   └─ Нет → Интеграция результатов и вывод
```

### Ключевые моменты реализации Supervisor

#### 1. Анализ и планирование задачи

```python
def analyze_task(task: str) -> dict:
    """Анализ задачи и генерация плана выполнения"""
    prompt = f"""
    Проанализируйте следующую задачу и определите, какие специализированные агенты необходимы:
    
    Задача: {task}
    
    Доступные агенты:
    - researcher: поиск информации, анализ данных
    - coder: написание кода, техническая реализация
    - writer: создание контента, написание документации
    
    Верните план выполнения в формате JSON:
    {{
        "agents_needed": ["agent1", "agent2"],
        "execution_order": ["step1", "step2"],
        "expected_outcome": "Описание ожидаемого результата"
    }}
    """
    
    # Вызов LLM для анализа
    response = llm.invoke(prompt)
    return parse_json(response)
```

#### 2. Логика выбора агента

```python
def select_next_agent(state: AgentState) -> str:
    """Выбор следующего агента на основе текущего состояния"""
    
    # Проверка завершённости задачи
    if is_task_complete(state):
        return "FINISH"
    
    # Получение списка выполненных агентов
    executed = state.get("executed_agents", [])
    
    # Выбор следующего агента согласно плану задачи
    plan = state.get("plan", {})
    remaining_agents = [
        agent for agent in plan["agents_needed"] 
        if agent not in executed
    ]
    
    if remaining_agents:
        return remaining_agents[0]
    
    # Если требуется итеративное улучшение
    if needs_refinement(state):
        return determine_refinement_agent(state)
    
    return "FINISH"
```

#### 3. Интеграция результатов

```python
def integrate_results(state: AgentState) -> str:
    """Интеграция результатов выполнения агентов"""
    results = state.get("results", {})
    
    prompt = f"""
    Пожалуйста, интегрируйте результаты работы специализированных агентов и сгенерируйте финальный вывод:
    
    Результаты исследования: {results.get('researcher', 'N/A')}
    Реализация кода: {results.get('coder', 'N/A')}
    Содержание документации: {results.get('writer', 'N/A')}
    
    Исходная задача: {state['task']}
    
    Пожалуйста, сгенерируйте полный, связный финальный результат.
    """
    
    final_output = llm.invoke(prompt)
    return final_output
```

---

## Полный пример реализации

### Подготовка окружения

```bash
# Установка зависимостей
pip install langgraph langchain langchain-openai langchain-community
```

### Полная реализация кода

```python
import operator
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json

# ============================================
# 1. Определение структуры состояния
# ============================================

class AgentState(TypedDict):
    """Состояние мультиагентной системы"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str  # Исходная задача
    next_agent: str  # Следующий агент для выполнения
    results: dict  # Результаты агентов
    executed_agents: list  # Список выполненных агентов
    iteration: int  # Количество итераций
    final_output: str  # Финальный вывод

# ============================================
# 2. Инициализация LLM
# ============================================

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ============================================
# 3. Определение агента Supervisor
# ============================================

def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor отвечает за анализ задач и调度 агентов"""
    
    messages = state["messages"]
    task = state["task"]
    results = state.get("results", {})
    executed = state.get("executed_agents", [])
    
    # Построение промпта для Supervisor
    system_prompt = """
    Вы являетесь координатором задач (Supervisor), управляющим несколькими специализированными агентами.
    
    Доступные агенты:
    - researcher: специалист по поиску информации, анализу данных, фоновым исследованиям
    - coder: специалист по написанию кода, технической реализации, проектированию алгоритмов
    - writer: специалист по созданию контента, написанию документации, генерации отчётов
    
    Ваши обязанности:
    1. Анализ требований задачи
    2. Выбор подходящего агента для выполнения подзадач
    3. Определение завершённости задачи
    
    Пожалуйста, верните решение в формате JSON:
    {
        "next_agent": "researcher/coder/writer/FINISH",
        "reason": "Причина выбора",
        "subtask": "Конкретная подзадача для этого агента"
    }
    """
    
    context = f"""
    Исходная задача: {task}
    
    Выполненные агенты: {executed}
    
    Текущие результаты:
    {json.dumps(results, ensure_ascii=False, indent=2)}
    
    Пожалуйста, решите следующий шаг.
    """
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context}
    ])
    
    # Парсинг решения
    try:
        decision = json.loads(response.content)
    except:
        # При ошибке парсинга завершаем по умолчанию
        decision = {"next_agent": "FINISH", "reason": "Ошибка парсинга", "subtask": ""}
    
    return {
        "messages": [AIMessage(content=f"Решение Supervisor: {decision['reason']}")],
        "next_agent": decision["next_agent"],
        "results": {**results, "supervisor_decision": decision}
    }

# ============================================
# 4. Определение узлов специализированных агентов
# ============================================

def researcher_node(state: AgentState) -> AgentState:
    """Агент-исследователь - отвечает за поиск информации и анализ"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    
    prompt = f"""
    Вы являетесь профессиональным исследователем, специализирующимся на поиске информации и анализе данных.
    
    Задача: {subtask}
    
    Пожалуйста, проведите глубокое исследование и предоставьте подробные результаты анализа.
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Researcher завершён: {result[:100]}...")],
        "results": {**state["results"], "researcher": result},
        "executed_agents": executed + ["researcher"]
    }

def coder_node(state: AgentState) -> AgentState:
    """Агент-программист - отвечает за написание кода"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    research_result = state["results"].get("researcher", "")
    
    prompt = f"""
    Вы являетесь профессиональным программистом, специализирующимся на написании кода и технической реализации.
    
    Задача: {subtask}
    
    Справочная информация: {research_result}
    
    Пожалуйста, предоставьте полную реализацию кода с комментариями и пояснениями.
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Coder завершён: Код сгенерирован")],
        "results": {**state["results"], "coder": result},
        "executed_agents": executed + ["coder"]
    }

def writer_node(state: AgentState) -> AgentState:
    """Агент-писатель - отвечает за создание контента"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    research_result = state["results"].get("researcher", "")
    code_result = state["results"].get("coder", "")
    
    prompt = f"""
    Вы являетесь профессиональным писателем, специализирующимся на создании контента и написании документации.
    
    Задача: {subtask}
    
    Материалы исследования: {research_result}
    Реализация кода: {code_result}
    
    Пожалуйста, создайте качественный контент или документацию.
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Writer завершён: Контент создан")],
        "results": {**state["results"], "writer": result},
        "executed_agents": executed + ["writer"]
    }

# ============================================
# 5. Определение логики маршрутизации
# ============================================

def route_agent(state: AgentState) -> Literal["researcher", "coder", "writer", "end"]:
    """Маршрутизация к соответствующему агенту на основе решения Supervisor"""
    
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
# 6. Построение графа рабочего процесса
# ============================================

def create_multi_agent_graph():
    """Создание графа рабочего процесса мультиагентной системы с Supervisor"""
    
    # Создание графа состояний
    workflow = StateGraph(AgentState)
    
    # Добавление узлов
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("writer", writer_node)
    
    # Установка точки входа
    workflow.set_entry_point("supervisor")
    
    # Добавление условных рёбер: от supervisor к различным агентам согласно решению
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
    
    # Добавление обычных рёбер: возврат к supervisor после выполнения агентов
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("coder", "supervisor")
    workflow.add_edge("writer", "supervisor")
    
    # Компиляция графа
    app = workflow.compile()
    
    return app

# ============================================
# 7. Пример выполнения
# ============================================

def run_multi_agent_system(task: str):
    """Запуск мультиагентной системы"""
    
    # Создание рабочего процесса
    app = create_multi_agent_graph()
    
    # Инициализация состояния
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "next_agent": "",
        "results": {},
        "executed_agents": [],
        "iteration": 0,
        "final_output": ""
    }
    
    # Выполнение рабочего процесса
    print(f"🚀 Начало выполнения задачи: {task}\n")
    print("=" * 60)
    
    final_state = app.invoke(initial_state)
    
    # Вывод результатов
    print("\n" + "=" * 60)
    print("📊 Результаты выполнения:\n")
    
    for agent, result in final_state["results"].items():
        if agent != "supervisor_decision":
            print(f"\n【{agent.upper()}】")
            print("-" * 60)
            print(result[:500] + "..." if len(result) > 500 else result)
    
    return final_state

# ============================================
# 8. Примеры тестирования
# ============================================

if __name__ == "__main__":
    # Пример задачи
    task = """
    Создайте программу на Python, реализующую простую систему управления задачами.
    Требования:
    1. Поддержка добавления, удаления и просмотра задач
    2. Использование базы данных SQLite
    3. Предоставление интерфейса командной строки
    4. Написание полной документации по использованию
    """
    
    result = run_multi_agent_system(task)
```

### Описание процесса выполнения

1. **Ввод задачи**: пользователь отправляет сложную задачу
2. **Анализ Supervisor**: анализ задачи, определение необходимых агентов
3. **Выполнение Researcher**: сбор соответствующих технических материалов и лучших практик
4. **Возврат к Supervisor**: передача результатов исследования
5. **Повторное решение Supervisor**: на основе результатов исследования, распределение задачи на программирование
6. **Выполнение Coder**: написание полной реализации кода
7. **Возврат к Supervisor**: передача кода
8. **Повторное решение Supervisor**: распределение задачи на написание документации
9. **Выполнение Writer**: написание документации по использованию
10. **Возврат к Supervisor**: передача документации
11. **Определение завершения Supervisor**: интеграция всех результатов, вывод финального продукта

---

## Расширенные функции и оптимизация

### 1. Добавление функции памяти

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Создание чекпоинт-сохранителя
memory = SqliteSaver.from_conn_string(":memory:")

# Добавление чекпоинтов при компиляции
app = workflow.compile(checkpointer=memory)

# Указание ID потока при выполнении
config = {"configurable": {"thread_id": "conversation_1"}}
result = app.invoke(initial_state, config=config)
```

### 2. Параллельное выполнение агентов

```python
from langgraph.graph import START

def create_parallel_graph():
    """Создание графа с поддержкой параллельного выполнения"""
    workflow = StateGraph(AgentState)
    
    # Добавление узлов
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("aggregator", aggregator_node)
    
    # Параллельная дистрибуция от supervisor к нескольким агентам
    workflow.add_edge("supervisor", "researcher")
    workflow.add_edge("supervisor", "coder")
    
    # После завершения нескольких агентов - объединение в aggregator
    workflow.add_edge("researcher", "aggregator")
    workflow.add_edge("coder", "aggregator")
    
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()
```

### 3. Узел ручной проверки

```python
from langgraph.checkpoint.sqlite import SqliteSaver

def human_review_node(state: AgentState) -> AgentState:
    """Узел ручной проверки"""
    
    # Приостановка выполнения, ожидание ручного ввода
    print("\n🔍 Требуется ручная проверка:")
    print(f"Текущие результаты: {state['results']}")
    
    approval = input("\nРазрешить продолжение? (yes/no): ")
    
    if approval.lower() == "yes":
        return {
            "messages": [HumanMessage(content="Ручная проверка пройдена")],
            "next_agent": "continue"
        }
    else:
        feedback = input("Пожалуйста, предоставьте комментарии по изменениям: ")
        return {
            "messages": [HumanMessage(content=f"Требуются изменения: {feedback}")],
            "next_agent": "revise"
        }

# Добавление узла ручной проверки в граф
workflow.add_node("human_review", human_review_node)
workflow.add_edge("coder", "human_review")
```

### 4. Обработка ошибок и повторные попытки

```python
def safe_agent_wrapper(agent_func, max_retries=3):
    """Обёртка для агента с обработкой ошибок и механизмом повторных попыток"""
    
    def wrapped_agent(state: AgentState) -> AgentState:
        for attempt in range(max_retries):
            try:
                return agent_func(state)
            except Exception as e:
                print(f"⚠️ Ошибка выполнения агента (попытка {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    # После последней неудачной попытки возвращаем состояние ошибки
                    return {
                        "messages": [AIMessage(content=f"Ошибка выполнения: {str(e)}")],
                        "results": {
                            **state.get("results", {}),
                            "error": str(e)
                        },
                        "next_agent": "FINISH"
                    }
                
                # Ожидание перед повторной попыткой
                import time
                time.sleep(2 ** attempt)
        
        return state
    
    return wrapped_agent

# Использование обёртки
workflow.add_node("researcher", safe_agent_wrapper(researcher_node))
```

### 5. Динамическая регистрация агентов

```python
class AgentRegistry:
    """Реестр агентов"""
    
    def __init__(self):
        self.agents = {}
    
    def register(self, name: str, agent_func, description: str):
        """Регистрация нового агента"""
        self.agents[name] = {
            "function": agent_func,
            "description": description
        }
    
    def get_agent(self, name: str):
        """Получение агента"""
        return self.agents.get(name, {}).get("function")
    
    def list_agents(self) -> str:
        """Список всех доступных агентов"""
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.agents.items()
        ])

# Пример использования
registry = AgentRegistry()
registry.register("researcher", researcher_node, "Поиск информации и анализ данных")
registry.register("coder", coder_node, "Написание кода и техническая реализация")
registry.register("writer", writer_node, "Создание контента и написание документации")

# Использование в Supervisor
def dynamic_supervisor(state: AgentState, registry: AgentRegistry):
    """Supervisor с поддержкой динамических агентов"""
    
    available_agents = registry.list_agents()
    
    prompt = f"""
    Доступные агенты:
    {available_agents}
    
    Задача: {state['task']}
    
    Пожалуйста, выберите подходящего агента.
    """
    
    # ... логика принятия решений
```

### 6. Потоковый вывод

```python
async def stream_multi_agent_system(task: str):
    """Мультиагентная система с поддержкой потокового вывода"""
    
    app = create_multi_agent_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "results": {},
        "executed_agents": []
    }
    
    # Потоковое выполнение
    async for event in app.astream(initial_state):
        for node_name, node_state in event.items():
            print(f"\n📍 Узел: {node_name}")
            
            if "messages" in node_state:
                latest_message = node_state["messages"][-1]
                print(f"💬 {latest_message.content}")
            
            # Отображение прогресса в реальном времени
            if "executed_agents" in node_state:
                print(f"✅ Завершено: {', '.join(node_state['executed_agents'])}")

# Запуск потоковой системы
import asyncio
asyncio.run(stream_multi_agent_system("Создать веб-приложение"))
```

---

## Частые вопросы и лучшие практики

### Частые вопросы

#### Q1: Как избежать бесконечных циклов в Supervisor?

**Решение**:
```python
class AgentState(TypedDict):
    # ... другие поля
    iteration: int
    max_iterations: int

def supervisor_node(state: AgentState) -> AgentState:
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)
    
    if iteration >= max_iter:
        return {
            "messages": [AIMessage(content="Достигнуто максимальное количество итераций")],
            "next_agent": "FINISH"
        }
    
    # ... обычная логика принятия решений
    
    return {
        "iteration": iteration + 1,
        # ... другие возвращаемые значения
    }
```

#### Q2: Как обработать зависимости между агентами?

**Решение**:
```python
def supervisor_node(state: AgentState) -> AgentState:
    executed = state.get("executed_agents", [])
    results = state.get("results", {})
    
    # Определение зависимостей
    dependencies = {
        "coder": ["researcher"],  # coder зависит от researcher
        "writer": ["researcher", "coder"]  # writer зависит от обоих
    }
    
    # Выбор следующего агента
    for agent, deps in dependencies.items():
        if agent not in executed:
            # Проверка выполнения зависимостей
            if all(dep in executed for dep in deps):
                return {"next_agent": agent}
    
    return {"next_agent": "FINISH"}
```

#### Q3: Как оптимизировать затраты на вызовы LLM?

**Решение**:
1. **Кэширование повторяющихся запросов**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_llm_call(prompt: str) -> str:
    return llm.invoke(prompt).content
```

2. **Использование различных моделей**
```python
# Supervisor использует мощную модель
supervisor_llm = ChatOpenAI(model="gpt-4")

# Простые агенты используют более слабую модель
agent_llm = ChatOpenAI(model="gpt-3.5-turbo")
```

3. **Пакетная обработка**
```python
def batch_process_agents(tasks: list) -> list:
    """Пакетная обработка нескольких задач"""
    prompts = [create_prompt(task) for task in tasks]
    responses = llm.batch(prompts)
    return responses
```

### Лучшие практики

#### 1. Чёткое проектирование состояния

```python
class AgentState(TypedDict):
    # Основные поля
    task: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Поля управления потоком
    next_agent: str
    iteration: int
    
    # Поля данных
    results: dict
    intermediate_steps: list
    
    # Метаданные
    executed_agents: list
    start_time: float
    metadata: dict
```

#### 2. Модульное проектирование агентов

```python
class BaseAgent:
    """Базовый класс агента"""
    
    def __init__(self, name: str, llm, tools: list = None):
        self.name = name
        self.llm = llm
        self.tools = tools or []
    
    def execute(self, state: AgentState) -> AgentState:
        """Выполнение задачи агента"""
        raise NotImplementedError
    
    def create_prompt(self, state: AgentState) -> str:
        """Создание промпта"""
        raise NotImplementedError

class ResearcherAgent(BaseAgent):
    """Агент-исследователь"""
    
    def create_prompt(self, state: AgentState) -> str:
        return f"Исследовательская задача: {state['task']}"
    
    def execute(self, state: AgentState) -> AgentState:
        prompt = self.create_prompt(state)
        result = self.llm.invoke(prompt).content
        
        return {
            "results": {**state["results"], self.name: result},
            "executed_agents": state["executed_agents"] + [self.name]
        }
```

#### 3. Полное логирование

```python
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_agent(agent_func):
    """Декоратор для добавления логирования к агенту"""
    
    def wrapper(state: AgentState) -> AgentState:
        agent_name = agent_func.__name__
        
        logger.info(f"[{datetime.now()}] {agent_name} начал выполнение")
        logger.debug(f"Входное состояние: {state}")
        
        try:
            result = agent_func(state)
            logger.info(f"[{datetime.now()}] {agent_name} выполнен успешно")
            logger.debug(f"Выходное состояние: {result}")
            return result
        except Exception as e:
            logger.error(f"[{datetime.now()}] {agent_name} выполнен с ошибкой: {e}")
            raise
    
    return wrapper

@logged_agent
def researcher_node(state: AgentState) -> AgentState:
    # ... реализация
    pass
```

#### 4. Стратегия тестирования

```python
import unittest

class TestMultiAgentSystem(unittest.TestCase):
    
    def setUp(self):
        """Подготовка перед тестами"""
        self.app = create_multi_agent_graph()
    
    def test_simple_task(self):
        """Тест простой задачи"""
        state = {
            "task": "Исследовать лучшие практики Python",
            "messages": [],
            "results": {},
            "executed_agents": []
        }
        
        result = self.app.invoke(state)
        
        # Проверка результатов
        self.assertIn("researcher", result["executed_agents"])
        self.assertIn("researcher", result["results"])
    
    def test_complex_workflow(self):
        """Тест сложного рабочего процесса"""
        state = {
            "task": "Создать веб-приложение и написать документацию",
            "messages": [],
            "results": {},
            "executed_agents": []
        }
        
        result = self.app.invoke(state)
        
        # Проверка выполнения всех необходимых агентов
        expected_agents = ["researcher", "coder", "writer"]
        for agent in expected_agents:
            self.assertIn(agent, result["executed_agents"])
    
    def test_error_handling(self):
        """Тест обработки ошибок"""
        # Симуляция сценария ошибки
        pass

if __name__ == "__main__":
    unittest.main()
```

#### 5. Мониторинг производительности

```python
import time
from functools import wraps

class PerformanceMonitor:
    """Монитор производительности"""
    
    def __init__(self):
        self.metrics = {}
    
    def track(self, agent_name: str):
        """Декоратор для отслеживания производительности агента"""
        
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
        """Генерация отчёта о производительности"""
        report = "Отчёт о производительности\n" + "=" * 50 + "\n"
        
        for agent, metrics in self.metrics.items():
            durations = [m["duration"] for m in metrics]
            avg_duration = sum(durations) / len(durations)
            
            report += f"\n{agent}:\n"
            report += f"  Количество вызовов: {len(metrics)}\n"
            report += f"  Среднее время: {avg_duration:.2f}с\n"
            report += f"  Общее время: {sum(durations):.2f}с\n"
        
        return report

# Пример использования
monitor = PerformanceMonitor()

@monitor.track("researcher")
def researcher_node(state: AgentState) -> AgentState:
    # ... реализация
    pass

# Просмотр отчёта после выполнения
print(monitor.get_report())
```

---

## Заключение

В данном руководстве подробно описано, как построить мультиагентную систему с Supervisor с использованием LangGraph:

### Ключевые моменты

1. **Управление состоянием**: использование TypedDict для определения чёткой структуры состояния
2. **Проектирование узлов**: каждый агент является независимым узлом с чёткими обязанностями
3. **Паттерн Supervisor**: центральный координатор отвечает за распределение задач и интеграцию результатов
4. **Условная маршрутизация**: динамическое определение потока выполнения на основе состоя