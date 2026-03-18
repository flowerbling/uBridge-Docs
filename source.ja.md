# LangGraph マルチエージェント + スーパーバイザー システム実装チュートリアル

## 目次

1. [はじめに](#はじめに)
2. [LangGraph コアコンセプト](#langgraph-コアコンセプト)
3. [マルチエージェントシステムアーキテクチャ設計](#マルチエージェントシステムアーキテクチャ設計)
4. [スーパーバイザーモード詳細](#スーパーバイザーモード詳細)
5. [完全実装例](#完全実装例)
6. [高度な機能と最適化](#高度な機能と最適化)
7. [よくある問題とベストプラクティス](#よくある問題とベストプラクティス)

---

## はじめに

### LangGraph とは？

LangGraphは、状態を持ち、複数の参加者が関与するアプリケーションを構築するためのフレームワークで、LangChainエコシステムに基づいています。グラフ（Graph）の概念を使用して複雑なAIワークフローを整理し、複数のエージェントが協力する必要がある複雑なシステムの構築に特に適しています。

### マルチエージェント + スーパーバイザーモードを使用する理由？

複雑なタスクシナリオでは、単一のエージェントですべての問題を効率的に処理することは困難な場合があります。マルチエージェント + スーパーバイザーモードには以下の利点があります：

- **専門分化**：各エージェントは特定の分野（研究、コーディング、執筆など）に專門
- **並列処理**：複数のエージェントが同時に異なるサブタスクを処理可能
- **柔軟な拡張**：新しい専門エージェントを簡単に追加可能
- **集中調整**：スーパーバイザーがタスク分配と結果統合を担当

### 適用シナリオ

- 複雑な研究与分析タスク
- 複数ステップのコンテンツ作成プロセス
- 多种多様な専門スキルが必要なプロジェクト
- エンタープライズグレードの自動化ワークフロー

---

## LangGraph コアコンセプト

### 1. 状態（State）

状態はグラフ内で流转するデータ構造で、ワークフロー全体のコンテキスト情報を記録します。

```python
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """エージェントシステムの状態構造を定義"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str  # 次に実行するノード
    task: str  # 現在のタスクの説明
    results: dict  # 各エージェントの実行結果
```

**重要なポイント**：
- `Annotated` は状態更新戦略を定義するために使用
- `operator.add` はメッセージリストが上書きされるのではなく累積されることを示す
- 状態はノード間で传递・更新される

### 2. ノード（Node）

ノードはグラフ内の実行単位で、各ノードは関数であり、状態を受け取り更新された状態を返します。

```python
def researcher_node(state: AgentState) -> AgentState:
    """リサーチャーエージェントノード"""
    # 研究タスクを実行
    result = perform_research(state["task"])
    
    return {
        "messages": [HumanMessage(content=f"研究完了: {result}")],
        "results": {**state.get("results", {}), "research": result}
    }
```

### 3. エッジ（Edge）

エッジはノード間の接続関係を定義し、2種類あります：

**通常エッジ（Normal Edge）**：固定の流转パス
```python
graph.add_edge("node_a", "node_b")
```

**条件付きエッジ（Conditional Edge）**：状態に 따라動的に次のノードを決定
```python
def route_function(state: AgentState) -> str:
    """状態に 따라ルーティングを決定"""
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

### 4. グラフ（Graph）

グラフはワークフロー全体のコンテナで、すべてのノードとエッジを整理します。

```python
from langgraph.graph import StateGraph, END

# グラフを作成
workflow = StateGraph(AgentState)

# ノードを追加
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)

# エントリーポイントを設定
workflow.set_entry_point("supervisor")

# グラフをコンパイル
app = workflow.compile()
```

---

## マルチエージェントシステムアーキテクチャ設計

### システムアーキテクチャ図

```
┌─────────────────────────────────────────┐
│           ユーザー入力タスク              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         スーパーバイザー                  │
│  - タスク分析                            │
│  - 適切なエージェント選択                  │
│  - 実行プロセスの調整                      │
└──────┬──────────┬──────────┬────────────┘
       │          │          │
       ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│Researcher│ │  Coder   │ │  Writer  │
│ エージェント│ │ エージェント│ │ エージェント│
│          │ │          │ │          │
│- 情報検索│ │- コード生成│ │- コンテンツ作成│
│- データ分析│ │- コードレビュー│ │- ドキュメント作成│
└──────┬───┘ └────┬─────┘ └────┬─────┘
       │          │          │
       └──────────┼──────────┘
                  │
                  ▼
         ┌────────────────┐
         │  結果統合と出力 │
         └────────────────┘
```

### エージェントロール定義

#### 1. スーパーバイザーエージェント

**職責**：
- ユーザータスクの理解
- 複雑なタスクをサブタスクに分解
- 適切な専門エージェントの選択
- 各エージェントの結果の統合
- タスク完了の判断

**コア能力**：
- タスク計画能力
- 意思決定能力
- 調整能力

#### 2. リサーチャーエージェント

**職責**：
- 情報検索と収集
- データ分析
- バックグラウンドリサーチ
- 事実確認

**ツール**：
- 検索エンジンAPI
- データベースクエリ
- ドキュメント解析ツール

#### 3. コーダーエージェント

**職責**：
- コーディング
- コードレビュー
- 技術方案設計
- デバッグと最適化

**ツール**：
- コード実行環境
- コード分析ツール
- ドキュメント生成ツール

#### 4. ライターエージェント

**職責**：
- コンテンツ作成
- ドキュメント作成
- レポート生成
- コンテンツブラッシュアップ

**ツール**：
- テンプレートエンジン
- フォーマット変換ツール
- 文法チェックツール

---

## スーパーバイザーモード詳細

### スーパーバイザーのワークフロー

```
1. タスク受信
   ↓
2. タスク要件分析
   ↓
3. 適切なエージェント選択
   ↓
4. サブタスク分配
   ↓
5. 実行進捗の監視
   ↓
6. エージェント結果の収集
   ↓
7. 継続が必要か判断
   ├─ はい → ステップ3に戻る
   └─ いいえ → 結果を統合して出力
```

### スーパーバイザー実装のポイント

#### 1. タスク分析と計画

```python
def analyze_task(task: str) -> dict:
    """タスクを分析して実行計画を生成"""
    prompt = f"""
    以下のタスクを分析し、どの専門エージェントが必要かを判断：
    
    タスク：{task}
    
    利用可能なエージェント：
    - researcher: 情報検索、データ分析
    - coder: コーディング、技術実装
    - writer: コンテンツ作成、ドキュメント作成
    
    JSON形式の実行計画を返してください：
    {{
        "agents_needed": ["agent1", "agent2"],
        "execution_order": ["step1", "step2"],
        "expected_outcome": "期待される結果の説明"
    }}
    """
    
    # LLMを呼び出して分析
    response = llm.invoke(prompt)
    return parse_json(response)
```

#### 2. エージェント選択ロジック

```python
def select_next_agent(state: AgentState) -> str:
    """現在の状態に基づいて次のエージェントを選択"""
    
    # タスクが完了したか確認
    if is_task_complete(state):
        return "FINISH"
    
    # 実行済みエージェントを取得
    executed = state.get("executed_agents", [])
    
    # タスク計画に基づいて次のエージェントを選択
    plan = state.get("plan", {})
    remaining_agents = [
        agent for agent in plan["agents_needed"] 
        if agent not in executed
    ]
    
    if remaining_agents:
        return remaining_agents[0]
    
    # 反復的な最適化が必要な場合
    if needs_refinement(state):
        return determine_refinement_agent(state)
    
    return "FINISH"
```

#### 3. 結果統合

```python
def integrate_results(state: AgentState) -> str:
    """各エージェントの実行結果を統合"""
    results = state.get("results", {})
    
    prompt = f"""
    以下の各専門エージェントの作業成果物を統合して、最終出力を生成してください：
    
    研究結果：{results.get('researcher', 'N/A')}
    コード実装：{results.get('coder', 'N/A')}
    ドキュメント内容：{results.get('writer', 'N/A')}
    
    元タスク：{state['task']}
    
    完全で一貫性のある最終結果を生成してください。
    """
    
    final_output = llm.invoke(prompt)
    return final_output
```

---

## 完全実装例

### 環境準備

```bash
# 依存関係をインストール
pip install langgraph langchain langchain-openai langchain-community
```

### 完全コード実装

```python
import operator
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json

# ============================================
# 1. 状態構造を定義
# ============================================

class AgentState(TypedDict):
    """マルチエージェントシステムの状態"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str  # 元タスク
    next_agent: str  # 次に実行するエージェント
    results: dict  # 各エージェントの結果
    executed_agents: list  # 実行済みエージェントリスト
    iteration: int  # 反復回数
    final_output: str  # 最終出力

# ============================================
# 2. LLMを初期化
# ============================================

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ============================================
# 3. スーパーバイザーエージェントを定義
# ============================================

def supervisor_node(state: AgentState) -> AgentState:
    """スーパーバイザーはタスク分析とエージェントスケジューリングを担当"""
    
    messages = state["messages"]
    task = state["task"]
    results = state.get("results", {})
    executed = state.get("executed_agents", [])
    
    # スーパーバイザーのプロンプトを構築
    system_prompt = """
    あなたはタスク調整者（スーパーバイザー）で、複数の専門エージェントを管理します。
    
    利用可能なエージェント：
    - researcher: 情報検索、データ分析、バックグラウンドリサーチが得意
    - coder: コーディング、技術実装、アルゴリズム設計が得意
    - writer: コンテンツ作成、ドキュメント作成、レポート生成が得意
    
    あなたの職責：
    1. タスク要件を分析
    2. 適切なエージェントを選択してサブタスクを実行
    3. タスクが完了したかを判断
    
    現在の状況に基づいて、JSON形式の決定を返してください：
    {
        "next_agent": "researcher/coder/writer/FINISH",
        "reason": "選択理由",
        "subtask": "そのエージェントに分配する具体的なサブタスク"
    }
    """
    
    context = f"""
    元タスク：{task}
    
    実行済みエージェント：{executed}
    
    現在の結果：
    {json.dumps(results, ensure_ascii=False, indent=2)}
    
    次の行動を決定してください。
    """
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context}
    ])
    
    # 決定を解析
    try:
        decision = json.loads(response.content)
    except:
        # 解析失敗時はデフォルトで終了
        decision = {"next_agent": "FINISH", "reason": "解析エラー", "subtask": ""}
    
    return {
        "messages": [AIMessage(content=f"スーパーバイザー決定: {decision['reason']}")],
        "next_agent": decision["next_agent"],
        "results": {**results, "supervisor_decision": decision}
    }

# ============================================
# 4. 専門エージェントノードを定義
# ============================================

def researcher_node(state: AgentState) -> AgentState:
    """リサーチャーエージェント - 情報検索と分析を担当"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    
    prompt = f"""
    あなたは専門のリサーチャーで、情報検索とデータ分析が得意です。
    
    タスク：{subtask}
    
    詳細な分析結果を提供してください。
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Researcher 完了: {result[:100]}...")],
        "results": {**state["results"], "researcher": result},
        "executed_agents": executed + ["researcher"]
    }

def coder_node(state: AgentState) -> AgentState:
    """コーダーエージェント - コーディングを担当"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    research_result = state["results"].get("researcher", "")
    
    prompt = f"""
    あなたは専門のプログラマーで、コーディングと技術実装が得意です。
    
    タスク：{subtask}
    
    参考情報：{research_result}
    
    コメントと説明を含む完全なコード実装を提供してください。
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Coder 完了: コード生成済み")],
        "results": {**state["results"], "coder": result},
        "executed_agents": executed + ["coder"]
    }

def writer_node(state: AgentState) -> AgentState:
    """ライターエージェント - コンテンツ作成を担当"""
    
    task = state["task"]
    supervisor_decision = state["results"].get("supervisor_decision", {})
    subtask = supervisor_decision.get("subtask", task)
    research_result = state["results"].get("researcher", "")
    code_result = state["results"].get("coder", "")
    
    prompt = f"""
    あなたは専門のライターで、コンテンツ作成とドキュメント作成が得意です。
    
    タスク：{subtask}
    
    研究資料：{research_result}
    コード実装：{code_result}
    
    高品質のコンテンツまたはドキュメントを作成してください。
    """
    
    response = llm.invoke(prompt)
    result = response.content
    
    executed = state.get("executed_agents", [])
    
    return {
        "messages": [AIMessage(content=f"Writer 完了: コンテンツ作成済み")],
        "results": {**state["results"], "writer": result},
        "executed_agents": executed + ["writer"]
    }

# ============================================
# 5. ルーティングロジックを定義
# ============================================

def route_agent(state: AgentState) -> Literal["researcher", "coder", "writer", "end"]:
    """スーパーバイザーの決定に基づいて相应のエージェントにルーティング"""
    
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
# 6. ワークフローグラフを構築
# ============================================

def create_multi_agent_graph():
    """マルチエージェント + スーパーバイザー ワークフローグラフを作成"""
    
    # 状態グラフを作成
    workflow = StateGraph(AgentState)
    
    # ノードを追加
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("writer", writer_node)
    
    # エントリーポイントを設定
    workflow.set_entry_point("supervisor")
    
    # 条件付きエッジを追加：supervisorから決定に基づいて不同のエージェントにルーティング
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
    
    # 通常エッジを追加：各エージェント完了後supervisorに戻る
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("coder", "supervisor")
    workflow.add_edge("writer", "supervisor")
    
    # グラフをコンパイル
    app = workflow.compile()
    
    return app

# ============================================
# 7. 実行例
# ============================================

def run_multi_agent_system(task: str):
    """マルチエージェントシステムを実行"""
    
    # ワークフローを作成
    app = create_multi_agent_graph()
    
    # 状態を初期化
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "next_agent": "",
        "results": {},
        "executed_agents": [],
        "iteration": 0,
        "final_output": ""
    }
    
    # ワークフローを実行
    print(f"🚀 タスク実行開始: {task}\n")
    print("=" * 60)
    
    final_state = app.invoke(initial_state)
    
    # 結果を出力
    print("\n" + "=" * 60)
    print("📊 実行結果：\n")
    
    for agent, result in final_state["results"].items():
        if agent != "supervisor_decision":
            print(f"\n【{agent.upper()}】")
            print("-" * 60)
            print(result[:500] + "..." if len(result) > 500 else result)
    
    return final_state

# ============================================
# 8. テスト例
# ============================================

if __name__ == "__main__":
    # サンプルタスク
    task = """
    PythonプログラムでシンプルなTodoリスト管理システムを作成してください。
    要件：
    1. 追加、削除、表示機能をサポート
    2. SQLiteデータベースで保存
    3. コマンドラインインターフェースを提供
    4. 完全な使用ドキュメントを作成
    """
    
    result = run_multi_agent_system(task)
```

### 実行フロー説明

1. **タスク入力**：ユーザーは複雑なタスクを提出
2. **スーパーバイザー分析**：タスクを分析し、必要なエージェントを決定
3. **リサーチャー実行**：関連技術資料とベストプラクティスを収集
4. **スーパーバイザーへ戻る**：研究結果を提出
5. **スーパーバイザー再決定**：研究結果に基づいてコーディングタスクを分配
6. **コーダー実行**：完全なコード実装を作成
7. **スーパーバイザーへ戻る**：コードを提出
8. **スーパーバイザー再決定**：ドキュメント作成タスクを分配
9. **ライター実行**：使用ドキュメントを作成
10. **スーパーバイザーへ戻る**：ドキュメントを提出
11. **スーパーバイザー完了判断**：すべての結果を統合し、最終成果物を出力

---

## 高度な機能と最適化

### 1. メモリ機能の追加

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# チェックポイントセーバーを作成
memory = SqliteSaver.from_conn_string(":memory:")

# コンパイル時にチェックポイントを追加
app = workflow.compile(checkpointer=memory)

# 実行時にスレッドIDを指定
config = {"configurable": {"thread_id": "conversation_1"}}
result = app.invoke(initial_state, config=config)
```

### 2. エージェントの並列実行

```python
from langgraph.graph import START

def create_parallel_graph():
    """並列実行をサポートするグラフを作成"""
    workflow = StateGraph(AgentState)
    
    # ノードを追加
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("aggregator", aggregator_node)
    
    # supervisorから複数のエージェントに並列分发信
    workflow.add_edge("supervisor", "researcher")
    workflow.add_edge("supervisor", "coder")
    
    # 複数のエージェント完了後aggregatorに集約
    workflow.add_edge("researcher", "aggregator")
    workflow.add_edge("coder", "aggregator")
    
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()
```

### 3. 人間によるレビューノード

```python
from langgraph.checkpoint.sqlite import SqliteSaver

def human_review_node(state: AgentState) -> AgentState:
    """人間によるレビューノード"""
    
    # 実行を一時停止し、人間の入力を待機
    print("\n🔍 人間によるレビューが必要です：")
    print(f"現在の結果：{state['results']}")
    
    approval = input("\n続行を承認しますか？(yes/no): ")
    
    if approval.lower() == "yes":
        return {
            "messages": [HumanMessage(content="人間によるレビュー承認済み")],
            "next_agent": "continue"
        }
    else:
        feedback = input("修正意見を入力してください: ")
        return {
            "messages": [HumanMessage(content=f"修正が必要: {feedback}")],
            "next_agent": "revise"
        }

# グラフに人間によるレビューノードを追加
workflow.add_node("human_review", human_review_node)
workflow.add_edge("coder", "human_review")
```

### 4. エラー処理とリトライ

```python
def safe_agent_wrapper(agent_func, max_retries=3):
    """エージェントにエラー処理とリトライ機構を追加"""
    
    def wrapped_agent(state: AgentState) -> AgentState:
        for attempt in range(max_retries):
            try:
                return agent_func(state)
            except Exception as e:
                print(f"⚠️ エージェント実行失敗 (試行 {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    # 最後の試行も失敗した場合、エラー状態を返す
                    return {
                        "messages": [AIMessage(content=f"実行失敗: {str(e)}")],
                        "results": {
                            **state.get("results", {}),
                            "error": str(e)
                        },
                        "next_agent": "FINISH"
                    }
                
                # 待機後にリトライ
                import time
                time.sleep(2 ** attempt)
        
        return state
    
    return wrapped_agent

# ラッパーを使用
workflow.add_node("researcher", safe_agent_wrapper(researcher_node))
```

### 5. 動的なエージェント登録

```python
class AgentRegistry:
    """エージェントレジストリ"""
    
    def __init__(self):
        self.agents = {}
    
    def register(self, name: str, agent_func, description: str):
        """新しいエージェントを登録"""
        self.agents[name] = {
            "function": agent_func,
            "description": description
        }
    
    def get_agent(self, name: str):
        """エージェントを取得"""
        return self.agents.get(name, {}).get("function")
    
    def list_agents(self) -> str:
        """すべての利用可能なエージェントを一覧表示"""
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.agents.items()
        ])

# 使用例
registry = AgentRegistry()
registry.register("researcher", researcher_node, "情報検索とデータ分析")
registry.register("coder", coder_node, "コーディングと技術実装")
registry.register("writer", writer_node, "コンテンツ作成とドキュメント作成")

# スーパーバイザーで使用
def dynamic_supervisor(state: AgentState, registry: AgentRegistry):
    """動的エージェントをサポートするスーパーバイザー"""
    
    available_agents = registry.list_agents()
    
    prompt = f"""
    利用可能なエージェント：
    {available_agents}
    
    タスク：{state['task']}
    
    適切なエージェントを選択してください。
    """
    
    # ... 決定ロジック
```

### 6. ストリーミング出力

```python
async def stream_multi_agent_system(task: str):
    """ストリーミング出力をサポートするマルチエージェントシステム"""
    
    app = create_multi_agent_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "results": {},
        "executed_agents": []
    }
    
    # ストリーミング実行
    async for event in app.astream(initial_state):
        for node_name, node_state in event.items():
            print(f"\n📍 ノード: {node_name}")
            
            if "messages" in node_state:
                latest_message = node_state["messages"][-1]
                print(f"💬 {latest_message.content}")
            
            # リアルタイムで進捗を表示
            if "executed_agents" in node_state:
                print(f"✅ 完了: {', '.join(node_state['executed_agents'])}")

# ストリーミングシステムを実行
import asyncio
asyncio.run(stream_multi_agent_system("Webアプリケーションを作成"))
```

---

## よくある問題とベストプラクティス

### よくある問題

#### Q1: スーパーバイザーはどのように無限ループを避けることができますか？

**解決策**：
```python
class AgentState(TypedDict):
    # ... 他のフィールド
    iteration: int
    max_iterations: int

def supervisor_node(state: AgentState) -> AgentState:
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)
    
    if iteration >= max_iter:
        return {
            "messages": [AIMessage(content="最大反復回数に達しました")],
            "next_agent": "FINISH"
        }
    
    # ... 通常の決定ロジック
    
    return {
        "iteration": iteration + 1,
        # ... 他の戻り値
    }
```

#### Q2: エージェント間の依存関係をどのように処理しますか？

**解決策**：
```python
def supervisor_node(state: AgentState) -> AgentState:
    executed = state.get("executed_agents", [])
    results = state.get("results", {})
    
    # 依存関係を定義
    dependencies = {
        "coder": ["researcher"],  # coderはresearcherに依存
        "writer": ["researcher", "coder"]  # writerは両方に依存
    }
    
    # 次のエージェントを選択
    for agent, deps in dependencies.items():
        if agent not in executed:
            # 依存関係が満たされているか確認
            if all(dep in executed for dep in deps):
                return {"next_agent": agent}
    
    return {"next_agent": "FINISH"}
```

#### Q3: LLM呼び出しコストをどのように最適化しますか？

**解決策**：
1. **重复クエリのキャッシュ**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_llm_call(prompt: str) -> str:
    return llm.invoke(prompt).content
```

2. **異なるモデルを使用**
```python
# スーパーバイザーは強力なモデルを使用
supervisor_llm = ChatOpenAI(model="gpt-4")

# シンプルなエージェントは弱いモデルを使用
agent_llm = ChatOpenAI(model="gpt-3.5-turbo")
```

3. **バッチ処理**
```python
def batch_process_agents(tasks: list) -> list:
    """複数のタスクをバッチ処理"""
    prompts = [create_prompt(task) for task in tasks]
    responses = llm.batch(prompts)
    return responses
```

### ベストプラクティス

#### 1. 明確な状態設計

```python
class AgentState(TypedDict):
    # コアフィールド
    task: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # 制御フローフィールド
    next_agent: str
    iteration: int
    
    # データフィールド
    results: dict
    intermediate_steps: list
    
    # メタデータフィールド
    executed_agents: list
    start_time: float
    metadata: dict
```

#### 2. モジュール化されたエージェント設計

```python
class BaseAgent:
    """エージェント基底クラス"""
    
    def __init__(self, name: str, llm, tools: list = None):
        self.name = name
        self.llm = llm
        self.tools = tools or []
    
    def execute(self, state: AgentState) -> AgentState:
        """エージェントタスクを実行"""
        raise NotImplementedError
    
    def create_prompt(self, state: AgentState) -> str:
        """プロンプトを作成"""
        raise NotImplementedError

class ResearcherAgent(BaseAgent):
    """リサーチャーエージェント"""
    
    def create_prompt(self, state: AgentState) -> str:
        return f"研究タスク: {state['task']}"
    
    def execute(self, state: AgentState) -> AgentState:
        prompt = self.create_prompt(state)
        result = self.llm.invoke(prompt).content
        
        return {
            "results": {**state["results"], self.name: result},
            "executed_agents": state["executed_agents"] + [self.name]
        }
```

#### 3. 適切なログ記録

```python
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_agent(agent_func):
    """エージェントにログデコレータを追加"""
    
    def wrapper(state: AgentState) -> AgentState:
        agent_name = agent_func.__name__
        
        logger.info(f"[{datetime.now()}] {agent_name} 実行開始")
        logger.debug(f"入力状態: {state}")
        
        try:
            result = agent_func(state)
            logger.info(f"[{datetime.now()}] {agent_name} 実行成功")
            logger.debug(f"出力状態: {result}")
            return result
        except Exception as e:
            logger.error(f"[{datetime.now()}] {agent_name} 実行失敗: {e}")
            raise
    
    return wrapper

@logged_agent
def researcher_node(state: AgentState) -> AgentState:
    # ... 実装
    pass
```

#### 4. テスト戦略

```python
import unittest

class TestMultiAgentSystem(unittest.TestCase):
    
    def setUp(self):
        """テスト前準備"""
        self.app = create_multi_agent_graph()
    
    def test_simple_task(self):
        """シンプルタスクをテスト"""
        state = {
            "task": "Pythonベストプラクティスを研究",
            "messages": [],
            "results": {},
            "executed_agents": []
        }
        
        result = self.app.invoke(state)
        
        # 結果を検証
        self.assertIn("researcher", result["executed_agents"])
        self.assertIn("researcher", result["results"])
    
    def test_complex_workflow(self):
        """複雑なワークフローをテスト"""
        state = {
            "task": "Webアプリケーションを作成してドキュメントを書く",
            "messages": [],
            "results": {},
            "executed_agents": []
        }
        
        result = self.app.invoke(state)
        
        # 必要なすべてのエージェントが実行されたことを検証
        expected_agents = ["researcher", "coder", "writer"]
        for agent in expected_agents:
            self.assertIn(agent, result["executed_agents"])
    
    def test_error_handling(self):
        """エラー処理をテスト"""
        # エラーシナリオをシミュレート
        pass

if __name__ == "__main__":
    unittest.main()
```

#### 5. パフォーマンス監視

```python
import time
from functools import wraps

class PerformanceMonitor:
    """パフォーマンスモニター"""
    
    def __init__(self):
        self.metrics = {}
    
    def track(self, agent_name: str):
        """エージェントパフォーマンスを追跡するデコレータ"""
        
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
        """パフォーマンスレポートを生成"""
        report = "パフォーマンスレポート\n" + "=" * 50 + "\n"
        
        for agent, metrics in self.metrics.items():
            durations = [m["duration"] for m in metrics]
            avg_duration = sum(durations) / len(durations)
            
            report += f"\n{agent}:\n"
            report += f"  呼び出し回数: {len(metrics)}\n"
            report += f"  平均所要時間: {avg_duration:.2f}s\n"
            report += f"  合計所要時間: {sum(durations):.2f}s\n"
        
        return report

# 使用例
monitor = PerformanceMonitor()

@monitor.track("researcher")
def researcher_node(state: AgentState) -> AgentState:
    # ... 実装
    pass

# 実行後にレポートを確認
print(monitor.get_report())
```

---

## まとめ

このチュートリアルでは、LangGraphを使用してマルチエージェント + スーパーバイザーシステムを構築する方法を詳しく紹介しました：

### コアポイント

1. **状態管理**：TypedDictを使用して明確な状態構造を定義
2. **ノード設計**：各エージェントは独立したノードで、職責が明確
3. **スーパーバイザーモード**：中央調整者がタスク分配と結果統合を担当
4. **条件付きルーティング**：状態に 따라実行フローを動的に決定
5. **拡張性**：新しいエージェントと機能を 쉽게追加可能

### 適用シナリオ

- 複雑な研究与分析タスク
- 複数ステップのコンテンツ作成
- 多种多様な専門スキルが必要なプロジェクト
- エンタープライズグレードの自動化ワークフロー

### 进阶方向

- 外部ツール（検索、データベース、API）との統合
- より複雑な協調パターン（階層型、メッシュ型）の実装
- 人間とエージェントの協調機能の追加
- パフォーマンスとコストの最適化
- 本番環境へのデプロイ

---

## 参考リソース

- [LangGraph 公式ドキュメント](https://langchain-ai.github.io/langgraph/)
- [LangChain ドキュメント](https://python.langchain.com/)
- [マルチエージェントシステム設計パターン](https://arxiv.org/abs/2308.10848)
- [LangGraph GitHub リポジトリ](https://github.com/langchain-ai/langgraph)

---

**ドキュメントバージョン**: 1.0  
**最終更新**: 2026年2月  
**著者**: AI アシスタント  
**ライセンス**: MIT License