# Agent 架构学习路线：概念、代码任务与 Codex Prompts

> 适合背景：已了解 LLM 基础，做过 RAG，试过 7B 左右微调；对更高级 agent 架构有概念但缺少动手经验；当前使用 Codex / vibe coding 辅助开发。  
> 核心目标：从手写最小 agent loop 开始，逐步掌握 tool calling、state、guardrails、agentic RAG、coding agent、planner/verifier、multi-agent handoff 与 eval。

---

## 0. 总原则

### 0.1 学习顺序

不要一开始就上 LangGraph / CrewAI / AutoGen / multi-agent demo。推荐顺序：

```text
手写 single-agent loop
→ tool registry
→ trace / state / max_steps
→ agentic RAG
→ verifier / evaluator
→ coding agent mini
→ planner-executor
→ router / handoff / multi-agent
→ 框架化重构
```

### 0.2 每个阶段都要掌握的 5 个问题

每完成一个小项目，都问自己：

```text
1. LLM 在这个系统里负责什么？
2. Runtime / harness 负责什么？
3. Tool 的输入输出 schema 是否明确？
4. Agent 如何知道任务完成了？
5. 失败时如何观测、复现和修复？
```

### 0.3 Codex 使用原则

把 Codex 当成“结对工程师”，不要让它一次性生成一个巨大黑盒项目。每次只给它一个小目标，并强制它输出：

```text
1. 修改计划
2. 文件列表
3. 关键设计取舍
4. 代码实现
5. 测试
6. 如何运行
7. 已知限制
```

---

## 1. 推荐仓库结构

让 Codex 先创建这个结构：

```text
agent-architecture-lab/
  README.md
  pyproject.toml
  src/
    agent_lab/
      __init__.py

      core/
        messages.py
        tools.py
        runtime.py
        trace.py
        state.py
        policy.py

      llm/
        base.py
        fake_llm.py
        openai_llm.py

      rag/
        corpus.py
        retriever.py
        agentic_rag.py
        verifier.py

      coding_agent/
        workspace.py
        file_tools.py
        shell_tools.py
        patch_tools.py
        coding_loop.py

      orchestration/
        planner.py
        executor.py
        router.py
        critic.py
        handoff.py

      evals/
        tasks.py
        runner.py
        metrics.py

  tests/
    test_tools.py
    test_runtime.py
    test_agent_loop.py
    test_agentic_rag.py
    test_coding_agent.py
    test_orchestration.py

  examples/
    01_minimal_agent.py
    02_agentic_rag.py
    03_coding_agent.py
    04_planner_verifier.py
    05_router_handoff.py

  docs/
    concepts.md
    trace_examples.md
    design_notes.md
```

---

## 2. 基础 Prompt 模板

后面每个任务都可以套这个模板。

```text
你是我的 agent 架构学习搭档和代码审查员。

背景：
- 我懂 LLM 基础，做过 RAG，试过 7B 左右微调。
- 我现在要学习 agent 架构。
- 我希望先手写核心机制，不要一开始引入高层 agent 框架。

当前任务：
<填入本次任务>

要求：
1. 先解释这个任务对应的 agent 架构概念。
2. 给出实现计划，列出要新增/修改的文件。
3. 代码要简单、可读、可测试，不要过度抽象。
4. 必须包含单元测试。
5. 必须包含一个 examples/ 下的可运行示例。
6. 关键步骤要打印 trace，方便我观察 agent loop。
7. 所有工具调用要有 schema、输入校验、错误返回。
8. 不要默认执行危险 shell 命令。
9. 遇到设计选择时，先给出取舍，再实现推荐方案。
10. 完成后告诉我如何运行测试和示例。
```

---

# Week 1：手写最小 Agent Loop

## 目标

掌握 agent 的最小闭环：

```text
User goal
→ LLM decides next action
→ Runtime validates tool call
→ Tool executes
→ Observation returned to LLM
→ LLM continues or finalizes
```

## 必须掌握

```text
- Message 格式
- Tool schema
- Tool registry
- Tool call
- Observation
- Agent runtime / harness
- max_steps 防死循环
- trace logging
- fake LLM 测试
```

---

## 1.1 任务：实现最小 tool registry

### 要 Codex 写什么

```text
src/agent_lab/core/tools.py
tests/test_tools.py
```

### 要掌握的概念

```text
Tool 不是随便一个 Python 函数。
一个可靠 tool 至少需要：
- name
- description
- input_schema
- callable
- timeout 或执行限制
- 错误处理
- 返回结构
```

### Codex Prompt

```text
请实现 agent_lab.core.tools 模块。

目标：
实现一个最小 Tool 和 ToolRegistry。

要求：
1. 定义 Tool dataclass：
   - name: str
   - description: str
   - input_schema: dict
   - fn: Callable
2. Tool.call(args: dict) 返回 ToolResult。
3. ToolResult 包含：
   - ok: bool
   - output: Any | None
   - error: str | None
4. ToolRegistry 支持：
   - register(tool)
   - get(name)
   - list_tools()
   - call(name, args)
5. call 不存在的 tool 时返回结构化错误，不要抛出未捕获异常。
6. fn 抛异常时要捕获并放入 ToolResult.error。
7. 写 tests/test_tools.py，覆盖：
   - 正常调用
   - 未知工具
   - 工具内部异常
   - list_tools 返回 schema 信息
8. 不要引入高层 agent 框架。
```

### 验收标准

```text
pytest tests/test_tools.py -q
```

应该通过。

你能回答：

```text
ToolRegistry 和直接调用 Python 函数有什么区别？
为什么 tool 返回结构化结果比直接返回字符串更好？
```

---

## 1.2 任务：实现 message 和 trace

### 要 Codex 写什么

```text
src/agent_lab/core/messages.py
src/agent_lab/core/trace.py
tests/test_runtime.py
```

### Codex Prompt

```text
请实现 agent 的消息和 trace 基础设施。

要求：
1. 在 src/agent_lab/core/messages.py 中定义消息类型：
   - UserMessage
   - AssistantMessage
   - ToolCallMessage
   - ToolObservationMessage
   - FinalMessage
2. 可以用 dataclass。
3. 每条消息至少包含：
   - role/type
   - content
   - timestamp
4. ToolCallMessage 需要包含：
   - tool_name
   - args
5. ToolObservationMessage 需要包含：
   - tool_name
   - ok
   - output
   - error
6. 在 src/agent_lab/core/trace.py 中实现 TraceRecorder：
   - record(message)
   - as_text()
   - as_json()
7. trace 输出要方便人类阅读 agent 每一步做了什么。
8. 写测试覆盖消息序列和 trace 输出。
```

### 验收标准

你能打印出类似：

```text
[USER] calculate 2 + 3
[ASSISTANT_TOOL_CALL] calculator {"expression": "2 + 3"}
[TOOL_OBSERVATION] calculator ok=True output=5
[FINAL] 2 + 3 = 5
```

---

## 1.3 任务：实现 FakeLLM 和最小 Runtime

### 要 Codex 写什么

```text
src/agent_lab/llm/base.py
src/agent_lab/llm/fake_llm.py
src/agent_lab/core/runtime.py
examples/01_minimal_agent.py
tests/test_agent_loop.py
```

### 要掌握的概念

这里的 LLM 可以先用 fake，不接真实 API。重点不是模型能力，而是 agent loop 的结构。

### Codex Prompt

```text
请实现一个不依赖真实 LLM API 的最小 agent runtime。

目标：
让我理解 agent loop，而不是追求智能。

要求：
1. 定义 LLMClient 抽象接口：
   - next(messages, tools) -> AssistantDecision
2. AssistantDecision 支持两类：
   - tool_call: tool_name + args
   - final: content
3. 实现 FakeLLM：
   - 初始化时传入一个 decisions 列表
   - 每次调用 next 返回下一个 decision
4. 实现 AgentRuntime：
   - 接收 llm、tool_registry、max_steps、trace_recorder
   - run(user_input) 执行 loop
   - 如果 LLM 返回 tool_call，则调用工具并把 observation 加回 messages
   - 如果 LLM 返回 final，则结束并返回 final 内容
   - 达到 max_steps 后返回错误
5. 实现 calculator tool：
   - 为了安全，不要使用 eval
   - 只支持简单四则运算，或用 ast 白名单解析
6. examples/01_minimal_agent.py 展示：
   - 用户问 2 + 3
   - FakeLLM 先调用 calculator
   - 再给 final answer
   - 打印 trace
7. 写 tests/test_agent_loop.py，覆盖：
   - 正常 tool call
   - tool error
   - max_steps
```

### 验收标准

运行：

```bash
python examples/01_minimal_agent.py
pytest tests/test_agent_loop.py -q
```

你应该能清楚看到：

```text
LLM 没有直接运行工具。
Runtime 根据 LLM 的结构化 tool_call 执行工具。
工具结果变成 observation 后再喂给 LLM。
```

---

## 1.4 任务：接入真实 OpenAI LLM，但保持可替换接口

### 要 Codex 写什么

```text
src/agent_lab/llm/openai_llm.py
examples/01b_real_llm_agent.py
```

### Codex Prompt

```text
请在不破坏现有 FakeLLM 测试的前提下，添加一个 OpenAI LLM adapter。

要求：
1. 新增 src/agent_lab/llm/openai_llm.py。
2. 实现 OpenAILLMClient，遵守 LLMClient 接口。
3. 将 ToolRegistry 中的 tools 转成 OpenAI tool/function schema。
4. 模型返回 tool call 时，转换成 AssistantDecision(tool_call)。
5. 模型返回普通文本时，转换成 AssistantDecision(final)。
6. 不要把 API key 写死在代码里，从环境变量读取。
7. examples/01b_real_llm_agent.py 复用 calculator tool。
8. README 中补充运行方式。
9. 真实 API 的测试不要默认运行；单元测试继续用 FakeLLM。
```

### 验收标准

你能说明：

```text
FakeLLM 为什么对 agent 架构测试很重要？
真实 LLM adapter 为什么应该和 Runtime 解耦？
```

---

# Week 2：Agentic RAG

## 目标

从普通 RAG 升级到 agentic RAG：

```text
普通 RAG：
固定检索一次 → 生成答案

Agentic RAG：
LLM 判断是否检索
→ 选择 query
→ 判断证据是否足够
→ 必要时改写 query 再检索
→ 生成答案
→ verifier 检查 grounding
```

## 必须掌握

```text
- retriever as tool
- query rewriting
- multi-hop retrieval
- evidence object
- grounded answer
- verifier
- retry loop
```

---

## 2.1 任务：实现 Toy Corpus 和 Retriever Tool

### 要 Codex 写什么

```text
src/agent_lab/rag/corpus.py
src/agent_lab/rag/retriever.py
examples/02_agentic_rag.py
tests/test_agentic_rag.py
```

### Codex Prompt

```text
请实现一个最小 agentic RAG 的检索工具。

要求：
1. 新增 Document dataclass：
   - id
   - title
   - text
   - metadata
2. 新增 InMemoryCorpus：
   - add(document)
   - list()
3. 新增 SimpleKeywordRetriever：
   - retrieve(query, top_k=3)
   - 可以用简单 keyword overlap，不需要向量数据库
4. 把 retriever 包装成 Tool：
   - tool name: retrieve_docs
   - args: {"query": str, "top_k": int}
   - output: list of evidence objects
5. evidence object 包含：
   - doc_id
   - title
   - snippet
   - score
6. 写一个 example：
   - 构造 5-10 篇 toy docs
   - agent 可以调用 retrieve_docs
   - 打印 trace
7. 写测试覆盖：
   - 正常检索
   - top_k
   - 无匹配
   - retriever tool schema
```

### 验收标准

你能回答：

```text
为什么 retriever 也可以被看作一个 tool？
普通 RAG 和 agentic RAG 的控制权差异在哪里？
```

---

## 2.2 任务：让 agent 自己决定是否检索、检索几次

### Codex Prompt

```text
请把现有最小 agent loop 升级为 agentic RAG example。

目标：
用户问一个问题后，LLM 可以：
1. 直接回答；
2. 调用 retrieve_docs；
3. 根据 observation 改写 query 再检索；
4. 最终基于 evidence 回答。

要求：
1. 不要写死“一定先检索”。
2. examples/02_agentic_rag.py 中使用 FakeLLM 构造一个多步流程：
   - 第一次 query 太宽泛
   - observation 信息不足
   - 第二次改写 query
   - 最终回答
3. trace 中必须能看到两次 retrieve_docs 的 query 不同。
4. 最终答案中必须包含引用的 doc_id。
5. 写测试验证：
   - agent 可以多次调用 retriever
   - final answer 中包含 evidence 引用
   - 达到 max_steps 会停止
```

### 验收标准

trace 应该类似：

```text
[USER] What is agent runtime?
[TOOL_CALL] retrieve_docs {"query": "agent", "top_k": 3}
[OBSERVATION] docs not specific enough
[TOOL_CALL] retrieve_docs {"query": "agent runtime tool observation loop", "top_k": 3}
[OBSERVATION] found doc_3
[FINAL] Agent runtime 是... [doc_3]
```

---

## 2.3 任务：实现 Grounding Verifier

### 要掌握的概念

Verifier 不是为了“让答案更漂亮”，而是为了检查：

```text
答案中的关键 claim 是否被 evidence 支持？
引用是否真实存在？
是否编造了 evidence 中没有的信息？
```

### Codex Prompt

```text
请实现一个简单 grounding verifier。

要求：
1. 新增 src/agent_lab/rag/verifier.py。
2. 定义 VerificationResult：
   - ok: bool
   - unsupported_claims: list[str]
   - missing_citations: list[str]
   - notes: str
3. 实现 SimpleGroundingVerifier：
   - 输入 answer: str, evidence: list[Evidence]
   - 检查 answer 中的引用 doc_id 是否存在
   - 检查 answer 至少包含一个引用
   - 可以用简单规则，不需要 LLM
4. 把 verifier 包装成 Tool：
   - tool name: verify_answer
5. 在 examples/02_agentic_rag.py 中演示：
   - 先生成一个缺引用答案
   - verifier 失败
   - agent 修复答案
6. 写测试覆盖：
   - 无引用失败
   - 引用不存在失败
   - 正确引用通过
```

### 验收标准

你能解释：

```text
RAG eval 和 agentic RAG eval 的区别是什么？
为什么 verifier 应该返回结构化结果而不是自然语言？
```

---

# Week 3：Coding Agent Mini

## 目标

做一个小型 coding agent，理解 Codex 类产品背后的核心动作：

```text
read/search/edit/run tests/observe/fix/retry
```

## 必须掌握

```text
- workspace boundary
- read_file / write_file
- search_code
- apply_patch
- run_shell
- command allowlist
- timeout
- stdout/stderr/exit_code
- test failure repair loop
- diff review
```

---

## 3.1 任务：实现受限 Workspace 和文件工具

### Codex Prompt

```text
请实现 coding agent 的 workspace 和文件工具。

要求：
1. 新增 src/agent_lab/coding_agent/workspace.py。
2. Workspace 只能访问指定 root 目录内的文件。
3. 防止 path traversal，例如 ../../secret.txt。
4. 实现工具：
   - list_files(path)
   - read_file(path)
   - write_file(path, content)
   - search_code(query)
5. search_code 可以先用简单 substring search。
6. 所有工具都包装成 ToolRegistry 的 Tool。
7. 写测试覆盖：
   - 正常读写
   - 不能访问 root 外文件
   - 搜索文件内容
   - 未找到时返回结构化空结果
```

### 验收标准

你能回答：

```text
为什么 coding agent 必须有 workspace boundary？
为什么不能直接把本机整个文件系统暴露给 agent？
```

---

## 3.2 任务：实现安全 run_shell

### Codex Prompt

```text
请实现 coding agent 的 run_shell 工具。

要求：
1. 新增 src/agent_lab/coding_agent/shell_tools.py。
2. run_shell 必须在 Workspace.root 下执行。
3. 返回：
   - stdout
   - stderr
   - exit_code
   - timed_out
4. 支持 timeout。
5. 默认不允许危险命令。
6. 实现一个简单 allowlist：
   - python
   - pytest
   - pip
   - ls
   - cat
7. 拒绝明显危险命令：
   - rm -rf
   - curl ... | bash
   - sudo
   - chmod 777
   - 访问 ~/.ssh
8. 不要使用 shell=True，除非你同时实现了严格校验；推荐使用 shlex.split + subprocess.run。
9. 写测试覆盖：
   - 正常 pytest
   - timeout
   - denylist
   - 非 allowlist 命令
   - 工作目录限制
```

### 验收标准

你能解释：

```text
LLM 请求 run_shell 和 runtime 真正执行 shell 之间为什么必须有 policy layer？
stdout/stderr/exit_code 分别有什么用？
```

---

## 3.3 任务：实现 apply_patch 和 diff 输出

### Codex Prompt

```text
请实现 apply_patch 工具和 diff 输出。

要求：
1. 新增 src/agent_lab/coding_agent/patch_tools.py。
2. 支持对 Workspace 内文件应用简单 patch。
3. 不要求完整 git apply，但至少支持：
   - 替换文件中的一段旧文本为新文本
   - 如果旧文本不存在，返回结构化错误
4. 每次修改后可以输出 unified diff。
5. 写测试覆盖：
   - 成功替换
   - old_text 不存在
   - 不能修改 root 外文件
   - diff 内容正确
```

### 验收标准

你能解释：

```text
为什么 coding agent 最好使用 patch，而不是直接让 LLM 重写整个文件？
```

---

## 3.4 任务：实现 Mini Coding Agent Loop

### Codex Prompt

```text
请把文件工具、shell 工具、patch 工具组合成一个 mini coding agent。

场景：
在 tests/fixtures/buggy_project 下创建一个小 Python 项目：
- src/price.py 有一个 format_price 函数 bug
- tests/test_price.py 有失败测试

Agent 流程：
1. run_shell("pytest -q")
2. 根据 stderr/stdout 识别失败测试
3. read_file 相关测试文件
4. search_code 相关函数名
5. read_file 源文件
6. apply_patch 修 bug
7. 再 run_shell("pytest -q")
8. 如果通过，输出 final answer 和 diff

要求：
1. 使用 FakeLLM 写一个确定性流程，先不要依赖真实 LLM。
2. examples/03_coding_agent.py 能跑完整流程。
3. trace 清楚展示每一步。
4. tests/test_coding_agent.py 验证最终测试通过。
5. 保留 max_steps。
6. 如果测试失败，agent 不要假装成功。
```

### 验收标准

运行：

```bash
python examples/03_coding_agent.py
pytest tests/test_coding_agent.py -q
```

你要看到：

```text
测试失败 → 读文件 → 修改 → 测试通过 → 输出 diff
```

---

## 3.5 任务：接入真实 LLM 做代码修复

### Codex Prompt

```text
请在 mini coding agent 中接入 OpenAILLMClient，但保留 FakeLLM 测试。

要求：
1. examples/03b_real_coding_agent.py 使用真实模型。
2. system prompt 要明确：
   - 先检查再修改
   - 修改前先 read_file
   - 修改后必须 run tests
   - 不要运行危险命令
   - 不要在测试失败时声称成功
3. 所有 shell 命令仍然经过 allowlist/denylist。
4. 修改文件时优先 apply_patch，不要直接全量重写。
5. 输出最终：
   - 修改摘要
   - 测试命令
   - 测试结果
   - diff
6. README 说明如何运行。
```

### 验收标准

你能解释：

```text
Codex 类 coding agent 的核心 loop 是什么？
为什么即使模型很强，仍然需要 sandbox、trace、allowlist 和测试？
```

---

# Week 4：Orchestration、Verifier、Router、Handoff 与 Eval

## 目标

从单个 agent loop 升级到可组合 agentic workflow。

## 必须掌握

```text
- planner-executor
- evaluator-optimizer
- router
- specialist agent
- handoff
- task state
- eval dataset
- success metric
- trace-based debugging
```

---

## 4.1 任务：Planner-Executor

### 要掌握的概念

Planner 负责拆任务；Executor 负责执行具体步骤。

```text
Planner:
  input: user goal
  output: steps

Executor:
  input: one step
  output: step result
```

### Codex Prompt

```text
请实现一个最小 Planner-Executor 架构。

要求：
1. 新增 src/agent_lab/orchestration/planner.py。
2. Plan 包含：
   - goal
   - steps: list[PlanStep]
3. PlanStep 包含：
   - id
   - description
   - status: pending/running/done/failed
   - result
4. Planner 可以先用 FakeLLM 或规则实现。
5. Executor 调用现有 AgentRuntime 执行每个 step。
6. examples/04_planner_verifier.py 展示：
   - 用户提出一个需要两步检索的问题
   - planner 拆成两个子问题
   - executor 分别检索
   - synthesizer 汇总
7. 写测试覆盖：
   - plan 创建
   - step 状态更新
   - step 失败时整体失败
```

### 验收标准

你能解释：

```text
什么时候用 planner-executor 比单一 agent loop 更好？
Planner 的输出为什么要结构化？
```

---

## 4.2 任务：Evaluator-Optimizer / Critic

### Codex Prompt

```text
请实现一个 evaluator-optimizer 模式。

目标：
让 agent 先生成答案，再由 critic 检查，必要时返回修改建议。

要求：
1. 新增 src/agent_lab/orchestration/critic.py。
2. CriticResult 包含：
   - pass_: bool
   - issues: list[str]
   - suggestions: list[str]
3. 实现一个 SimpleCritic：
   - 检查答案是否包含引用
   - 检查是否包含 "I don't know" 但又给出确定结论
   - 检查是否包含 unsupported doc_id
4. Runtime 支持：
   - generate answer
   - critic evaluate
   - 如果失败，带 suggestions 重新生成
   - 最多 retry 2 次
5. 写 example 和测试。
```

### 验收标准

你能回答：

```text
Critic 和 Verifier 的区别是什么？
为什么 critic loop 必须有最大重试次数？
```

---

## 4.3 任务：Router + Specialist Agents

### 要掌握的概念

Router 根据任务类型把请求交给不同 specialist：

```text
Router
  ├── RAG Agent
  ├── Coding Agent
  └── Writing Agent
```

### Codex Prompt

```text
请实现一个最小 router + specialist agents 架构。

要求：
1. 新增 src/agent_lab/orchestration/router.py。
2. RouterDecision 包含：
   - target_agent
   - confidence
   - reason
3. 实现三个 specialist：
   - rag_agent: 回答知识库问题
   - coding_agent: 处理代码修复任务
   - writing_agent: 做简单文本改写
4. Router 可以先用规则：
   - 包含 "test", "bug", "fix", "pytest" → coding_agent
   - 包含 "doc", "according to", "knowledge" → rag_agent
   - 包含 "rewrite", "email", "tone" → writing_agent
5. examples/05_router_handoff.py 展示 3 个输入被路由到不同 agent。
6. trace 中要显示：
   - router decision
   - selected agent
   - specialist trace
7. 写测试覆盖路由逻辑。
```

### 验收标准

你能解释：

```text
Router 是 workflow 还是 agent？
什么时候 router 用规则比 LLM 更好？
```

---

## 4.4 任务：Handoff

### 要掌握的概念

Handoff 是 agent 把任务交给另一个 agent。不是简单函数调用，因为要传递：

```text
- 当前任务目标
- 已知上下文
- 已执行步骤
- 失败原因
- 接收方 agent 的职责边界
```

### Codex Prompt

```text
请实现一个最小 handoff 机制。

要求：
1. 新增 src/agent_lab/orchestration/handoff.py。
2. HandoffRequest 包含：
   - from_agent
   - to_agent
   - reason
   - context_summary
   - artifacts
3. HandoffResult 包含：
   - accepted
   - output
   - notes
4. 在 examples/05_router_handoff.py 中演示：
   - RAG agent 回答时发现问题其实需要改代码
   - handoff 给 coding_agent
5. trace 中要显示 handoff boundary。
6. 写测试覆盖：
   - handoff request 构造
   - target agent 接收
   - target agent 拒绝不属于自己职责的任务
```

### 验收标准

你能解释：

```text
handoff 和普通 tool call 的区别是什么？
handoff 时为什么需要 context_summary？
```

---

## 4.5 任务：Eval Runner

### Codex Prompt

```text
请实现 agent eval runner。

要求：
1. 新增 src/agent_lab/evals/tasks.py。
2. EvalTask 包含：
   - id
   - input
   - expected_behavior
   - success_criteria
   - tags
3. 新增 src/agent_lab/evals/runner.py。
4. EvalResult 包含：
   - task_id
   - success
   - final_answer
   - steps
   - tool_calls
   - cost_estimate 可先为空
   - failure_reason
5. 新增 src/agent_lab/evals/metrics.py。
6. 支持运行一组任务并输出 summary：
   - success_rate
   - avg_steps
   - tool_error_rate
   - max_steps_failures
7. 用 FakeLLM 构造 5 个 deterministic eval tasks。
8. 写 tests 覆盖 metrics。
```

### 验收标准

你能回答：

```text
为什么 agent eval 不能只看最终答案？
应该如何从 trace 中定位失败原因？
```

---

# 5. 进阶：学习 OpenAI Agents SDK / LangGraph / MCP

完成上面手写版本后，再学习框架。否则容易只会调 API，不懂内部机制。

## 5.1 OpenAI Agents SDK 对照表

你手写的模块和 SDK 概念可以这样对应：

```text
你的 AgentRuntime        ↔ Runner
你的 ToolRegistry        ↔ tools / function_tool
你的 TraceRecorder       ↔ tracing
你的 Guardrails/Policy   ↔ guardrails
你的 Router/Handoff      ↔ handoffs
你的 Message/State       ↔ sessions / context
你的 Workspace           ↔ sandbox workspace
```

### Codex Prompt

```text
请基于我已经手写的 agent-architecture-lab 项目，写一份 docs/sdk_mapping.md。

要求：
1. 把我们的模块和 OpenAI Agents SDK 概念逐一对应。
2. 解释哪些部分 SDK 已经帮我们做了。
3. 解释哪些部分仍然应该由应用层负责。
4. 不要重构代码，只写文档。
```

---

## 5.2 LangGraph 对照表

LangGraph 适合学习 state machine / graph orchestration。

### Codex Prompt

```text
请写一份 docs/langgraph_mapping.md。

目标：
把我们手写的 planner-executor、critic loop、router-handoff 映射到 graph/state machine 的视角。

要求：
1. 解释 node、edge、state、conditional edge 分别对应我们代码里的什么。
2. 给出一个伪代码 graph：
   - plan
   - retrieve
   - synthesize
   - verify
   - retry or final
3. 不要直接重构项目。
```

---

## 5.3 MCP 对照表

MCP 可以理解为 agent 使用外部工具和数据源的协议层。

### Codex Prompt

```text
请写一份 docs/mcp_mapping.md。

目标：
解释如果要把我们现有 tools 暴露成 MCP server，需要做哪些抽象。

要求：
1. 说明 ToolRegistry 和 MCP tools 的关系。
2. 说明 read_file/search_code/run_shell 哪些适合暴露，哪些需要权限控制。
3. 给出一个最小 MCP server 的伪代码。
4. 强调安全边界和 secret handling。
```

---

# 6. 你最终应该掌握的知识清单

## 6.1 Agent 核心

```text
[ ] Agent loop
[ ] Tool calling
[ ] Tool schema
[ ] Observation
[ ] Runtime / harness
[ ] State
[ ] Memory 类型
[ ] max_steps
[ ] Termination criteria
[ ] Error handling
[ ] Trace
```

## 6.2 RAG → Agentic RAG

```text
[ ] Retriever as tool
[ ] Query rewriting
[ ] Multi-hop retrieval
[ ] Evidence object
[ ] Citation grounding
[ ] Verifier
[ ] Retrieval retry
[ ] Answer synthesis
[ ] RAG eval vs agent eval
```

## 6.3 Coding Agent

```text
[ ] Workspace boundary
[ ] read_file
[ ] write_file
[ ] search_code
[ ] apply_patch
[ ] run_shell
[ ] stdout/stderr/exit_code
[ ] timeout
[ ] allowlist/denylist
[ ] test repair loop
[ ] diff review
[ ] sandbox
```

## 6.4 Orchestration

```text
[ ] Workflow vs agent
[ ] Planner-executor
[ ] Evaluator-optimizer
[ ] Router
[ ] Specialist agent
[ ] Handoff
[ ] Multi-agent boundary
[ ] Context summary
[ ] Task state
```

## 6.5 Safety / Guardrails

```text
[ ] Tool input validation
[ ] Output validation
[ ] Human approval
[ ] Dangerous command blocking
[ ] Secret masking
[ ] Path traversal prevention
[ ] Cost limit
[ ] Step limit
[ ] Permission boundary
[ ] Audit log
```

## 6.6 Evaluation

```text
[ ] Golden tasks
[ ] Task success rate
[ ] Tool call accuracy
[ ] Tool error rate
[ ] Unsupported claims
[ ] Grounding accuracy
[ ] Avg steps
[ ] Latency
[ ] Cost estimate
[ ] Regression tests
```

---

# 7. 学习节奏建议

## 第 1 周

```text
Day 1: ToolRegistry
Day 2: Messages + Trace
Day 3: FakeLLM + Runtime
Day 4: calculator tool + example
Day 5: max_steps + tool error handling
Day 6: OpenAI LLM adapter
Day 7: 写 docs/concepts.md 总结 agent loop
```

## 第 2 周

```text
Day 1: Corpus + retriever
Day 2: retriever tool
Day 3: multi-step retrieval
Day 4: evidence citation
Day 5: verifier
Day 6: retry after verifier failure
Day 7: 写 docs/agentic_rag_notes.md
```

## 第 3 周

```text
Day 1: Workspace
Day 2: file tools
Day 3: safe shell tool
Day 4: apply_patch
Day 5: deterministic coding agent with FakeLLM
Day 6: real LLM coding agent
Day 7: 写 docs/coding_agent_notes.md
```

## 第 4 周

```text
Day 1: Planner
Day 2: Executor
Day 3: Critic
Day 4: Router
Day 5: Handoff
Day 6: Eval runner
Day 7: 写 docs/final_architecture.md
```

---

# 8. 每次让 Codex 改代码后的自检 Prompt

把这个 prompt 作为每次修改后的 code review：

```text
请审查你刚刚的修改。

重点检查：
1. 是否破坏了已有测试？
2. 是否有未处理异常？
3. 是否有不安全的 shell / path / file 操作？
4. 是否有过度抽象？
5. 是否有隐藏状态导致测试不稳定？
6. trace 是否足够清楚？
7. tool 返回是否结构化？
8. max_steps / timeout 是否生效？
9. 是否遗漏 README 或 example？
10. 是否有更简单的实现？

请按：
- 问题
- 风险
- 建议修改
- 是否需要立即修复
输出。
```

---

# 9. Debug Prompt 模板

当代码跑不起来时，用这个：

```text
当前项目运行失败。请不要直接大改。

失败命令：
<粘贴命令>

错误输出：
<粘贴 stderr/stdout>

要求：
1. 先定位最可能的 3 个原因。
2. 说明你要检查哪些文件。
3. 只做最小修复。
4. 修复后补充或更新测试。
5. 再给出运行命令。
```

---

# 10. 让 Codex 解释代码的 Prompt

```text
请用 agent 架构视角解释这部分代码。

请按以下结构：
1. 这个文件在 agent 系统中的职责
2. 它属于 LLM、runtime、tool、state、policy、eval 中哪一层
3. 关键类/函数说明
4. 数据流：输入是什么，输出是什么
5. 失败模式
6. 如何测试
7. 和真实 Codex / coding agent 的对应关系
```

---

# 11. 最终 Capstone 项目

完成 4 周后，做一个综合项目：

```text
Agent Debugger
```

功能：

```text
输入：
- 一个小 repo
- 一个失败测试命令
- 用户目标

Agent 能够：
1. 运行测试
2. 读取失败信息
3. 搜索相关代码
4. 修改代码
5. 重新运行测试
6. 如果失败，进入 repair loop
7. 如果成功，输出 diff 和解释
8. verifier 检查是否真的有测试通过证据
9. eval runner 批量跑 10 个 bug fixture
```

### Capstone Prompt

```text
请基于当前 agent-architecture-lab 项目，实现一个 capstone：Agent Debugger。

要求：
1. 使用已有模块，不要重写整套架构。
2. 输入：
   - workspace path
   - test command
   - user goal
3. 流程：
   - run tests
   - parse failure
   - inspect files
   - propose patch
   - apply patch
   - rerun tests
   - verify success
   - output final report
4. 必须保留：
   - trace
   - max_steps
   - shell allowlist
   - workspace boundary
   - diff
5. 添加 3 个 fixture bug projects。
6. 添加 eval runner，一次跑 3 个 bug。
7. README 写清楚如何运行。
8. 不要声称成功，除非测试 exit_code == 0。
```

### Capstone 验收标准

```text
pytest -q
python examples/capstone_agent_debugger.py
python -m agent_lab.evals.runner
```

最终报告应包含：

```text
- 原始失败测试
- 修改文件
- diff
- 最终测试命令
- stdout/stderr 摘要
- exit_code
- 是否成功
- trace 路径
```

---

# 12. 参考资料

这些资料用于把你的手写实现和真实 agent 平台概念对齐：

```text
OpenAI Codex:
- https://openai.com/index/introducing-codex/
- https://developers.openai.com/codex/app/features
- https://openai.com/index/unrolling-the-codex-agent-loop/

OpenAI Agents SDK:
- https://openai.github.io/openai-agents-python/agents/
- https://openai.github.io/openai-agents-python/quickstart/
- https://openai.github.io/openai-agents-python/guardrails/
- https://openai.github.io/openai-agents-python/handoffs/
- https://openai.github.io/openai-agents-python/results/
```

---

# 13. 最重要的一句话

Agent 架构的核心不是“让 LLM 更聪明”，而是把 LLM 放进一个：

```text
可观察
可控制
可恢复
可评估
有权限边界
能调用工具
能根据结果继续决策
```

的执行系统里。

你这套学习路线的最终目标不是“会用一个 agent 框架”，而是能清楚地区分：

```text
LLM 负责决策
Runtime 负责执行
Tools 负责连接环境
State 负责保存进展
Policy 负责安全边界
Trace 负责可观测性
Eval 负责判断系统是否真的有效
```
