# AGENTS.md

## 0. 总原则

### 0.1 学习顺序

不要一开始就上 LangGraph / CrewAI / AutoGen / multi-agent demo。推荐顺序：

```text
手写 single-agent loop
-> tool registry
-> trace / state / max_steps
-> agentic RAG
-> verifier / evaluator
-> coding agent mini
-> planner-executor
-> router / handoff / multi-agent
-> 框架化重构
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

## 2. 基础 Prompt 模板

后面每个任务都应用这个模板。

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
