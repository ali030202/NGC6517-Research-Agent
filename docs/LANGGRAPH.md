# LangGraph 科研工作流

## 1 目标与范围

LangGraph 管理科研实验的控制流程，现有 `Agent` 服务管理训练、模型提案与知识工具。每轮实验的选择、提交、等待、分析和继续判断都在状态图节点中执行。旧 `run-plan` 保留原有预算循环，以兼容已有调用方。

默认图不调用 LLM，研究目标用于记录与检索，实验候选由本地规则生成。假设生成、代码生成、BFTS、论文草稿与审稿模块仍是可选扩展，未自动串入这张图。节点表示工作职责，不表示部署了多个独立语言模型。

## 2 节点与角色

| 节点 | 角色 | 行为 | 输出 |
|---|---|---|---|
| `inspect_data` | 数据检查员 | 检查训练/验证清单、文件、标签、类别计数及划分交集 | 数据状态与 CSV 指纹 |
| `wait_for_data` | 数据准备协调员 | 调用 `interrupt()` 持久暂停，不启动训练 | 同一 ID 恢复后重新检查 |
| `retrieve_knowledge` | 资料检索员 | 按目标检索本地片段，保存来源 ID | 参考资料；不会自动改变候选 |
| `plan_experiments` | 实验规划员 | 制定基线、门控消融、Dropout、学习率及种子计划 | 有限候选与共享截止时间 |
| `select_trial` | 预算协调员 | 查看取消、超时、已完成实验和最佳验证结果 | 下一轮或报告 |
| `submit_training` | 实验执行员 | 核对输入，以稳定请求键提交独立训练 | 持久任务 ID |
| `await_training` | 任务监控员 | 等待训练，监控取消和总预算，检测进程退出 | 终态任务 |
| `analyze_result` | 结果分析员 | 保留成功/失败记录，仅用真实验证结果选择最佳任务 | 更新记录并继续判断 |
| `write_report` | 报告整理员 | 汇总任务、指标、知识来源和停止原因 | Markdown 报告与终态 |

图定义位于 `ngc_agent/workflow.py`。`graph.invoke(..., durability="sync")` 在节点间同步保存检查点，每个工作流固定使用 `thread_id=workflow_id`。

## 3 状态与存储

图状态保存目标、配置、数据指纹、冻结源码摘要、计划、当前任务、已完成实验、最佳结果、截止时间及报告路径。只保存可序列化的基本类型，不保存 Torch 模型、数据库连接或模型客户端。

| 文件 | 内容 |
|---|---|
| `.agent/state.sqlite3` | 工作流登记、PID、任务、事件、取消标记及幂等请求键 |
| `.agent/workflows/checkpoints.sqlite3` | LangGraph 节点状态、执行位置与中断 |
| `.agent/workflows/<id>/request.json` | 初始请求、配置与源码摘要 |
| `.agent/workflows/<id>/source/` | 冻结的训练器、加载器、模型和检查代码 |
| `.agent/workflows/<id>/worker.log` | 工作流进程输出 |
| `.agent/workflows/<id>/report.md` | 工作流报告 |
| `.agent/runs/<job_id>/` | 每轮训练快照、命令、日志、权重和指标 |

一个项目最多同时运行一个工作流控制进程和一个训练任务。等待数据的图暂停，不占训练任务。SQLite 事务保护领取和恢复；请求键唯一索引保护训练重复提交。检查点关闭 pickle，并限制 MsgPack 类型恢复。以上本地状态不上传 GitHub。

## 4 启动与恢复

Python 3.10+，在已有 PyTorch 环境中运行：

```bash
python -m pip install -r requirements-workflow.txt
python agent.py workflow-start --goal "检查先验门控的影响" --config configs/baseline.json --budget 3
python agent.py workflow-list
python agent.py workflow-status WORKFLOW_ID
python agent.py workflow-events WORKFLOW_ID
python agent.py workflow-report WORKFLOW_ID
```

启动命令立即返回。独立工作流进程会等待受管理的训练任务，正常运行时自动执行后续实验，无需手动逐轮恢复。网页刷新只读状态，不会触发训练。

缺少数据时，图在 `wait_for_data` 暂停并退出控制进程。准备好数据后，或控制进程异常退出后，恢复同一个 ID：

```bash
python agent.py workflow-resume WORKFLOW_ID
```

恢复读取同一线程检查点。每轮使用 `langgraph:<workflow_id>:<trial_index>` 请求键；如果训练提交已经写入任务数据库而图检查点尚未写入，重放提交节点会复用旧任务。任务仍运行则继续等待，已完成则读取结果。

**恢复边界：**图恢复不恢复优化器状态。训练工作进程退出且没有残留子进程时，该轮记为中断并保留证据，之后按预算处理下一候选。有残留训练子进程时需先处理残留进程；系统不会对历史 PID 强制杀进程。所有实验失败且报告已生成时，图已结束，应新建工作流。

`timeout_seconds` 在数据就绪、生成计划后成为整个图的墙钟预算，包含等待和后续离线时间。每轮训练的任务超时不超过提交时剩余预算；剩余不足 60 秒时不再启动新训练。控制进程退出期间，已有训练仍受自身任务超时约束，图恢复时再核对总截止时间。

启动时冻结训练器、加载器和模型源码；每轮提交核对训练/验证 CSV 指纹。CSV 内容改变会拒绝继续，需恢复原文件或新建工作流。指纹并不检查每个 NPZ 的内容，观测文件应保持不变。

## 5 取消与最终评估

```bash
python agent.py workflow-cancel WORKFLOW_ID
```

取消持久保存标记并请求取消关联的活动训练，取消后不能使用恢复命令重新启动。图不自动应用候选代码，也不自动做最终测试。需要最终测试时，从状态的 `best.job_id` 获取训练任务 ID：

```bash
python agent.py evaluate BEST_JOB_ID
```

检查点、阈值和实验选择只依据验证数据，测试结果不反馈到图中。

## 6 接口

- CLI：`workflow-start/list/status/resume/cancel/events/report`。
- 网页：“科研工作流”页支持启动、进度、恢复、取消与下载报告。
- MCP：`start_research_workflow`、`list_research_workflows`、`get_research_workflow`、`resume_research_workflow`、`cancel_research_workflow`、`get_research_workflow_events`、`export_research_workflow_report`。
- 对话：`/workflow 研究目标`、`/workflow-status ID`、`/workflow-resume ID`、`/workflow-cancel ID`。
- `run_ai_scientist()` 返回 `workflow_id`，旧 `train`、`run-plan`、`report` 和提案/回滚工具保留。
- 网页安装 `requirements-ui.txt`；MCP 安装 `requirements-agent.txt`，训练另需 `requirements.txt`。
- API/Token 配置暂缓，安装 LangGraph 不会自动配置或调用 LLM。

## 7 当前验证范围

本次仅进行了源代码静态检查和发布范围检查，没有添加或运行测试，没有触发真实训练，尚未完成运行时、端到端或 GPU 验证。本文描述实现流程，不代表已测得准确率、吞吐或故障恢复成功率。

## 8 官方资料

- [StateGraph](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [检查点与同步持久化](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [中断与恢复](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [SQLite Checkpointer](https://pypi.org/project/langgraph-checkpoint-sqlite/)
