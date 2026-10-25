"""Checkpointed LangGraph research orchestration over the managed training service.

Importing this module does not import LangGraph, Torch, an LLM SDK or a UI.
Graph checkpoints and the task ledger have separate responsibilities: the graph
remembers control flow; the ledger makes training submission idempotent.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid

from .code import atomic_text, digest
from .config import TrainConfig
from .runner import process_alive, process_options, snapshot
from .store import TERMINAL, now


WORKFLOW_TERMINAL = {"succeeded", "cancelled", "timed_out"}


def runtime():
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        from langgraph.types import Command, interrupt
        from typing_extensions import TypedDict
    except ImportError as exc:
        raise ValueError("LangGraph 工作流依赖未安装或版本不兼容。请在当前环境运行 python -m pip install -r requirements-langgraph.txt") from exc
    return StateGraph, START, END, SqliteSaver, JsonPlusSerializer, Command, interrupt, TypedDict


class ResearchWorkflow:
    def __init__(self, agent):
        self.agent = agent
        self.store = agent.store
        self.directory = self.store.directory / "workflows"

    def folder(self, workflow_id):
        if not re.fullmatch(r"[0-9a-f]{32}", workflow_id):
            raise ValueError("Invalid workflow ID")
        return self.directory / workflow_id

    @contextmanager
    def graph(self):
        StateGraph, START, END, SqliteSaver, Serializer, _, interrupt, TypedDict = runtime()

        class ResearchState(TypedDict, total=False):
            workflow_id: str
            goal: str
            config: dict
            budget: int
            target_f1: float
            proposal_id: str
            source_sha256: dict
            phase: str
            data: dict
            data_fingerprints: dict
            knowledge: list
            plan: dict
            trials: list
            best: dict | None
            job_id: str
            deadline: float
            stop_reason: str
            report_path: str

        builder = StateGraph(ResearchState)
        nodes = {
            "inspect_data": self.inspect_data,
            "wait_for_data": lambda state: self.wait_for_data(state, interrupt),
            "retrieve_knowledge": self.retrieve_knowledge,
            "plan_experiments": self.plan_experiments,
            "select_trial": self.select_trial,
            "submit_training": self.submit_training,
            "await_training": self.await_training,
            "analyze_result": self.analyze_result,
            "write_report": self.write_report,
        }
        for name, node in nodes.items():
            builder.add_node(name, node)
        builder.add_edge(START, "inspect_data")
        builder.add_conditional_edges("inspect_data", self.route_data,
                                      {"ready": "retrieve_knowledge", "wait": "wait_for_data", "stop": "write_report"})
        builder.add_edge("wait_for_data", "inspect_data")
        builder.add_edge("retrieve_knowledge", "plan_experiments")
        builder.add_edge("plan_experiments", "select_trial")
        builder.add_conditional_edges("select_trial", lambda state: "stop" if state.get("stop_reason") else "run",
                                      {"stop": "write_report", "run": "submit_training"})
        builder.add_conditional_edges("submit_training", lambda state: "stop" if state.get("stop_reason") else "wait",
                                      {"stop": "write_report", "wait": "await_training"})
        builder.add_edge("await_training", "analyze_result")
        builder.add_edge("analyze_result", "select_trial")
        builder.add_edge("write_report", END)
        self.directory.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.directory / "checkpoints.sqlite3", timeout=30, check_same_thread=False)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            # Only simple state plus LangGraph's built-in safe types are needed.
            # Pickle and arbitrary module reconstruction are never enabled.
            serde = Serializer(pickle_fallback=False, allowed_msgpack_modules=[])
            saver = SqliteSaver(connection, serde=serde)
            yield builder.compile(checkpointer=saver)
        finally:
            connection.close()

    @staticmethod
    def graph_config(workflow_id):
        return {"configurable": {"thread_id": workflow_id}, "recursion_limit": 100}

    def start(self, goal, config, budget, target_f1, proposal_id=None):
        runtime()
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 2000:
            raise ValueError("研究目标必须是 1—2000 字符的文本")
        self.agent.plan(config, budget, target_f1)  # same public config/budget validation
        if any(item["state"] in {"queued", "running"} for item in self.store.workflows()):
            raise ValueError("已有活动工作流；如其进程已退出，请先恢复或取消该工作流。")
        if proposal_id:
            _, candidate = self.agent.code.read(proposal_id)
        else:
            proposal = self.agent.code.stage(self.agent.code.target.read_text(encoding="utf-8-sig"), "Frozen baseline for LangGraph workflow")
            proposal_id = proposal["id"]
            _, candidate = self.agent.code.read(proposal_id)
        workflow_id = uuid.uuid4().hex
        folder = self.folder(workflow_id)
        folder.mkdir(parents=True)
        _, hashes = snapshot(self.agent.root, folder, candidate)
        initial = {
            "workflow_id": workflow_id, "goal": goal.strip(), "config": config.to_dict(),
            "budget": budget, "target_f1": target_f1, "proposal_id": proposal_id,
            "source_sha256": hashes, "phase": "created", "trials": [], "best": None,
            "stop_reason": "", "job_id": "",
        }
        atomic_text(folder / "request.json", json.dumps(initial, ensure_ascii=False, indent=2))
        self.store.create_workflow(workflow_id, initial)
        self.launch(workflow_id)
        return self.status(workflow_id)

    def launch(self, workflow_id):
        record = self.store.workflow(workflow_id)
        environment = os.environ.copy()
        environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        try:
            with (self.folder(workflow_id) / "worker.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [sys.executable, str(self.agent.root / "agent.py"), "--root", str(self.agent.root), "_workflow_worker", workflow_id],
                    cwd=self.agent.root, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, **process_options(),
                )
        except Exception as exc:
            self.store.record_workflow_launch(workflow_id, record["worker_pid"], error=f"Unable to start workflow worker: {exc}")
            raise
        # Never replace the PID of a worker that has already claimed the graph.
        self.store.record_workflow_launch(workflow_id, record["worker_pid"], process.pid)

    def status(self, workflow_id):
        self.folder(workflow_id)
        record = self.store.workflow(workflow_id)
        with self.graph() as graph:
            saved = graph.get_state(self.graph_config(workflow_id))
            values = dict(saved.values or {})
            interrupts = [{"id": item.id, "value": item.value}
                          for task in saved.tasks for item in task.interrupts]
        alive = process_alive(record["worker_pid"])
        active = self.store.get(record["active_job_id"]) if record["active_job_id"] else None
        return {
            "workflow_id": workflow_id, "state": record["state"], "phase": record["phase"],
            "worker_alive": alive, "needs_resume": record["state"] in {"queued", "running", "failed", "waiting_data"} and not alive,
            "cancel_requested": bool(record["cancel_requested"]), "goal": record["request"]["goal"],
            "created_at": record["created_at"], "updated_at": record["updated_at"],
            "next_nodes": list(saved.next), "interrupts": interrupts,
            "completed_trials": len(values.get("trials", [])), "budget": record["request"]["budget"],
            "best": values.get("best"), "stop_reason": values.get("stop_reason"),
            "active_job": active, "error": record["error"],
            "report_path": record["report_path"] or values.get("report_path"),
            "checkpoint_db": str(self.directory / "checkpoints.sqlite3"),
        }

    def resume(self, workflow_id):
        runtime()
        record = self.store.workflow(workflow_id)
        if record["cancel_requested"] or record["state"] in WORKFLOW_TERMINAL:
            raise ValueError("该工作流已结束；请新建工作流。取消不会被恢复操作撤销。")
        if process_alive(record["worker_pid"]):
            return self.status(workflow_id)
        with self.graph() as graph:
            saved = graph.get_state(self.graph_config(workflow_id))
            if saved.values and not saved.next:
                raise ValueError("图已执行完毕；如需新一轮实验，请新建工作流。")
        try:
            queued = self.store.requeue_workflow(workflow_id, record["updated_at"], record["worker_pid"])
        except sqlite3.IntegrityError as exc:
            raise ValueError("另一个工作流正在运行，暂不能恢复。") from exc
        if queued:
            self.launch(workflow_id)
        return self.status(workflow_id)

    def cancel(self, workflow_id):
        record = self.store.workflow(workflow_id)
        if record["state"] not in WORKFLOW_TERMINAL:
            self.store.cancel_workflow(workflow_id)
            if not process_alive(self.store.workflow(workflow_id)["worker_pid"]):
                self.store.update_workflow(workflow_id, state="cancelled", phase="cancelled")
        return self.status(workflow_id)

    def enter(self, state, phase):
        self.store.update_workflow(state["workflow_id"], phase=phase)
        self.store.event(state["workflow_id"], "graph_node", {"node": phase})

    def cancelled(self, state):
        return bool(self.store.workflow(state["workflow_id"])["cancel_requested"])

    def inspect_data(self, state):
        self.enter(state, "inspect_data")
        if self.cancelled(state):
            return {"stop_reason": "cancelled"}
        result = self.agent.inspect(TrainConfig.from_dict(state["config"]))
        return {"data": result, "phase": "data_ready" if result["ready"] else "waiting_data",
                "data_fingerprints": {name: info.get("sha256") for name, info in result["splits"].items()}}

    @staticmethod
    def route_data(state):
        return "stop" if state.get("stop_reason") else "ready" if state["data"]["ready"] else "wait"

    def wait_for_data(self, state, interrupt):
        self.enter(state, "waiting_data")
        # This node has no training/file side effects before interrupt. Resuming
        # always retries inspection against the current local data.
        interrupt({"reason": "data_not_ready", "data": state["data"],
                   "message": "准备好本地训练/验证数据后，恢复同一工作流。"})
        return {"phase": "inspect_data"}

    def retrieve_knowledge(self, state):
        self.enter(state, "retrieve_knowledge")
        return {"knowledge": self.agent.knowledge(state["goal"], top_k=4)}

    def plan_experiments(self, state):
        self.enter(state, "plan_experiments")
        plan = self.agent.plan(TrainConfig.from_dict(state["config"]), state["budget"], state["target_f1"])
        return {"plan": plan, "deadline": time.time() + state["config"]["timeout_seconds"]}

    def select_trial(self, state):
        self.enter(state, "select_trial")
        if self.cancelled(state):
            return {"stop_reason": "cancelled"}
        if time.time() >= state["deadline"]:
            return {"stop_reason": "timed_out"}
        if state.get("best") and state["best"]["val_f1"] >= state["target_f1"]:
            return {"stop_reason": "target_reached"}
        if len(state["trials"]) >= len(state["plan"]["trials"]):
            return {"stop_reason": "budget_exhausted" if state.get("best") else "all_trials_failed"}
        return {"phase": "submit_training", "job_id": ""}

    def submit_training(self, state):
        self.enter(state, "submit_training")
        if self.cancelled(state):
            return {"stop_reason": "cancelled"}
        index = len(state["trials"])
        request_key = f"langgraph:{state['workflow_id']}:{index}"
        existing = self.store.find_request(request_key)
        if existing:
            # A submission may have committed before the graph checkpoint did.
            # Reuse it even if the remaining timeout has changed after restart.
            job = self.agent._submit("train", existing["config"], proposal_id=state["proposal_id"],
                                     request_key=request_key, workflow_id=state["workflow_id"])
        else:
            remaining = state["deadline"] - time.time()
            if remaining < 60:
                return {"stop_reason": "timed_out"}
            config = TrainConfig.from_dict(state["plan"]["trials"][index]["config"])
            check = self.agent.inspect(config)
            hashes = {name: info.get("sha256") for name, info in check["splits"].items()}
            if not check["ready"] or hashes != state["data_fingerprints"]:
                raise ValueError("实验期间的数据清单已改变或数据不可用；请恢复原始数据，或新建工作流。")
            source = self.folder(state["workflow_id"]) / "source"
            for relative, expected in state["source_sha256"].items():
                if digest((source / relative).read_bytes()) != expected:
                    raise ValueError(f"Workflow frozen source checksum mismatch: {relative}")
            config = replace(config, timeout_seconds=min(config.timeout_seconds, int(remaining)))
            job = self.agent._submit("train", config.to_dict(), proposal_id=state["proposal_id"],
                                     request_key=request_key, workflow_id=state["workflow_id"], frozen_root=source)
        self.store.update_workflow(state["workflow_id"], active_job_id=job["id"])
        return {"job_id": job["id"], "phase": "await_training"}

    def await_training(self, state):
        self.enter(state, "await_training")
        while True:
            job = self.store.get(state["job_id"])
            if job["state"] in TERMINAL:
                return {"phase": "analyze_result"}
            if self.cancelled(state) or time.time() >= state["deadline"]:
                if not job["cancel_requested"]:
                    self.store.cancel(job["id"])
            if not process_alive(job["worker_pid"]):
                if process_alive(job["child_pid"]):
                    raise ValueError("训练工作进程退出，但训练子进程仍在运行；请先处理残留子进程，再恢复工作流。")
                if job["state"] == "queued":
                    self.agent._launch_job(job["id"])
                else:
                    self.agent.recover(job["id"])
            time.sleep(1)

    def analyze_result(self, state):
        self.enter(state, "analyze_result")
        job = self.store.get(state["job_id"])
        planned = state["plan"]["trials"][len(state["trials"])]
        trial = {"name": planned["name"], "job_id": job["id"], "state": job["state"],
                 "result": job["result"], "error": job["error"]}
        best = state.get("best")
        if job["state"] == "succeeded":
            result = job["result"] or {}
            score = result.get("val_f1")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Recorded successful job has no valid validation metric")
            if result.get("final_test") is not None:
                raise ValueError("Final test results cannot enter workflow selection")
            if best is None or score > best["val_f1"]:
                best = {**result, "job_id": job["id"]}
        self.store.update_workflow(state["workflow_id"], active_job_id=None)
        return {"trials": [*state["trials"], trial], "best": best, "phase": "select_trial"}

    def write_report(self, state):
        self.enter(state, "write_report")
        reason = state.get("stop_reason") or "completed"
        phase = reason if reason in {"cancelled", "timed_out"} else "succeeded" if state.get("best") else "failed"
        lines = ["# LangGraph 科研实验报告", "", f"工作流：`{state['workflow_id']}`", f"研究目标：{state['goal']}",
                 f"生成时间：{now()}", f"停止原因：{reason}", "",
                 "规划方式：本地规则消融；知识检索只提供参考片段，不自动改变实验设计。",
                 "模型选择只使用验证 F1；最终测试需要冻结方案后单独显式提交。", "",
                 "| 实验 | 任务 | 状态 | 验证 F1 |", "|---|---|---|---|"]
        for trial in state.get("trials", []):
            score = (trial.get("result") or {}).get("val_f1", "未产生")
            lines.append(f"| {trial['name']} | {trial['job_id']} | {trial['state']} | {score} |")
        if state.get("best"):
            lines += ["", f"选中任务：`{state['best']['job_id']}`。", f"权重：`{state['best']['weight_path']}`。"]
        else:
            lines += ["", "尚无成功训练记录，不能给出模型性能结论。"]
        lines += ["", "## 本地知识来源", ""]
        lines += [f"- {hit['source_id']}" for hit in state.get("knowledge", [])] or ["未检索到相关片段。"]
        lines += ["", "## 实验记录", "", "```json", json.dumps(state.get("trials", []), ensure_ascii=False, indent=2), "```",
                  "", "图恢复会复用已提交的任务。被中断的训练不会恢复优化器状态；失败记录被保留。",
                  "不同数据、种子和协议的结果不能直接解释为模块的因果收益。"]
        path = self.folder(state["workflow_id"]) / "report.md"
        atomic_text(path, "\n".join(lines) + "\n")
        return {"phase": phase, "stop_reason": reason, "report_path": str(path)}


def run_workflow(root, workflow_id):
    from .service import Agent
    engine = ResearchWorkflow(Agent(root))
    engine.folder(workflow_id)
    if not engine.store.claim_workflow(workflow_id, os.getpid()):
        return
    try:
        with engine.graph() as graph:
            config = engine.graph_config(workflow_id)
            saved = graph.get_state(config)
            if not saved.values:
                incoming = engine.store.workflow(workflow_id)["request"]
            elif any(task.interrupts for task in saved.tasks):
                incoming = runtime()[5](resume={"retry": True})
            elif saved.next:
                incoming = None
            else:
                incoming = None
                engine.store.update_workflow(workflow_id, state=saved.values.get("phase", "failed"),
                                              phase=saved.values.get("phase", "failed"), report_path=saved.values.get("report_path"))
                return
            graph.invoke(incoming, config, durability="sync")
            saved = graph.get_state(config)
            paused = any(task.interrupts for task in saved.tasks)
            phase = "waiting_data" if paused else saved.values.get("phase", "failed")
            engine.store.update_workflow(workflow_id, state=phase, phase=phase,
                                          report_path=saved.values.get("report_path"), error=None)
            engine.store.event(workflow_id, "workflow_paused" if paused else "workflow_finished", {"state": phase})
    except BaseException as exc:
        engine.store.update_workflow(workflow_id, state="failed", error=f"{type(exc).__name__}: {exc}")
        engine.store.event(workflow_id, "workflow_failed", {"error": str(exc)})
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
