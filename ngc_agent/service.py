"""The same agent operations are used by CLI, Gradio and MCP."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from .code import CodeManager, atomic_text, digest
from .config import ROOT, TrainConfig
from .data import preflight
from .knowledge import search
from .runner import process_alive, process_options, snapshot
from .store import Store, TERMINAL, now


class Agent:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.store = Store(self.root)
        self.code = CodeManager(self.root)

    def capabilities(self):
        dependencies = {name: importlib.util.find_spec(name) is not None for name in ("torch", "numpy", "tqdm", "gradio", "fastmcp", "openai", "arxiv", "pypdf")}
        llm = all(os.getenv(key) for key in ("AGENT_API_KEY", "AGENT_BASE_URL", "AGENT_MODEL")) or all(os.getenv(key) for key in ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL"))
        return {
            "local_core": True, "llm": bool(llm and dependencies["openai"]),
            "llm_state": "available" if llm and dependencies["openai"] else "deferred: API credentials/model not configured",
            "dependencies": dependencies, "python": sys.executable,
            "selection_metric": "val_f1", "root": str(self.root),
        }

    def inspect(self, config=None, include_test=False):
        return preflight(self.root, config or TrainConfig(), final_test=include_test)

    def plan(self, config=None, budget=3, target_f1=0.9):
        config = config or TrainConfig()
        if type(budget) is not int or not 1 <= budget <= 10:
            raise ValueError("budget must be between 1 and 10")
        if isinstance(target_f1, bool) or not isinstance(target_f1, (int, float)) or not 0 <= target_f1 <= 1:
            raise ValueError("target_f1 must be in [0,1]")
        # A local ablation plan is explicit rule-based planning, not an LLM result
        # and not a claim of novelty or guaranteed performance.
        choices = [
            ("baseline", config, "建立可重复的基线"),
            ("prior_ablation", replace(config, no_prior=not config.no_prior), "检验先验门控是否有效"),
            ("regularization", replace(config, dropout=min(0.9, config.dropout + 0.1)), "检验增加正则化的影响"),
            ("learning_rate", replace(config, lr=max(1e-6, config.lr * 0.5)), "检验较小学习率"),
        ]
        for seed in range(1, 7):
            choices.append((f"seed_{seed}", replace(config, seed=(config.seed + seed) % (2**32)), "估计随机种子敏感性；单次结果不代表稳定收益"))
        return {
            "planner": "local_rules", "goal": "maximize validation F1", "target_f1": target_f1,
            "budget": budget, "stop": "达到验证目标、预算耗尽、取消或超时即停止",
            "timeout_seconds": config.timeout_seconds,
            "trials": [{"name": name, "config": trial.to_dict(), "rationale": reason} for name, trial, reason in choices[:budget]],
            "limits": ["目标值不保证达到", "验证集参与搜索，最终性能只在冻结方案后单独评估测试集", "默认每轮保持除消融变量外的设置一致"],
        }

    def _submit(self, kind, payload, source_run=None, proposal_id=None):
        for previous in self.store.jobs(1000):
            if previous.get("child_pid") and process_alive(previous["child_pid"]):
                raise ValueError("已有训练子进程运行；请等待完成或取消。工作进程异常退出时，需先结束残留训练进程。")
        capabilities = self.capabilities()
        if not all(capabilities["dependencies"][name] for name in ("torch", "numpy", "tqdm")):
            raise ValueError("当前 Python 缺少训练依赖；使用已安装 PyTorch 的环境，或安装 requirements.txt。")
        config = TrainConfig.from_dict(payload["trials"][0]["config"] if kind == "search" else payload)
        check = self.inspect(config, include_test=kind == "evaluate")
        if not check["ready"]:
            raise ValueError("数据检查未通过：" + json.dumps(check, ensure_ascii=False))
        source = None
        if proposal_id:
            _, source = self.code.read(proposal_id)
        if kind == "evaluate":
            original = self.store.get(source_run)
            if original["state"] != "succeeded" or original["kind"] not in {"train", "search"}:
                raise ValueError("Final evaluation requires a successful training/search job")
            source = self.run_directory(source_run) / "source" / "model_proto.py"
        job_id = uuid.uuid4().hex
        folder = self.run_directory(job_id)
        folder.mkdir(parents=True)
        try:
            snapshot_root = self.run_directory(source_run) / "source" if kind == "evaluate" else self.root
            _, hashes = snapshot(snapshot_root, folder, source)
            if kind == "evaluate" and hashes["model_proto.py"] != original["result"]["model_sha256"]:
                raise ValueError("Source model checksum mismatch")
            atomic_text(folder / "manifest.json", json.dumps({
                "id": job_id, "kind": kind, "created_at": now(), "config": payload,
                "source_run": source_run, "proposal_id": proposal_id,
                "source_sha256": hashes, "selection_metric": "val_f1",
            }, ensure_ascii=False, indent=2))
            self.store.create(job_id, kind, payload, source_run=source_run, proposal_id=proposal_id)
        except Exception:
            # Keep a failed preparation manifest for inspection; no job or process
            # is created when another job already holds the active slot.
            raise
        environment = os.environ.copy()
        environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        try:
            with (folder / "worker.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [sys.executable, str(self.root / "agent.py"), "--root", str(self.root), "_worker", job_id],
                    cwd=self.root, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, **process_options(),
                )
            self.store.update(job_id, worker_pid=process.pid)
        except Exception as exc:
            self.store.update(job_id, state="failed", error=f"Worker launch failed: {exc}")
            raise
        return self.store.get(job_id)

    def submit(self, config=None, proposal_id=None):
        return self._submit("train", (config or TrainConfig()).to_dict(), proposal_id=proposal_id)

    def run_plan(self, config=None, budget=3, target_f1=0.9, proposal_id=None):
        return self._submit("search", self.plan(config, budget, target_f1), proposal_id=proposal_id)

    def evaluate(self, job_id):
        original = self.store.get(job_id)
        if original["state"] != "succeeded" or original["kind"] not in {"train", "search"}:
            raise ValueError("请先选择一个训练成功的任务")
        # Recover the exact winning configuration instead of current UI defaults.
        config = original["result"].get("config", original["config"])
        return self._submit("evaluate", config, source_run=job_id)

    def retry(self, job_id):
        original = self.store.get(job_id)
        if original["state"] not in TERMINAL:
            raise ValueError("活动任务不能重试；请先取消或等待完成")
        # An explicit new run, not optimizer/checkpoint continuation.
        if original["kind"] == "evaluate":
            return self.evaluate(original["source_run"])
        proposal = self.code.stage((self.run_directory(job_id) / "source" / "model_proto.py").read_text(encoding="utf-8-sig"), f"Retry frozen model from {job_id}")
        return self._submit(original["kind"], original["config"], proposal_id=proposal["id"])

    def recover(self, job_id):
        job = self.store.get(job_id)
        if job["state"] in TERMINAL:
            return job
        if process_alive(job["worker_pid"]):
            raise ValueError("任务进程仍在运行；请用取消功能停止它。")
        if process_alive(job.get("child_pid")):
            raise ValueError("后台工作进程已退出，但训练子进程仍在运行；请先结束该进程，再标记中断。")
        self.store.update(job_id, state="interrupted", error="Worker exited before recording a terminal state; use retry to start a new run")
        return self.store.get(job_id)

    def run_directory(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("Invalid job ID")
        return self.store.directory / "runs" / job_id

    def logs(self, job_id, lines=80):
        self.store.get(job_id)
        folder = self.run_directory(job_id)
        files = sorted(folder.glob("trial_*/console.log")) or [folder / "console.log"]
        path = files[-1]
        if not path.is_file():
            path = folder / "worker.log"
        if not path.is_file():
            return "任务正在准备，尚无日志。"
        # Bound memory even for very long runs.
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            text = handle.read().decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-min(max(int(lines), 1), 500):])

    def knowledge(self, query, top_k=4):
        return search(self.root, query, top_k)

    def propose(self, direction):
        if not self.capabilities()["llm"]:
            return {"status": "deferred", "reason": "未配置 API；本地功能可用。也可以通过 stage 命令导入手写模型提案。"}
        from skills.llm_client import create_llm_client, get_response_from_llm
        client, model = create_llm_client()
        current = self.code.target.read_text(encoding="utf-8-sig")
        response, _ = get_response_from_llm(
            f"改进方向：{direction}\n当前代码：\n```python\n{current}\n```",
            client, model,
            system_message="生成完整 model_proto.py，保持 NGC6517Encoder(embed_dim, dropout, use_prior) 和 prototypical_logits 接口。仅输出一个 python 代码块。不要运行命令，不要声称有实验收益。",
        )
        match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.S)
        if not match:
            raise ValueError("LLM response has no complete Python code block")
        return self.code.stage(match.group(1), direction)

    def report(self, job_id=None):
        jobs = [self.store.get(job_id)] if job_id else list(reversed(self.store.jobs(1000)))
        lines = ["# NGC6517 科研智能体实验报告", "", f"生成时间：{now()}", "",
                 "模型选择指标：验证集 F1。测试集只用于显式最终评估；本报告不会据此推荐下一个实验。", "",
                 "| 任务 | 类型 | 状态 | 验证 F1 | 最终测试 F1 |", "|---|---|---|---|---|"]
        for job in jobs:
            result = job.get("result") or {}
            test = result.get("final_test") or {}
            lines.append(f"| {job['id']} | {job['kind']} | {job['state']} | {result.get('val_f1', '未产生')} | {test.get('f1', '未评估')} |")
        successful = [job for job in jobs if job["state"] == "succeeded" and job["kind"] in {"train", "search"}]
        if successful:
            best = max(successful, key=lambda job: job["result"]["val_f1"])
            lines += ["", f"验证 F1 最高任务：`{best['id']}`，F1={best['result']['val_f1']:.6f}。",
                      "不同划分、种子、shot 数或协议的结果不可直接视作严格消融比较。"]
        else:
            lines += ["", "尚无成功训练记录，无法给出性能结论。"]
        for job in jobs:
            lines += ["", f"## {job['id']}", "", "```json", json.dumps({"kind": job["kind"], "state": job["state"], "config": job["config"], "result": job["result"], "error": job["error"]}, ensure_ascii=False, indent=2), "```",
                      "", f"原始证据：`.agent/runs/{job['id']}/manifest.json`、控制台日志及 summary.json。"]
        lines += ["", "## 局限", "", "静态代码检查不证明模型正确性。自动计划和分析属于实验建议；新颖性、因果解释和论文结论需要人工核实。", "缺少验证重复、独立测试或真实日志时，不生成虚构的收益数字。"]
        text = "\n".join(lines) + "\n"
        path = self.store.directory / "reports" / (f"{job_id}.md" if job_id else "experiments.md")
        atomic_text(path, text)
        return {"path": str(path), "markdown": text}

    def chat(self, message, last_job=None):
        text = message.strip()
        command, _, argument = text.partition(" ")
        if text in {"帮助", "help", "/help"}:
            reply = "可用：状态 /status；数据 /data；计划 /plan；训练 /train；自动实验 /run-plan；报告 /report；/search 关键词；/logs 任务ID；/cancel 任务ID。训练和自动实验会启动本地计算任务。/propose 改进方向 仅在大模型接入后生成代码提案。"
        elif text in {"状态", "/status"}:
            reply = json.dumps({"capabilities": self.capabilities(), "jobs": self.store.jobs(10)}, ensure_ascii=False, indent=2)
        elif text in {"数据", "/data"}:
            reply = json.dumps(self.inspect(), ensure_ascii=False, indent=2)
        elif text in {"计划", "/plan"}:
            reply = json.dumps(self.plan(), ensure_ascii=False, indent=2)
        elif text in {"训练", "开始训练", "/train"}:
            job = self.submit()
            last_job = job["id"]
            reply = f"训练已提交，任务 `{last_job}`。可查看日志或取消。"
        elif text in {"自动实验", "/run-plan"}:
            job = self.run_plan()
            last_job = job["id"]
            reply = f"已提交最多 3 次的本地消融计划，任务 `{last_job}`；按验证 F1 决定是否停止。"
        elif command in {"/cancel", "/logs"}:
            job_id = argument.strip() or last_job
            if not job_id:
                raise ValueError("请提供任务 ID")
            if command == "/cancel":
                self.store.cancel(job_id)
                reply = f"已请求取消 `{job_id}`。"
            else:
                reply = "```text\n" + self.logs(job_id) + "\n```"
        elif text in {"报告", "/report"}:
            reply = self.report()["markdown"]
        elif command == "/search" and argument:
            hits = self.knowledge(argument)
            reply = "\n\n".join(f"**{hit['source_id']}**\n\n{hit['content']}" for hit in hits) or "本地资料未找到相关内容。"
        elif command == "/propose" and argument:
            reply = json.dumps(self.propose(argument), ensure_ascii=False, indent=2)
        else:
            reply = "本地模式支持实验工具和明确命令。输入「帮助」查看功能；开放式大模型对话待接入 API。"
        return {"reply": reply, "last_job": last_job}
