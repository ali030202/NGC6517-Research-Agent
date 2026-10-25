"""Detached worker: bounded subprocess training, cooperative cancellation, provenance."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .code import atomic_text, digest
from .config import TrainConfig
from .data import preflight
from .store import Store


def process_options():
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def process_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: do not assume dead
        try:
            code = ctypes.c_uint32()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        # A detached worker can remain a zombie until its long-lived UI parent
        # reaps it. Such a PID cannot make progress and must not block recovery.
        status = Path(f"/proc/{int(pid)}/stat")
        if status.is_file():
            try:
                state = status.read_text().rsplit(")", 1)[1].split()[0]
                if state in {"Z", "X"}:
                    return False
            except (OSError, IndexError):
                pass
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_child(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        # This PID comes from the current Popen object, never from user input.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=15, creationflags=subprocess.CREATE_NO_WINDOW, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=10)


def snapshot(root, folder, proposal_path=None):
    source = folder / "source"
    source.mkdir(parents=True, exist_ok=True)
    files = ["train_fewshot.py", "dataset_ngc6517.py", "model_proto.py"]
    hashes = {}
    for relative in files:
        origin = proposal_path if relative == "model_proto.py" and proposal_path else root / relative
        content = origin.read_bytes()
        (source / relative).write_bytes(content)
        hashes[relative] = digest(content)
    package = source / "ngc_agent"
    package.mkdir()
    # Training uses dependency-light split checking from the same frozen snapshot.
    for name in ("__init__.py", "data.py"):
        content = (root / "ngc_agent" / name).read_bytes()
        (package / name).write_bytes(content)
        hashes[f"ngc_agent/{name}"] = digest(content)
    return source, hashes


class JobStopped(Exception):
    def __init__(self, state):
        self.state = state


def execute_trial(root, store, job, config, folder, source, manifest, started, timeout):
    check = preflight(root, config, final_test=job["kind"] == "evaluate")
    if not check["ready"]:
        raise ValueError("数据预检查未通过: " + json.dumps(check, ensure_ascii=False))
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(source / "train_fewshot.py")]
    command += config.command_args(root)
    command += ["--out_dir", str(folder), "--save_path", str(folder / "model.pt")]
    checkpoint = None
    if job["kind"] == "evaluate":
        source_job = store.get(job["source_run"])
        checkpoint = Path(source_job["result"]["weight_path"])
        if not checkpoint.is_file():
            raise ValueError("Source checkpoint is missing")
        command += ["--evaluate_checkpoint", str(checkpoint), "--final_test"]
    trial_manifest = {**manifest, "command": command, "data": check, "config": config.to_dict(),
                      "python": sys.version, "executable": sys.executable}
    atomic_text(folder / "trial_manifest.json", json.dumps(trial_manifest, ensure_ascii=False, indent=2))
    environment = os.environ.copy()
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    child = None
    try:
        with (folder / "console.log").open("ab", buffering=0) as console:
            child = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=console, stderr=subprocess.STDOUT, **process_options())
            store.update(job["id"], child_pid=child.pid)
            while child.poll() is None:
                if store.get(job["id"])["cancel_requested"]:
                    raise JobStopped("cancelled")
                if time.monotonic() - started > timeout:
                    raise JobStopped("timed_out")
                store.update(job["id"])
                time.sleep(1)
        if store.get(job["id"])["cancel_requested"]:
            raise JobStopped("cancelled")
        if child.returncode != 0:
            raise RuntimeError(f"Training exited with code {child.returncode}; see {folder / 'console.log'}")
    finally:
        if child is not None and child.poll() is None:
            stop_child(child)
        if child is not None and child.poll() is not None:
            store.update(job["id"], child_pid=0)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    val_f1 = summary.get("best", {}).get("f1")
    if isinstance(val_f1, bool) or not isinstance(val_f1, (int, float)) or not math.isfinite(val_f1) or not 0 <= val_f1 <= 1:
        raise ValueError("训练结果缺少有效验证 F1；不会将该实验标记成功。")
    result = {
        "config": config.to_dict(), "val_f1": val_f1, "val_acc": summary["best"].get("acc"),
        "val_cost": summary["best"].get("cost"), "best_episode": summary["best"].get("episode"),
        "weight_path": str(checkpoint or folder / "model.pt"),
        "summary_path": str(folder / "summary.json"), "model_sha256": manifest["source_sha256"]["model_proto.py"],
        "final_test": summary.get("final_test"), "selection_metric": "val_f1",
    }
    if not Path(result["weight_path"]).is_file():
        raise ValueError("训练返回成功但没有生成权重")
    if job["kind"] != "evaluate" and result["final_test"] is not None:
        raise ValueError("训练阶段意外评估了测试集；该结果不进入自动搜索。")
    return result


def run_job(root, job_id):
    root = Path(root).resolve()
    store = Store(root)
    store.claim(job_id, os.getpid())
    job = store.get(job_id)
    folder = store.directory / "runs" / job_id
    trials = []
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        source = folder / "source"
        for relative, expected in manifest["source_sha256"].items():
            if digest((source / relative).read_bytes()) != expected:
                raise ValueError(f"Frozen source checksum mismatch: {relative}")
        if job["kind"] == "search":
            plan = job["config"]
            planned_trials = plan["trials"]
            timeout = plan["timeout_seconds"]
        else:
            plan = None
            planned_trials = [{"name": job["kind"], "config": job["config"]}]
            timeout = job["config"]["timeout_seconds"]
        started = time.monotonic()
        best = None
        for index, trial in enumerate(planned_trials, 1):
            if store.get(job_id)["cancel_requested"]:
                raise JobStopped("cancelled")
            if time.monotonic() - started > timeout:
                raise JobStopped("timed_out")
            config = TrainConfig.from_dict(trial["config"])
            output = folder / f"trial_{index:02d}" if plan else folder
            store.event(job_id, "trial_started", {"index": index, "name": trial["name"]})
            try:
                result = execute_trial(root, store, job, config, output, source, manifest, started, timeout)
                trials.append({"index": index, "name": trial["name"], "state": "succeeded", "result": result})
                if best is None or result["val_f1"] > best["val_f1"]:
                    best = result
                store.event(job_id, "trial_succeeded", {"index": index, "val_f1": result["val_f1"]})
            except JobStopped:
                raise
            except Exception as exc:
                if not plan:
                    raise
                trials.append({"index": index, "name": trial["name"], "state": "failed", "error": f"{type(exc).__name__}: {exc}"})
                store.event(job_id, "trial_failed", trials[-1])
            progress = {"trials": trials, "best": best, "completed_trials": len(trials)}
            atomic_text(folder / "progress.json", json.dumps(progress, ensure_ascii=False, indent=2))
            if plan and best is not None and best["val_f1"] >= plan["target_f1"]:
                break
        if best is None:
            raise ValueError("所有实验均失败；请查看各轮日志。")
        final = {**best, "trials": trials,
                 "stop_reason": "target_reached" if plan and best["val_f1"] >= plan["target_f1"] else "budget_exhausted" if plan else "completed"}
        store.update(job_id, state="succeeded", result=final)
    except JobStopped as exc:
        store.update(job_id, state=exc.state, result={"trials": trials})
    except BaseException as exc:
        store.update(job_id, state="failed", error=f"{type(exc).__name__}: {exc}", result={"trials": trials})
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
