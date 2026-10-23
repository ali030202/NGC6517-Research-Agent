"""Legacy synchronous research adapter around managed jobs, not a second trainer."""
import time
from ngc_agent.config import TrainConfig
from ngc_agent.service import Agent
from ngc_agent.store import TERMINAL


def set_mcp_instance(instance):
    # Preserved for extensions importing the old registration hook.
    return None


def init_global_log():
    Agent()


def get_train_logs():
    agent = Agent()
    rows = []
    for job in reversed(agent.store.jobs(1000)):
        if job["kind"] not in {"train", "search"} or job["state"] != "succeeded":
            continue
        result = job["result"]
        rows.append({**result["config"], **result, "job_id": job["id"],
                     "train_output_dir": str(agent.run_directory(job["id"]))})
    return rows


def wait_job(agent, job_id):
    try:
        while True:
            job = agent.store.get(job_id)
            if job["state"] in TERMINAL:
                if job["state"] != "succeeded":
                    raise RuntimeError(f"Job {job_id}: {job['state']}; {job.get('error') or ''}")
                return {**job["result"]["config"], **job["result"], "job_id": job_id}
            time.sleep(1)
    except KeyboardInterrupt:
        agent.store.cancel(job_id)
        raise


def run_train(lr=0.001, embed_dim=128, dropout=0.1, temperature=10.0, k_shot=20, episodes=500,
              proposal_id=None, no_prior=False, seed=2026):
    agent = Agent()
    config = TrainConfig(lr=lr, embed_dim=embed_dim, dropout=dropout, temperature=temperature,
                         k_shot=k_shot, q_shot=k_shot, episodes=episodes, no_prior=no_prior, seed=seed)
    return wait_job(agent, agent.submit(config, proposal_id)["id"])


def suggest_next_experiment():
    logs = get_train_logs()
    used = {(log["lr"], log["dropout"], log["no_prior"]) for log in logs}
    for trial in Agent().plan(budget=4)["trials"]:
        config = trial["config"]
        if (config["lr"], config["dropout"], config["no_prior"]) not in used:
            return {**config, "reason": trial["rationale"]}
    best = max(logs, key=lambda log: log["val_f1"]) if logs else TrainConfig().to_dict()
    return {**best, "reason": "基础消融已经完成；建议固定方案后做独立种子重复。"}


def evaluate_progress(target_f1=0.9):
    logs = get_train_logs()
    best = max(logs, key=lambda log: log["val_f1"]) if logs else None
    score = best["val_f1"] if best else 0.0
    return {"target_f1": target_f1, "selection_metric": "val_f1", "best_f1": score,
            "gap": max(0, target_f1-score), "need_continue": score < target_f1,
            "best_config": best, "best_weight": best.get("weight_path") if best else None}


def auto_iteration(target_f1=0.9, max_retry_times=3):
    if not 0 <= max_retry_times <= 3:
        raise ValueError("max_retry_times must be in [0,3]")
    config = suggest_next_experiment()
    errors = []
    for _ in range(max_retry_times + 1):
        try:
            result = run_train(**{key: config[key] for key in ("lr", "embed_dim", "dropout", "temperature", "k_shot", "episodes", "no_prior", "seed")})
            progress = evaluate_progress(target_f1)
            return {"status": "success", "new_val_f1": result["val_f1"],
                    "train_result": result, "progress": progress,
                    "best_f1_all": progress["best_f1"], "need_continue": progress["need_continue"]}
        except Exception as exc:
            errors.append(str(exc))
    return {"status": "fail_all_retry", "msg": errors[-1], "errors": errors}


def analyze_model_defect(target_f1=0.9):
    progress = evaluate_progress(target_f1)
    return {"status": "needs_evidence" if progress["need_continue"] else "target_reached",
            "progress": progress,
            "analysis": "仅凭 F1 无法诊断过拟合或表征坍缩；请结合验证损失、混淆矩阵和重复实验。"}
