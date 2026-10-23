"""Optional LLM architecture experiments using isolated, managed candidate runs."""
from ngc_agent.service import Agent
from skills.train_tools import wait_job


def perform_experiments(idea, max_runs=3, max_iters=3):
    if not 1 <= max_runs <= 5 or not 1 <= max_iters <= 4:
        raise ValueError("Invalid experiment budget")
    agent = Agent()
    if not agent.capabilities()["llm"]:
        return {"final_status": "deferred", "reason": "LLM credentials are not configured", "runs": []}
    logs = [job for job in agent.store.jobs(1000) if job["state"] == "succeeded" and job["kind"] in {"train", "search"}]
    baseline = max((job["result"]["val_f1"] for job in logs), default=0.0)
    result = {"idea_name": idea.get("Name", "unnamed"), "initial_best_f1": baseline, "runs": []}
    direction = idea.get("Experiment") or idea.get("Title") or "设计一个可消融的架构改进"
    best = baseline
    for index in range(max_runs):
        trial = {"run": index + 1}
        try:
            proposal = None
            for _ in range(max_iters):
                try:
                    proposal = agent.propose(direction)
                    break
                except (ValueError, SyntaxError):
                    continue
            if proposal is None or "id" not in proposal:
                raise ValueError("No valid code proposal was produced")
            trial["proposal_id"] = proposal["id"]
            job = agent.submit(proposal_id=proposal["id"])
            train = wait_job(agent, job["id"])
            best = max(best, train["val_f1"])
            trial.update(status="success", new_f1=train["val_f1"], train_result={"new_val_f1": train["val_f1"], "train_result": train})
        except Exception as exc:
            trial.update(status="failed", error=str(exc))
        result["runs"].append(trial)
    result.update(final_best_f1=best, final_status="improved" if best > baseline else "no_improvement")
    result["limitation"] = "Comparisons across different data/configuration protocols require manual review. Candidates are staged, never automatically applied."
    return result


def analyze_experiment_results(train_result, idea):
    score = train_result.get("new_val_f1")
    return {"status": "observed" if score is not None else "needs_evidence",
            "val_f1": score, "idea": idea.get("Title", ""),
            "analysis": "F1 is an observation. Use controlled ablations, losses, confusion matrices and repeated seeds before attributing a cause."}
