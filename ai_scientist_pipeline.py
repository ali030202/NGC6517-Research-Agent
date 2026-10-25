"""Public LangGraph workflow facade. Local research runs without an LLM."""
from ngc_agent.service import Agent


def run_ai_scientist(target_f1=0.9, max_ideas=3, max_experiment_runs=1,
                     enable_writeup=True, enable_review=True):
    budget = max_ideas * max_experiment_runs
    if not 1 <= budget <= 10:
        raise ValueError("The total local experiment budget must be in [1,10]")
    agent = Agent()
    workflow = agent.workflow_start(budget=budget, target_f1=target_f1)
    return {"status": "submitted", "workflow_id": workflow["workflow_id"],
            "workflow": workflow, "report_command": f"python agent.py workflow-report {workflow['workflow_id']}",
            "writeup": "Optional LLM drafting is separate; deterministic reports are always available",
            "review": "Optional LLM review is separate and does not establish scientific validity"}


def do_idea(idea, max_experiment_runs=3, max_iters=3, enable_writeup=True, enable_review=True):
    from skills.agent_experiments import perform_experiments
    result = perform_experiments(idea, max_experiment_runs, max_iters)
    result["report"] = Agent().report()["path"]
    # Explicit API-dependent extensions can be called after evidence is collected.
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(run_ai_scientist(), ensure_ascii=False, indent=2))
