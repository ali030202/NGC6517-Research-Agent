#!/usr/bin/env python3
"""Dependency-light command line entrypoint; imports no Torch, UI or LLM SDK."""
import argparse
import json
from pathlib import Path
import sys

from ngc_agent.config import ROOT, TrainConfig
from ngc_agent.service import Agent


def main():
    parser = argparse.ArgumentParser(description="NGC6517 Research Agent")
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--include-test", action="store_true")
    inspect.add_argument("--config", type=Path)
    for name in ("plan", "run-plan", "train"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--config", type=Path)
        if name != "train":
            cmd.add_argument("--budget", type=int, default=3)
            cmd.add_argument("--target-f1", type=float, default=0.9)
        if name != "plan":
            cmd.add_argument("--proposal")
    for name in ("job", "logs", "cancel", "retry", "recover", "evaluate", "events", "_worker"):
        cmd = commands.add_parser(name)
        cmd.add_argument("job_id")
    report = commands.add_parser("report")
    report.add_argument("--job-id")
    search = commands.add_parser("search")
    search.add_argument("query")
    propose = commands.add_parser("propose")
    propose.add_argument("direction")
    stage = commands.add_parser("stage")
    stage.add_argument("file", type=Path)
    stage.add_argument("--reason", default="Manual proposal")
    apply = commands.add_parser("apply")
    apply.add_argument("proposal_id")
    rollback = commands.add_parser("rollback")
    rollback.add_argument("sha256")
    args = parser.parse_args()
    try:
        agent = Agent(args.root)
        config_path = getattr(args, "config", None)
        config = TrainConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8-sig"))) if config_path else TrainConfig()
        command = args.command
        if command == "status":
            result = {"capabilities": agent.capabilities(), "jobs": agent.store.jobs()}
        elif command == "inspect":
            result = agent.inspect(config, args.include_test)
        elif command == "plan":
            result = agent.plan(config, args.budget, args.target_f1)
        elif command == "run-plan":
            result = agent.run_plan(config, args.budget, args.target_f1, args.proposal)
        elif command == "train":
            result = agent.submit(config, args.proposal)
        elif command == "job":
            result = agent.store.get(args.job_id)
        elif command == "logs":
            print(agent.logs(args.job_id))
            return
        elif command == "cancel":
            agent.store.cancel(args.job_id)
            result = agent.store.get(args.job_id)
        elif command in {"retry", "recover", "evaluate"}:
            result = getattr(agent, command)(args.job_id)
        elif command == "events":
            result = agent.store.events(args.job_id)
        elif command == "report":
            result = agent.report(args.job_id)
        elif command == "search":
            result = agent.knowledge(args.query)
        elif command == "propose":
            result = agent.propose(args.direction)
        elif command == "stage":
            result = agent.code.stage(args.file.read_text(encoding="utf-8-sig"), args.reason)
        elif command == "apply":
            result = agent.code.apply(args.proposal_id)
        elif command == "rollback":
            result = agent.code.rollback(args.sha256)
        elif command == "_worker":
            from ngc_agent.runner import run_job
            run_job(args.root, args.job_id)
            return
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
