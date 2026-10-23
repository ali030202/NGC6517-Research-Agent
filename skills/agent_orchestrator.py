"""Compatibility entrypoint backed by the unified dependency-light service."""
from ngc_agent.service import Agent


def handle_message(message, history=None):
    agent = Agent()
    result = agent.chat(message)
    return {**result, "mode": "local", "stage": "ready",
            "update_panels": {"status": str(agent.capabilities()), "jobs": agent.store.jobs(20)}}
