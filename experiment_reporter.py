"""Compatibility wrapper: reports come from the persistent experiment ledger."""
from ngc_agent.service import Agent


def generate_experiment_report(target_f1=0.9):
    return Agent().report()
