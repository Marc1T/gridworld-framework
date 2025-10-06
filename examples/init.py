"""
Exemples d'utilisation du framework GridWorld RL.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gridworld_framework.examples.basic_usage import (   # type: ignore
    demo_random_agent,
    demo_policy_iteration, 
    demo_value_iteration,
    demo_monte_carlo,
    demo_q_learning,
    compare_all_agents
)

__all__ = [
    "demo_random_agent",
    "demo_policy_iteration",
    "demo_value_iteration", 
    "demo_monte_carlo",
    "demo_q_learning",
    "compare_all_agents"
]