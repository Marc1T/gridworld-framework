"""
GridWorld Reinforcement Learning Framework

Un framework modulaire pour l'apprentissage par renforcement basé sur des environnements GridWorld,
inspiré de Gymnasium.
"""

__version__ = "0.1.0"
__author__ = "NANKOULI Marc Thierry"

from gridworld_framework.core.gridworld_env import GridWorldEnv
from gridworld_framework.core.mdp import MDP

# Agents
from gridworld_framework.agents.random_agent import RandomAgent
from gridworld_framework.agents.policy_iteration import PolicyIterationAgent
from gridworld_framework.agents.value_iteration import ValueIterationAgent
from gridworld_framework.agents.monte_carlo import MonteCarloAgent
from gridworld_framework.agents.q_learning import QLearningAgent
from gridworld_framework.agents.td_linear import TDLinearAgent

__all__ = [
    "GridWorldEnv",
    "MDP",
    "RandomAgent", 
    "PolicyIterationAgent",
    "ValueIterationAgent",
    "MonteCarloAgent",
    "QLearningAgent",
    "TDLinearAgent"
]