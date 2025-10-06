"""
Fonctions pour calculer des métriques et comparer les agents.
"""

import numpy as np
from typing import Dict, List, Any
from gridworld_framework.agents.base_agent import BaseAgent


def calculate_metrics(agent: BaseAgent, env, n_episodes: int = 100) -> Dict[str, float]:
    """
    Calcule les métriques de performance d'un agent.
    
    Args:
        agent: Agent à évaluer
        env: Environnement
        n_episodes: Nombre d'épisodes d'évaluation
        
    Returns:
        Dictionnaire de métriques
    """
    results = agent.evaluate(n_episodes=n_episodes)
    
    # Métriques supplémentaires
    V = agent.get_value_function()
    policy = agent.get_policy()
    
    metrics = {
        'mean_reward': results['mean_reward'],
        'std_reward': results['std_reward'],
        'mean_length': results['mean_length'],
        'success_rate': np.mean([r > 0 for r in agent.rewards_history[-100:]]) if agent.rewards_history else 0,
        'value_range': np.ptp(V),  # Peak-to-peak
        'value_mean': np.mean(np.abs(V)),
        'policy_entropy': _calculate_policy_entropy(agent.policy),
        'convergence_speed': _estimate_convergence_speed(agent.convergence_history) if agent.convergence_history else 0
    }
    
    return metrics


def _calculate_policy_entropy(policy: np.ndarray) -> float:
    """
    Calcule l'entropie moyenne de la politique.
    
    Args:
        policy: Matrice de politique
        
    Returns:
        Entropie moyenne
    """
    if policy.ndim == 1:
        # Politique déterministe
        return 0.0
    
    entropy = 0
    for s in range(policy.shape[0]):
        for a in range(policy.shape[1]):
            if policy[s, a] > 0:
                entropy -= policy[s, a] * np.log(policy[s, a])
    
    return entropy / policy.shape[0]


def _estimate_convergence_speed(convergence_history: List[float]) -> float:
    """
    Estime la vitesse de convergence.
    
    Args:
        convergence_history: Historique de convergence
        
    Returns:
        Estimation de la vitesse
    """
    if len(convergence_history) < 2:
        return 0
    
    # Calculer les différences
    diffs = np.diff(convergence_history)
    
    # Trouver quand la convergence se stabilise
    threshold = np.max(np.abs(diffs)) * 0.01
    stable_point = np.where(np.abs(diffs) < threshold)[0]
    
    if len(stable_point) > 0:
        return stable_point[0] / len(convergence_history)
    else:
        return 1.0


def compare_agents(agents: Dict[str, BaseAgent], env, 
                  n_episodes: int = 100) -> Dict[str, Dict[str, float]]:
    """
    Compare plusieurs agents sur les mêmes métriques.
    
    Args:
        agents: Dictionnaire {nom: agent}
        env: Environnement
        n_episodes: Nombre d'épisodes d'évaluation
        
    Returns:
        Dictionnaire des métriques par agent
    """
    results = {}
    
    for name, agent in agents.items():
        print(f"Évaluation de {name}...")
        results[name] = calculate_metrics(agent, env, n_episodes)
    
    return results