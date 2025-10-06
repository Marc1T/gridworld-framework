"""
Agent qui sélectionne des actions aléatoires.

Cet agent est utile comme baseline pour comparer les performances
d'autres algorithmes plus sophistiqués.
"""

import numpy as np
from gridworld_framework.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent qui choisit des actions aléatoires uniformément.
    
    Cet agent ne apprend pas et sert principalement de référence
    pour évaluer la performance d'autres algorithmes.
    """
    
    def __init__(self, env, **kwargs):
        """
        Initialise l'agent aléatoire.
        
        Args:
            env: Environnement Gymnasium
            **kwargs: Arguments supplémentaires
        """
        super().__init__(env, **kwargs)
        
    def act(self, state: int, explore: bool = True) -> int:
        """
        Sélectionne une action aléatoire.
        
        Args:
            state: État courant (ignoré pour l'agent aléatoire)
            explore: Non utilisé pour l'agent aléatoire
            
        Returns:
            Action aléatoire
        """
        return np.random.randint(self.n_actions)
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool) -> None:
        """
        Ne fait rien car l'agent aléatoire n'apprend pas.
        
        Args:
            state: État (ignoré)
            action: Action (ignoré)
            reward: Récompense (ignoré)
            next_state: Prochain état (ignoré)
            done: Terminé (ignoré)
        """
        pass  # L'agent aléatoire n'apprend pas