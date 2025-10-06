"""
Classe pour représenter un Processus de Décision Markovien (MDP).

Cette classe encapsule les matrices de transition et de récompense
et fournit des méthodes pour résoudre le MDP.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


class MDP:
    """
    Représente un Processus de Décision Markovien.
    
    Attributes:
        n_states (int): Nombre d'états
        n_actions (int): Nombre d'actions
        P (np.ndarray): Matrice de transition P[s, a, s']
        R (np.ndarray): Matrice de récompense R[s, a] ou R[s, a, s']
        gamma (float): Facteur d'actualisation
    """
    
    def __init__(self, n_states: int, n_actions: int, 
                 P: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 gamma: float = 0.99):
        """
        Initialise le MDP.
        
        Args:
            n_states: Nombre d'états
            n_actions: Nombre d'actions
            P: Matrice de transition (n_states, n_actions, n_states)
            R: Matrice de récompense (n_states, n_actions) ou (n_states, n_actions, n_states)
            gamma: Facteur d'actualisation
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        
        # Initialiser les matrices si non fournies
        if P is None:
            self.P = np.zeros((n_states, n_actions, n_states))
        else:
            self.P = P
            
        if R is None:
            self.R = np.zeros((n_states, n_actions))
        else:
            self.R = R
            
    def validate(self) -> bool:
        """
        Valide les matrices du MDP.
        
        Returns:
            True si le MDP est valide, False sinon
        """
        # Vérifier les dimensions
        if self.P.shape != (self.n_states, self.n_actions, self.n_states):
            return False
            
        if self.R.shape not in [(self.n_states, self.n_actions), 
                               (self.n_states, self.n_actions, self.n_states)]:
            return False
            
        # Vérifier que les probabilités somment à 1
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if not np.isclose(self.P[s, a].sum(), 1.0):
                    return False
                    
        return True
    
    def get_expected_reward(self, state: int, action: int) -> float:
        """
        Calcule la récompense attendue pour un état et une action.
        
        Args:
            state: État
            action: Action
            
        Returns:
            Récompense attendue
        """
        if self.R.ndim == 2:
            return self.R[state, action]
        else:
            # R[s, a, s']
            return np.sum(self.P[state, action] * self.R[state, action])
    
    def from_gridworld_env(self, env) -> 'MDP':
        """
        Crée un MDP à partir d'un environnement GridWorld.
        
        Args:
            env: Environnement GridWorld
            
        Returns:
            Instance MDP
        """
        self.n_states = env.n_states
        self.n_actions = env.action_space.n
        self.P = env.get_transition_matrix()
        self.R = env.get_reward_matrix()
        
        return self