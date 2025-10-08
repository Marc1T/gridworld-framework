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
            gamma: Facteur d'actualisation (entre 0 et 1)
        
        Raises:
            ValueError: Si les paramètres sont invalides
        """
        if n_states <= 0 or n_actions <= 0:
            raise ValueError("n_states et n_actions doivent être positifs")
        
        if not 0 <= gamma <= 1:
            raise ValueError("gamma doit être entre 0 et 1")
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        
        # Initialiser les matrices si non fournies
        if P is None:
            # Initialisation uniforme par défaut
            self.P = np.ones((n_states, n_actions, n_states)) / n_states
        else:
            if P.shape != (n_states, n_actions, n_states):
                raise ValueError(f"P doit avoir la forme ({n_states}, {n_actions}, {n_states})")
            self.P = P.copy()
            
        if R is None:
            self.R = np.zeros((n_states, n_actions))
        else:
            if R.shape not in [(n_states, n_actions), (n_states, n_actions, n_states)]:
                raise ValueError(f"R doit avoir la forme ({n_states}, {n_actions}) ou ({n_states}, {n_actions}, {n_states})")
            self.R = R.copy()
    
    def validate(self) -> Tuple[bool, str]:
        """
        Valide les matrices du MDP.
        
        Returns:
            Tuple (is_valid, error_message)
        """
        # Vérifier les dimensions
        if self.P.shape != (self.n_states, self.n_actions, self.n_states):
            return False, f"P a la mauvaise forme: {self.P.shape}"
            
        if self.R.shape not in [(self.n_states, self.n_actions), 
                               (self.n_states, self.n_actions, self.n_states)]:
            return False, f"R a la mauvaise forme: {self.R.shape}"
        
        # Vérifier que les probabilités sont valides
        if np.any(self.P < 0) or np.any(self.P > 1):
            return False, "P contient des probabilités invalides (hors [0,1])"
        
        # Vérifier que les probabilités somment à 1
        for s in range(self.n_states):
            for a in range(self.n_actions):
                prob_sum = self.P[s, a].sum()
                if not np.isclose(prob_sum, 1.0, atol=1e-6):
                    return False, f"P[{s},{a}] ne somme pas à 1 (somme={prob_sum:.6f})"
        
        return True, "MDP valide"
    
    def get_expected_reward(self, state: int, action: int) -> float:
        """
        Calcule la récompense attendue pour un état et une action.
        
        Args:
            state: État (0 <= state < n_states)
            action: Action (0 <= action < n_actions)
            
        Returns:
            Récompense attendue E[R(s,a)]
        
        Raises:
            ValueError: Si state ou action sont invalides
        """
        if not (0 <= state < self.n_states):
            raise ValueError(f"État invalide: {state}")
        if not (0 <= action < self.n_actions):
            raise ValueError(f"Action invalide: {action}")
        
        if self.R.ndim == 2:
            # R[s, a] - récompense déterministe
            return self.R[state, action]
        else:
            # R[s, a, s'] - récompense dépendante de l'état suivant
            return np.sum(self.P[state, action] * self.R[state, action])
    
    def get_transition_prob(self, state: int, action: int, next_state: int) -> float:
        """
        Retourne P(s'|s,a).
        
        Args:
            state: État courant
            action: Action
            next_state: État suivant
            
        Returns:
            Probabilité de transition
        """
        return self.P[state, action, next_state]
    
    def sample_next_state(self, state: int, action: int) -> int:
        """
        Échantillonne le prochain état selon P(s'|s,a).
        
        Args:
            state: État courant
            action: Action
            
        Returns:
            Prochain état échantillonné
        """
        return np.random.choice(self.n_states, p=self.P[state, action])
    
    @classmethod
    def from_gridworld_env(cls, env) -> 'MDP':
        """
        Crée un MDP à partir d'un environnement GridWorld.
        
        Args:
            env: Environnement GridWorld
            
        Returns:
            Instance MDP
        """
        n_states = env.n_states
        n_actions = env.action_space.n
        P = env.get_transition_matrix()
        R = env.get_reward_matrix()
        
        mdp = cls(n_states, n_actions, P, R, gamma=0.99)
        
        # Valider le MDP créé
        is_valid, msg = mdp.validate()
        if not is_valid:
            raise ValueError(f"MDP invalide créé depuis l'environnement: {msg}")
        
        return mdp
    
    def get_info(self) -> Dict[str, Any]:
        """
        Retourne les informations du MDP.
        
        Returns:
            Dictionnaire avec les informations du MDP
        """
        is_valid, msg = self.validate()
        
        return {
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'gamma': self.gamma,
            'is_valid': is_valid,
            'validation_message': msg,
            'reward_type': 'R(s,a)' if self.R.ndim == 2 else 'R(s,a,s\')',
            'min_reward': np.min(self.R),
            'max_reward': np.max(self.R),
            'mean_reward': np.mean(self.R)
        }
    
    def __repr__(self) -> str:
        """Représentation string du MDP."""
        return f"MDP(n_states={self.n_states}, n_actions={self.n_actions}, gamma={self.gamma})"