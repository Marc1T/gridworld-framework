"""
Implémentation de l'algorithme Value Iteration.

Value Iteration calcule directement la fonction de valeur optimale
en utilisant l'équation d'optimalité de Bellman.
"""

import numpy as np
from typing import Dict
from gridworld_framework.agents.base_agent import BaseAgent


class ValueIterationAgent(BaseAgent):
    """
    Agent utilisant l'algorithme Value Iteration.
    
    Attributes:
        theta (float): Seuil de convergence
        max_iterations (int): Nombre maximal d'itérations
    """
    
    def __init__(self, env, gamma: float = 0.99, theta: float = 1e-6,
                 max_iterations: int = 1000, **kwargs):
        """
        Initialise l'agent Value Iteration.
        
        Args:
            env: Environnement Gymnasium
            gamma: Facteur d'actualisation
            theta: Seuil de convergence
            max_iterations: Nombre maximal d'itérations
            **kwargs: Arguments supplémentaires
        """
        super().__init__(env, gamma=gamma, **kwargs)
        self.theta = theta
        self.max_iterations = max_iterations
        
        # Récupérer les matrices MDP de l'environnement
        self.P = env.get_transition_matrix()
        self.R = env.get_reward_matrix()
        
    def value_iteration(self) -> None:
        """
        Exécute l'algorithme Value Iteration.
        
        Met à jour la fonction de valeur V en utilisant l'équation
        d'optimalité de Bellman jusqu'à convergence.
        """
        for i in range(self.max_iterations):
            delta = 0
            for s in range(self.n_states):
                v_old = self.V[s]
                
                # Calculer les valeurs Q pour chaque action
                q_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for next_s in range(self.n_states):
                        q_values[a] += self.P[s, a, next_s] * (
                            self.R[s, a] + self.gamma * self.V[next_s]
                        )
                
                # Mettre à jour V avec la valeur maximale
                self.V[s] = np.max(q_values)
                delta = max(delta, abs(v_old - self.V[s]))
            
            # Enregistrer la convergence
            self.convergence_history.append(np.max(self.V))
            
            if delta < self.theta:
                break
    
    def extract_policy(self) -> None:
        """
        Extrait la politique optimale à partir de la fonction de valeur V.
        """
        for s in range(self.n_states):
            q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for next_s in range(self.n_states):
                    q_values[a] += self.P[s, a, next_s] * (
                        self.R[s, a] + self.gamma * self.V[next_s]
                    )
            
            # Politique gloutonne
            best_action = np.argmax(q_values)
            self.policy[s] = np.eye(self.n_actions)[best_action]
            self.Q[s] = q_values
    
    def train(self, n_episodes: int = None, verbose: bool = True) -> Dict:
        """
        Entraîne l'agent avec Value Iteration.
        
        Args:
            n_episodes: Non utilisé (conservé pour compatibilité)
            verbose: Afficher les progrès
            
        Returns:
            Historique de convergence
        """
        self.convergence_history = []
        
        if verbose:
            print("Début de Value Iteration...")
        
        self.value_iteration()
        self.extract_policy()
        
        if verbose:
            print(f"Value Iteration terminé. V max: {np.max(self.V):.4f}")
            print(f"Nombre d'itérations: {len(self.convergence_history)}")
        
        return {'convergence': self.convergence_history}
    
    def act(self, state: int, explore: bool = True) -> int:
        """
        Sélectionne une action selon la politique optimale.
        
        Args:
            state: État courant
            explore: Non utilisé pour Value Iteration
            
        Returns:
            Action optimale
        """
        return np.argmax(self.policy[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool) -> None:
        """
        Non utilisé pour Value Iteration (algorithme hors-ligne).
        
        Args:
            state: État (ignoré)
            action: Action (ignoré)
            reward: Récompense (ignoré)
            next_state: Prochain état (ignoré)
            done: Terminé (ignoré)
        """
        pass  # Value Iteration est un algorithme hors-ligne