"""
Implémentation de l'algorithme Policy Iteration.

Policy Iteration alterne entre l'évaluation de politique (Policy Evaluation)
et l'amélioration de politique (Policy Improvement) jusqu'à convergence.
"""

import numpy as np
from typing import Dict
from gridworld_framework.agents.base_agent import BaseAgent


class PolicyIterationAgent(BaseAgent):
    """
    Agent utilisant l'algorithme Policy Iteration.
    
    Attributes:
        theta (float): Seuil de convergence pour l'évaluation
        max_iterations (int): Nombre maximal d'itérations
    """
    
    def __init__(self, env, gamma: float = 0.99, theta: float = 1e-6,
                 max_iterations: int = 1000, **kwargs):
        """
        Initialise l'agent Policy Iteration.
        
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
        
    def policy_evaluation(self) -> None:
        """
        Évalue la politique courante.
        
        Calcule la fonction de valeur V pour la politique π courante
        en résolvant le système d'équations de Bellman.
        """
        while True:
            delta = 0
            for s in range(self.n_states):
                v_old = self.V[s]
                # Valeur attendue selon la politique
                new_v = 0
                for a in range(self.n_actions):
                    # Probabilité de prendre l'action a dans l'état s
                    prob_a = self.policy[s, a]
                    # Récompense attendue
                    expected_reward = 0
                    for next_s in range(self.n_states):
                        expected_reward += self.P[s, a, next_s] * (
                            self.R[s, a] + self.gamma * self.V[next_s]
                        )
                    new_v += prob_a * expected_reward
                
                self.V[s] = new_v
                delta = max(delta, abs(v_old - self.V[s]))
            
            if delta < self.theta:
                break
    
    def policy_improvement(self) -> bool:
        """
        Améliore la politique en la rendant gloutonne par rapport à V.
        
        Returns:
            True si la politique a changé, False sinon (convergence)
        """
        policy_stable = True
        
        for s in range(self.n_states):
            old_action = np.argmax(self.policy[s])
            
            # Calculer les valeurs Q pour chaque action
            q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for next_s in range(self.n_states):
                    q_values[a] += self.P[s, a, next_s] * (
                        self.R[s, a] + self.gamma * self.V[next_s]
                    )
            
            # Trouver l'action optimale
            best_actions = np.where(q_values == np.max(q_values))[0]
            best_action = np.random.choice(best_actions)
            
            # Mettre à jour la politique
            new_policy = np.zeros(self.n_actions)
            new_policy[best_action] = 1.0
            
            if old_action != best_action:
                policy_stable = False
                
            self.policy[s] = new_policy
        
        return not policy_stable
    
    def train(self, n_episodes: int = None, verbose: bool = True) -> Dict:
        """
        Entraîne l'agent avec Policy Iteration.
        
        Args:
            n_episodes: Non utilisé (conservé pour compatibilité)
            verbose: Afficher les progrès
            
        Returns:
            Historique de convergence
        """
        self.convergence_history = []
        
        for i in range(self.max_iterations):
            # Évaluation de politique
            self.policy_evaluation()
            
            # Amélioration de politique
            policy_changed = self.policy_improvement()
            
            # Enregistrer la convergence
            self.convergence_history.append(np.max(self.V))
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}, V max: {np.max(self.V):.4f}")
            
            if not policy_changed:
                if verbose:
                    print(f"Convergence atteinte après {i + 1} itérations")
                break
        else:
            if verbose:
                print(f"Maximum d'itérations ({self.max_iterations}) atteint")
        
        # Mettre à jour Q avec les valeurs finales
        self._update_q_function()
        
        return {'convergence': self.convergence_history}
    
    def _update_q_function(self) -> None:
        """Met à jour la fonction Q basée sur V et la politique."""
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.Q[s, a] = 0
                for next_s in range(self.n_states):
                    self.Q[s, a] += self.P[s, a, next_s] * (
                        self.R[s, a] + self.gamma * self.V[next_s]
                    )
    
    def act(self, state: int, explore: bool = True) -> int:
        """
        Sélectionne une action selon la politique optimale.
        
        Args:
            state: État courant
            explore: Non utilisé pour Policy Iteration
            
        Returns:
            Action optimale
        """
        return np.argmax(self.policy[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool) -> None:
        """
        Non utilisé pour Policy Iteration (algorithme hors-ligne).
        
        Args:
            state: État (ignoré)
            action: Action (ignoré)
            reward: Récompense (ignoré)
            next_state: Prochain état (ignoré)
            done: Terminé (ignoré)
        """
        pass  # Policy Iteration est un algorithme hors-ligne