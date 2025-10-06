"""
Implémentation de l'algorithme Q-learning.

Q-learning est un algorithme TD (Temporal Difference) off-policy
qui apprend directement la fonction de valeur-action optimale.
"""

import numpy as np
from typing import Dict
from gridworld_framework.agents.base_agent import BaseAgent # type: ignore


class QLearningAgent(BaseAgent):
    """
    Agent utilisant l'algorithme Q-learning avec exploration ε-greedy.
    
    Attributes:
        epsilon (float): Probabilité d'exploration
        epsilon_decay (float): Taux de décroissance de epsilon
        epsilon_min (float): Valeur minimale de epsilon
    """
    
    def __init__(self, env, gamma: float = 0.99, learning_rate: float = 0.1,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, **kwargs):
        """
        Initialise l'agent Q-learning.
        
        Args:
            env: Environnement Gymnasium
            gamma: Facteur d'actualisation
            learning_rate: Taux d'apprentissage
            epsilon: Probabilité initiale d'exploration
            epsilon_decay: Taux de décroissance de epsilon
            epsilon_min: Valeur minimale de epsilon
            **kwargs: Arguments supplémentaires
        """
        super().__init__(env, gamma=gamma, learning_rate=learning_rate, **kwargs)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    def act(self, state: int, explore: bool = True) -> int:
        """
        Sélectionne une action avec stratégie ε-greedy.
        
        Args:
            state: État courant
            explore: Si True, utilise l'exploration
            
        Returns:
            Action sélectionnée
        """
        if explore and np.random.random() < self.epsilon:
            # Exploration: action aléatoire
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: meilleure action selon Q
            return np.argmax(self.Q[state]) # type: ignore
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool) -> None:
        """
        Met à jour la fonction Q avec la règle de Q-learning.
        
        Formule: Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: État
            action: Action
            reward: Récompense
            next_state: Prochain état
            done: Si l'épisode est terminé
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        # Mise à jour Q-learning
        self.Q[state, action] += self.learning_rate * (target - self.Q[state, action])
        
        # Mettre à jour la politique (gloutonne par rapport à Q)
        best_action = np.argmax(self.Q[state])
        self.policy[state] = np.eye(self.n_actions)[best_action]
    
    def train(self, n_episodes: int = 1000, max_steps: int = 1000,
              verbose: bool = True) -> Dict:
        """
        Entraîne l'agent avec Q-learning.
        
        Args:
            n_episodes: Nombre d'épisodes d'entraînement
            max_steps: Nombre maximal de pas par épisode
            verbose: Afficher les progrès
            
        Returns:
            Historique des métriques
        """
        self.rewards_history = []
        self.convergence_history = []
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.update(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # Décroissance de epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Enregistrement des métriques
            self.rewards_history.append(total_reward)
            self.convergence_history.append(np.max(self.Q))
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])
                avg_q = np.mean(self.convergence_history[-100:])
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Reward moyen: {avg_reward:.2f}, "
                      f"Q max: {avg_q:.4f}, "
                      f"Epsilon: {self.epsilon:.4f}")
        
        return {
            'rewards': self.rewards_history,
            'convergence': self.convergence_history
        }