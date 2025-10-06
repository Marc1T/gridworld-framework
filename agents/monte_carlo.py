"""
Implémentation de l'algorithme Monte Carlo.

Les méthodes Monte Carlo apprennent directement à partir d'épisodes complets
sans avoir besoin d'un modèle de l'environnement.
"""

import numpy as np
from typing import Dict
from collections import defaultdict
from gridworld_framework.agents.base_agent import BaseAgent


class MonteCarloAgent(BaseAgent):
    """
    Agent utilisant la méthode Monte Carlo avec exploration ε-greedy.
    
    Attributes:
        epsilon (float): Probabilité d'exploration
        epsilon_decay (float): Taux de décroissance de epsilon
        epsilon_min (float): Valeur minimale de epsilon
        first_visit (bool): Si True, utilise First-Visit MC, sinon Every-Visit
        returns (defaultdict): Somme des retours pour chaque (s, a)
        returns_count (defaultdict): Nombre de visites pour chaque (s, a)
    """
    
    def __init__(self, env, gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 first_visit: bool = True, **kwargs):
        """
        Initialise l'agent Monte Carlo.
        
        Args:
            env: Environnement Gymnasium
            gamma: Facteur d'actualisation
            epsilon: Probabilité initiale d'exploration
            epsilon_decay: Taux de décroissance de epsilon
            epsilon_min: Valeur minimale de epsilon
            first_visit: Utiliser First-Visit ou Every-Visit MC
            **kwargs: Arguments supplémentaires
        """
        super().__init__(env, gamma=gamma, **kwargs)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.first_visit = first_visit
        
        # Pour stocker les retours
        self.returns = defaultdict(float)
        self.returns_count = defaultdict(int)
        
        # Pour stocker l'épisode courant
        self.episode = []
    
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
            return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool) -> None:
        """
        Stocke la transition dans l'épisode courant.
        
        Args:
            state: État
            action: Action
            reward: Récompense
            next_state: Prochain état
            done: Si l'épisode est terminé
        """
        self.episode.append((state, action, reward))
    
    def update_policy(self) -> None:
        """
        Met à jour la politique et la fonction Q à partir de l'épisode complet.
        """
        if not self.episode:
            return
            
        # Calculer les retours
        G = 0
        visited = set() if self.first_visit else None
        
        # Parcourir l'épisode à l'envers
        for t in range(len(self.episode) - 1, -1, -1):
            state, action, reward = self.episode[t]
            G = self.gamma * G + reward
            
            # Pour First-Visit, ne considérer que la première visite
            state_action = (state, action)
            if self.first_visit:
                if state_action not in visited:
                    visited.add(state_action)
                    self._update_q_value(state, action, G)
            else:
                # Every-Visit: mettre à jour à chaque visite
                self._update_q_value(state, action, G)
    
    def _update_q_value(self, state: int, action: int, G: float) -> None:
        """
        Met à jour la valeur Q(s, a) avec le retour G.
        
        Args:
            state: État
            action: Action
            G: Retour actualisé
        """
        state_action = (state, action)
        
        # Mise à jour incrémentale
        self.returns_count[state_action] += 1
        count = self.returns_count[state_action]
        
        # Q(s,a) = Q(s,a) + (1/N(s,a)) * (G - Q(s,a))
        self.Q[state, action] += (G - self.Q[state, action]) / count
        
        # Mettre à jour la politique (gloutonne par rapport à Q)
        best_action = np.argmax(self.Q[state])
        self.policy[state] = np.eye(self.n_actions)[best_action]
    
    def train(self, n_episodes: int = 1000, max_steps: int = 1000,
              verbose: bool = True) -> Dict:
        """
        Entraîne l'agent avec Monte Carlo.
        
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
            # Réinitialiser l'épisode
            self.episode = []
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            
            # Générer un épisode
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
            
            # Mettre à jour la politique avec l'épisode complet
            self.update_policy()
            
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