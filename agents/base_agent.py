"""
Classe de base pour tous les agents de Reinforcement Learning.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Optional
import gymnasium as gym


class BaseAgent(ABC):
    """
    Classe abstraite de base pour tous les agents RL.
    
    Attributes:
        env (gym.Env): Environnement
        n_states (int): Nombre d'états
        n_actions (int): Nombre d'actions
        gamma (float): Facteur d'actualisation
        learning_rate (float): Taux d'apprentissage
        policy (np.ndarray): Politique π[s, a]
        V (np.ndarray): Fonction de valeur V[s]
        Q (np.ndarray): Fonction de valeur-action Q[s, a]
    """
    
    def __init__(self, env: gym.Env, gamma: float = 0.99, 
                 learning_rate: float = 0.1, **kwargs):
        """
        Initialise l'agent.
        
        Args:
            env: Environnement Gymnasium
            gamma: Facteur d'actualisation
            learning_rate: Taux d'apprentissage
            **kwargs: Arguments supplémentaires
        """
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Initialiser les structures de données
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        self.V = np.zeros(self.n_states)
        self.Q = np.zeros((self.n_states, self.n_actions))
        
        # Historique pour visualisation
        self.rewards_history = []
        self.convergence_history = []
        
    @abstractmethod
    def act(self, state: int, explore: bool = True) -> int:
        """
        Sélectionne une action selon la politique.
        
        Args:
            state: État courant
            explore: Si True, exploration; sinon exploitation
            
        Returns:
            Action sélectionnée
        """
        pass
    
    @abstractmethod
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool) -> None:
        """
        Met à jour l'agent avec l'expérience (s, a, r, s').
        
        Args:
            state: État
            action: Action
            reward: Récompense
            next_state: Prochain état
            done: Si l'épisode est terminé
        """
        pass
    
    def train(self, n_episodes: int = 1000, 
              max_steps: int = 1000,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Entraîne l'agent sur plusieurs épisodes.
        
        Args:
            n_episodes: Nombre d'épisodes d'entraînement
            max_steps: Nombre maximal de pas par épisode
            verbose: Afficher les progrès
            
        Returns:
            Historique des métriques
        """
        self.rewards_history = []
        episode_lengths = []
        
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
            
            self.rewards_history.append(total_reward)
            episode_lengths.append(steps)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Reward moyen: {avg_reward:.2f}, "
                      f"Longueur moyenne: {np.mean(episode_lengths[-100:]):.2f}")
        
        return {
            'rewards': self.rewards_history,
            'lengths': episode_lengths
        }
    
    def evaluate(self, n_episodes: int = 100, 
                 max_steps: int = 1000) -> Dict[str, float]:
        """
        Évalue la performance de l'agent.
        
        Args:
            n_episodes: Nombre d'épisodes d'évaluation
            max_steps: Nombre maximal de pas par épisode
            
        Returns:
            Métriques d'évaluation
        """
        total_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                action = self.act(state, explore=False)  # Pas d'exploration
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }
    
    def get_policy(self) -> np.ndarray:
        """
        Retourne la politique optimale.
        
        Returns:
            Politique π[s] (action optimale pour chaque état)
        """
        return np.argmax(self.policy, axis=1)
    
    def get_value_function(self) -> np.ndarray:
        """
        Retourne la fonction de valeur.
        
        Returns:
            Fonction de valeur V[s]
        """
        return self.V.copy()
    
    def get_q_function(self) -> np.ndarray:
        """
        Retourne la fonction de valeur-action.
        
        Returns:
            Fonction Q[s, a]
        """
        return self.Q.copy()
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde les paramètres de l'agent.
        
        Args:
            filepath: Chemin du fichier
        """
        np.savez(filepath,
                 policy=self.policy,
                 V=self.V,
                 Q=self.Q,
                 gamma=self.gamma,
                 learning_rate=self.learning_rate)
    
    def load(self, filepath: str) -> None:
        """
        Charge les paramètres de l'agent.
        
        Args:
            filepath: Chemin du fichier
        """
        data = np.load(filepath)
        self.policy = data['policy']
        self.V = data['V']
        self.Q = data['Q']
        self.gamma = data['gamma']
        self.learning_rate = data['learning_rate']