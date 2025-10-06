"""
Implémentation de TD(0) avec approximation de fonction linéaire.

Cet agent utilise des features pour approximer la fonction de valeur V(s)
plutôt que de stocker une table complète. C'est essentiel pour les grands espaces d'états.
"""

import numpy as np
from typing import Dict
from gridworld_framework.agents.base_agent import BaseAgent

class TDLinearAgent(BaseAgent):
    """
    Agent TD(0) avec approximation de fonction linéaire (on-policy).
    
    Utilise des features pour représenter les états et des poids w pour
    approximer V(s) = φ(s)·w.
    
    Attributes:
        feature_dim (int): Dimension du vecteur de features
        w (np.ndarray): Vecteur de poids pour l'approximation linéaire
        features (np.ndarray): Matrice des features pour chaque état
        epsilon (float): Probabilité d'exploration
        epsilon_decay (float): Taux de décroissance de epsilon
        epsilon_min (float): Valeur minimale de epsilon
    """
    
    def __init__(self, env, feature_dim: int = 10, gamma: float = 0.99,
                 learning_rate: float = 0.01, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 **kwargs):
        """
        Initialise l'agent TD(0) linéaire.
        
        Args:
            env: Environnement Gymnasium
            feature_dim: Dimension des features
            gamma: Facteur d'actualisation
            learning_rate: Taux d'apprentissage pour les poids
            epsilon: Probabilité initiale d'exploration
            epsilon_decay: Taux de décroissance de epsilon
            epsilon_min: Valeur minimale de epsilon
            **kwargs: Arguments supplémentaires
        """
        super().__init__(env, gamma=gamma, learning_rate=learning_rate, **kwargs)
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialiser les poids
        self.w = np.random.randn(feature_dim) * 0.01
        
        # Générer les features pour tous les états
        self.features = self._create_features()
        
        # Pour TD(0), on a besoin de V(s) mais pas de Q(s,a) explicitement
        # La politique est toujours ε-greedy basée sur les valeurs d'action
        self._compute_action_values()
    
    def _create_features(self) -> np.ndarray:
        """
        Crée les features pour chaque état.
        
        Returns:
            Matrice de features (n_states, feature_dim)
        """
        features = np.zeros((self.n_states, self.feature_dim))
        
        for state in range(self.n_states):
            # Feature 0: constante (bias)
            features[state, 0] = 1.0
            
            # Features 1-2: position normalisée
            row, col = self._state_to_coords(state)
            features[state, 1] = row / (self.env.grid_shape[0] - 1)  # Ligne normalisée
            features[state, 2] = col / (self.env.grid_shape[1] - 1)  # Colonne normalisée
            
            # Feature 3: distance au but (Manhattan)
            goal_row, goal_col = self._state_to_coords(self.env.goal_state)
            dist_to_goal = abs(row - goal_row) + abs(col - goal_col)
            max_dist = sum(self.env.grid_shape) - 2  # Distance maximale possible
            features[state, 3] = 1.0 - (dist_to_goal / max_dist)  # 1 = au but, 0 = loin
            
            # Feature 4: est-ce un obstacle?
            features[state, 4] = 1.0 if state in self.env.obstacles else 0.0
            
            # Feature 5: est-ce le but?
            features[state, 5] = 1.0 if state == self.env.goal_state else 0.0
            
            # Features 6+: radial basis functions pour couvrir l'espace
            for i in range(6, self.feature_dim):
                # Centre aléatoire dans la grille
                center_row = np.random.randint(0, self.env.grid_shape[0])
                center_col = np.random.randint(0, self.env.grid_shape[1])
                
                # Distance euclidienne au centre
                dist = np.sqrt((row - center_row)**2 + (col - center_col)**2)
                features[state, i] = np.exp(-dist)
        
        return features
    
    def _state_to_coords(self, state: int) -> tuple:
        """Convertit un état en coordonnées (ligne, colonne)."""
        return state // self.env.grid_shape[1], state % self.env.grid_shape[1]
    
    def value(self, state: int) -> float:
        """
        Calcule V(s) = φ(s)·w.
        
        Args:
            state: État
            
        Returns:
            Valeur approximée de l'état
        """
        return np.dot(self.features[state], self.w)
    
    def _compute_action_values(self):
        """
        Calcule Q(s,a) pour tous les états et actions.
        Nécessaire pour la sélection d'actions.
        """
        self.Q = np.zeros((self.n_states, self.n_actions))
        
        for state in range(self.n_states):
            for action in range(self.n_actions):
                # Pour TD(0), on estime Q(s,a) = E[r + γV(s')]
                next_state_probs = self.env.P[state, action]
                expected_next_value = 0
                expected_reward = 0
                
                for next_state, prob in enumerate(next_state_probs):
                    if prob > 0:
                        expected_next_value += prob * self.value(next_state)
                        expected_reward += prob * self.env.R[state, action]
                
                self.Q[state, action] = expected_reward + self.gamma * expected_next_value
    
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
        Met à jour les poids avec la règle TD(0).
        
        Formule: w = w + α * [r + γV(s') - V(s)] * ∇w V(s)
        Avec V(s) = φ(s)·w, donc ∇w V(s) = φ(s)
        
        Args:
            state: État
            action: Action
            reward: Récompense
            next_state: Prochain état
            done: Si l'épisode est terminé
        """
        # Calculer la cible TD
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.value(next_state)
        
        # Erreur TD
        td_error = target - self.value(state)
        
        # Mise à jour des poids
        self.w += self.learning_rate * td_error * self.features[state]
        
        # Mettre à jour les valeurs Q
        self._compute_action_values()
        
        # Mettre à jour la politique (gloutonne par rapport à Q)
        best_action = np.argmax(self.Q[state])
        self.policy[state] = np.eye(self.n_actions)[best_action]
    
    def train(self, n_episodes: int = 1000, max_steps: int = 1000,
              verbose: bool = True) -> Dict:
        """
        Entraîne l'agent avec TD(0).
        
        Args:
            n_episodes: Nombre d'épisodes d'entraînement
            max_steps: Nombre maximal de pas par épisode
            verbose: Afficher les progrès
            
        Returns:
            Historique des métriques
        """
        self.rewards_history = []
        self.convergence_history = []
        self.td_errors_history = []
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            episode_td_errors = []
            
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Sauvegarder l'erreur TD avant la mise à jour
                old_value = self.value(state)
                next_value = self.value(next_state) if not done else 0
                td_error_before = reward + self.gamma * next_value - old_value
                episode_td_errors.append(abs(td_error_before))
                
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
            self.convergence_history.append(np.max(self.w))  # Suivre les poids
            self.td_errors_history.append(np.mean(episode_td_errors))
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])
                avg_td_error = np.mean(self.td_errors_history[-100:])
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Reward moyen: {avg_reward:.2f}, "
                      f"TD error moyen: {avg_td_error:.4f}, "
                      f"Epsilon: {self.epsilon:.4f}, "
                      f"||w||: {np.linalg.norm(self.w):.4f}")
        
        return {
            'rewards': self.rewards_history,
            'convergence': self.convergence_history,
            'td_errors': self.td_errors_history
        }
    
    def get_value_function(self) -> np.ndarray:
        """
        Retourne la fonction de valeur approximée.
        
        Returns:
            V[s] pour tous les états
        """
        V = np.zeros(self.n_states)
        for state in range(self.n_states):
            V[state] = self.value(state)
        return V
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Retourne l'importance des features basée sur les poids.
        
        Returns:
            Importance normalisée des features
        """
        return np.abs(self.w) / np.sum(np.abs(self.w))