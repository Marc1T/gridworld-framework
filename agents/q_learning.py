"""
Implémentation de l'algorithme Q-learning.

Q-learning est un algorithme TD (Temporal Difference) off-policy
qui apprend directement la fonction de valeur-action optimale.
"""

import numpy as np
from typing import Dict, Optional
from gridworld_framework.agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Agent utilisant l'algorithme Q-learning avec exploration ε-greedy.
    
    Attributes:
        epsilon (float): Probabilité d'exploration
        epsilon_decay (float): Taux de décroissance de epsilon
        epsilon_min (float): Valeur minimale de epsilon
        use_double_q (bool): Utiliser Double Q-Learning
    """
    
    def __init__(self, env, gamma: float = 0.99, learning_rate: float = 0.1,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, use_double_q: bool = False, **kwargs):
        """
        Initialise l'agent Q-learning.
        
        Args:
            env: Environnement Gymnasium
            gamma: Facteur d'actualisation
            learning_rate: Taux d'apprentissage (alpha)
            epsilon: Probabilité initiale d'exploration
            epsilon_decay: Taux de décroissance de epsilon
            epsilon_min: Valeur minimale de epsilon
            use_double_q: Si True, utilise Double Q-Learning
            **kwargs: Arguments supplémentaires
        """
        super().__init__(env, gamma=gamma, learning_rate=learning_rate, **kwargs)
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_double_q = use_double_q
        
        # Pour Double Q-Learning
        if use_double_q:
            self.Q2 = np.zeros((self.n_states, self.n_actions))
        
        # Tracking pour analyses
        self.epsilon_history = []
        self.td_errors = []
    
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
            if self.use_double_q:
                # Pour Double Q-Learning, moyenne des deux Q
                q_avg = (self.Q[state] + self.Q2[state]) / 2
                return np.argmax(q_avg)
            else:
                return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool) -> None:
        """
        Met à jour la fonction Q avec la règle de Q-learning.
        
        Formule: Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: État
            action: Action
            reward: Récompense
            next_state: Prochain état
            done: Si l'épisode est terminé
        """
        if self.use_double_q:
            # Double Q-Learning: alterner entre Q1 et Q2
            if np.random.random() < 0.5:
                # Mettre à jour Q1
                best_next_action = np.argmax(self.Q[next_state])
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * self.Q2[next_state, best_next_action]
                
                td_error = target - self.Q[state, action]
                self.Q[state, action] += self.learning_rate * td_error
            else:
                # Mettre à jour Q2
                best_next_action = np.argmax(self.Q2[next_state])
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * self.Q[next_state, best_next_action]
                
                td_error = target - self.Q2[state, action]
                self.Q2[state, action] += self.learning_rate * td_error
        else:
            # Q-Learning standard
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.Q[next_state])
            
            # Calculer l'erreur TD
            td_error = target - self.Q[state, action]
            
            # Mise à jour Q-learning
            self.Q[state, action] += self.learning_rate * td_error
            
            # Enregistrer l'erreur TD pour analyse
            self.td_errors.append(abs(td_error))
        
        # Mettre à jour V (pour compatibilité)
        self.V[state] = np.max(self.Q[state])
        
        # Mettre à jour la politique (gloutonne par rapport à Q)
        best_action = np.argmax(self.Q[state])
        self.policy[state] = np.zeros(self.n_actions)
        self.policy[state, best_action] = 1.0
    
    def train(self, n_episodes: int = 1000, max_steps: int = 1000,
              verbose: bool = True, save_frequency: int = 100,
              checkpoint_dir: Optional[str] = None,
              track_q_history: bool = False) -> Dict:
        """
        Entraîne l'agent avec Q-learning.
        
        Args:
            n_episodes: Nombre d'épisodes d'entraînement
            max_steps: Nombre maximal de pas par épisode
            verbose: Afficher les progrès
            save_frequency: Fréquence de sauvegarde
            checkpoint_dir: Répertoire pour checkpoints
            track_q_history: Si True, sauvegarde Q à chaque épisode (lourd en mémoire)
            
        Returns:
            Historique des métriques
        """
        self.rewards_history = []
        self.convergence_history = []
        self.epsilon_history = []
        self.td_errors = []
        
        if track_q_history:
            self.q_history = []
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            episode_td_errors = []
            
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Enregistrer l'erreur TD avant la mise à jour
                old_q = self.Q[state, action]
                
                self.update(state, action, reward, next_state, done)
                
                # Calculer l'erreur TD
                new_q = self.Q[state, action]
                episode_td_errors.append(abs(new_q - old_q))
                
                total_reward += reward
                state = next_state
                steps += 1
                self.total_steps += 1
                
                if done:
                    break
            
            self.total_episodes += 1
            
            # Décroissance de epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Enregistrement des métriques
            self.rewards_history.append(total_reward)
            self.convergence_history.append(np.max(self.Q))
            self.epsilon_history.append(self.epsilon)
            self.episode_lengths.append(steps)
            self.success_history.append(1 if terminated else 0)
            
            if track_q_history:
                self.q_history.append(self.Q.copy())
            
            # Sauvegarde automatique
            if checkpoint_dir and (episode + 1) % save_frequency == 0:
                from pathlib import Path
                checkpoint_path = Path(checkpoint_dir) / f"{self.name}_ep{episode+1}.npz"
                self.save(str(checkpoint_path))
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])
                avg_q = np.mean(self.convergence_history[-100:])
                success_rate = np.mean(self.success_history[-100:]) * 100
                avg_td = np.mean(episode_td_errors) if episode_td_errors else 0
                
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Reward: {avg_reward:.2f}, "
                      f"Q max: {avg_q:.4f}, "
                      f"ε: {self.epsilon:.4f}, "
                      f"Succès: {success_rate:.1f}%, "
                      f"TD error: {avg_td:.4f}")
        
        return {
            'rewards': self.rewards_history,
            'convergence': self.convergence_history,
            'epsilon': self.epsilon_history,
            'lengths': self.episode_lengths,
            'success_rate': self.success_history,
            'td_errors': self.td_errors
        }
    
    def reset_training(self) -> None:
        """Réinitialise l'agent pour un nouvel entraînement."""
        super().reset_training()
        self.epsilon = self.epsilon_initial
        self.epsilon_history = []
        self.td_errors = []
        
        if self.use_double_q:
            self.Q2 = np.zeros((self.n_states, self.n_actions))
    
    def save(self, filepath: str, include_history: bool = True) -> None:
        """
        Sauvegarde l'agent Q-Learning.
        
        Args:
            filepath: Chemin du fichier
            include_history: Inclure l'historique
        """
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'policy': self.policy,
            'V': self.V,
            'Q': self.Q,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'epsilon_initial': self.epsilon_initial,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'use_double_q': self.use_double_q,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'agent_name': self.name
        }
        
        if self.use_double_q:
            save_dict['Q2'] = self.Q2
        
        if include_history:
            save_dict.update({
                'rewards_history': np.array(self.rewards_history),
                'convergence_history': np.array(self.convergence_history),
                'epsilon_history': np.array(self.epsilon_history),
                'episode_lengths': np.array(self.episode_lengths),
                'success_history': np.array(self.success_history)
            })
        
        np.savez_compressed(filepath, **save_dict)
        print(f"✓ Q-Learning agent sauvegardé: {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Charge l'agent Q-Learning.
        
        Args:
            filepath: Chemin du fichier
        """
        from pathlib import Path
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Fichier introuvable: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        
        self.policy = data['policy']
        self.V = data['V']
        self.Q = data['Q']
        self.gamma = float(data['gamma'])
        self.learning_rate = float(data['learning_rate'])
        self.epsilon = float(data['epsilon'])
        self.epsilon_initial = float(data['epsilon_initial'])
        self.epsilon_decay = float(data['epsilon_decay'])
        self.epsilon_min = float(data['epsilon_min'])
        self.use_double_q = bool(data['use_double_q'])
        
        if self.use_double_q and 'Q2' in data:
            self.Q2 = data['Q2']
        
        if 'total_steps' in data:
            self.total_steps = int(data['total_steps'])
        if 'total_episodes' in data:
            self.total_episodes = int(data['total_episodes'])
        
        # Charger l'historique
        if 'rewards_history' in data:
            self.rewards_history = list(data['rewards_history'])
        if 'convergence_history' in data:
            self.convergence_history = list(data['convergence_history'])
        if 'epsilon_history' in data:
            self.epsilon_history = list(data['epsilon_history'])
        if 'episode_lengths' in data:
            self.episode_lengths = list(data['episode_lengths'])
        if 'success_history' in data:
            self.success_history = list(data['success_history'])
        
        print(f"✓ Q-Learning agent chargé: {filepath}")