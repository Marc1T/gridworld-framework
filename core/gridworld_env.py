"""
Environnement GridWorld compatible avec l'API Gymnasium.

Cet environnement implémente un monde de grille 2D avec des états, actions, récompenses
et transitions configurables.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, List
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """
    Environnement GridWorld 2D.
    
    Attributes:
        grid_shape (Tuple[int, int]): Dimensions de la grille (lignes, colonnes)
        initial_state (int): État initial
        goal_state (int): État objectif
        goal_fixed (bool): Si True, l'objectif est fixe
        obstacles (List[int]): Liste des états obstacles
        action_space (spaces.Discrete): Espace d'actions
        observation_space (spaces.Discrete): Espace d'observations
        metadata (Dict): Métadonnées de l'environnement
    """
    
    metadata = {'render_modes': ['human', 'matplotlib', 'rgb_array'], 'render_fps': 4}
    
    # Directions: 0=Haut, 1=Droite, 2=Bas, 3=Gauche
    ACTION_NAMES = ['Haut', 'Droite', 'Bas', 'Gauche']
    ACTION_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    def __init__(self, 
                 grid_shape: Tuple[int, int] = (4, 4),
                 initial_state: int = 0, 
                 goal_state: int = 15,
                 goal_fixed: bool = True, 
                 obstacles: Optional[List[int]] = None,
                 stochastic: bool = False,
                 noise: float = 0.0,
                 step_penalty: float = -0.01,
                 goal_reward: float = 1.0,
                 obstacle_penalty: float = -1.0,
                 render_mode: Optional[str] = None):
        """
        Initialise l'environnement GridWorld.
        
        Args:
            grid_shape: Dimensions de la grille (lignes, colonnes)
            initial_state: État initial (index linéaire)
            goal_state: État objectif (index linéaire)
            goal_fixed: Si True, l'objectif reste fixe
            obstacles: Liste des états obstacles
            stochastic: Si True, transitions stochastiques
            noise: Probabilité de prendre une action aléatoire (si stochastic=True)
            step_penalty: Pénalité par pas de temps
            goal_reward: Récompense pour atteindre le but
            obstacle_penalty: Pénalité pour collision avec obstacle
            render_mode: Mode de rendu ('human', 'matplotlib', 'rgb_array', None)
        """
        super().__init__()
        
        # Validation des paramètres
        if grid_shape[0] <= 0 or grid_shape[1] <= 0:
            raise ValueError("grid_shape doit contenir des valeurs positives")
        
        self.grid_shape = grid_shape
        self.n_states = grid_shape[0] * grid_shape[1]
        
        if not (0 <= initial_state < self.n_states):
            raise ValueError(f"initial_state doit être entre 0 et {self.n_states-1}")
        if not (0 <= goal_state < self.n_states):
            raise ValueError(f"goal_state doit être entre 0 et {self.n_states-1}")
        
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.goal_fixed = goal_fixed
        self.obstacles = obstacles if obstacles is not None else []
        self.stochastic = stochastic
        self.noise = noise if stochastic else 0.0
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.obstacle_penalty = obstacle_penalty
        self.render_mode = render_mode
        
        # Validation des obstacles
        for obs in self.obstacles:
            if not (0 <= obs < self.n_states):
                raise ValueError(f"Obstacle {obs} hors des limites")
            if obs == initial_state:
                raise ValueError("L'état initial ne peut pas être un obstacle")
            if obs == goal_state:
                raise ValueError("L'état objectif ne peut pas être un obstacle")
        
        # Espaces d'actions et d'observations
        self.action_space = spaces.Discrete(4)  # 4 directions
        self.observation_space = spaces.Discrete(self.n_states)
        
        # État courant
        self.current_state = initial_state
        self.step_count = 0
        self.max_steps = 1000
        
        # Historique pour analyses
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Initialisation des matrices de transition et récompense
        self._setup_mdp()
        
        # Pour le rendu
        self.fig = None
        self.ax = None
        
    def _setup_mdp(self):
        """Initialise les matrices de transition et récompense."""
        # Matrice de transition P[s, a, s']
        self.P = np.zeros((self.n_states, 4, self.n_states))
        # Matrice de récompense R[s, a]
        self.R = np.zeros((self.n_states, 4))
        
        for state in range(self.n_states):
            if state in self.obstacles:
                # États obstacles : rester sur place
                for action in range(4):
                    self.P[state, action, state] = 1.0
                    self.R[state, action] = 0.0
                continue
            
            if state == self.goal_state:
                # État terminal : rester sur place
                for action in range(4):
                    self.P[state, action, state] = 1.0
                    self.R[state, action] = 0.0
                continue
                
            for action in range(4):
                next_state = self._get_next_state(state, action)
                
                if self.stochastic and self.noise > 0:
                    # Transitions stochastiques
                    # Probabilité (1-noise) d'aller dans la direction voulue
                    self.P[state, action, next_state] = 1.0 - self.noise
                    
                    # Probabilité noise/3 d'aller dans les autres directions
                    for other_action in range(4):
                        if other_action != action:
                            other_next = self._get_next_state(state, other_action)
                            self.P[state, action, other_next] += self.noise / 3
                else:
                    # Transitions déterministes
                    self.P[state, action, next_state] = 1.0
                
                # Récompense (indépendante du stochastique)
                if next_state == self.goal_state:
                    self.R[state, action] = self.goal_reward
                elif next_state in self.obstacles or next_state == state:
                    # Collision avec obstacle ou mur
                    self.R[state, action] = self.obstacle_penalty
                else:
                    self.R[state, action] = self.step_penalty
    
    def _get_next_state(self, state: int, action: int) -> int:
        """
        Calcule le prochain état en fonction de l'action (déterministe).
        
        Args:
            state: État courant
            action: Action à prendre
            
        Returns:
            Prochain état
        """
        if state in self.obstacles or state == self.goal_state:
            return state  # États terminaux
            
        row, col = self._state_to_coords(state)
        delta_row, delta_col = self.ACTION_DELTAS[action]
        
        new_row = max(0, min(self.grid_shape[0] - 1, row + delta_row))
        new_col = max(0, min(self.grid_shape[1] - 1, col + delta_col))
        
        next_state = self._coords_to_state(new_row, new_col)
        
        # Si c'est un obstacle, rester sur place
        if next_state in self.obstacles:
            return state
            
        return next_state
    
    def _state_to_coords(self, state: int) -> Tuple[int, int]:
        """Convertit un état en coordonnées (ligne, colonne)."""
        return state // self.grid_shape[1], state % self.grid_shape[1]
    
    def _coords_to_state(self, row: int, col: int) -> int:
        """Convertit des coordonnées en état."""
        return row * self.grid_shape[1] + col
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[int, Dict]:
        """
        Réinitialise l'environnement.
        
        Args:
            seed: Graine aléatoire
            options: Options supplémentaires
            
        Returns:
            Tuple (observation, info)
        """
        super().reset(seed=seed)
        
        self.current_state = self.initial_state
        self.step_count = 0
        
        if not self.goal_fixed and self.n_states > 1:
            # Objectif aléatoire (éviter l'état initial et les obstacles)
            possible_goals = [s for s in range(self.n_states) 
                            if s != self.initial_state and s not in self.obstacles]
            if possible_goals:
                self.goal_state = self.np_random.choice(possible_goals)
                self._setup_mdp()  # Recalculer les récompenses
        
        info = {
            'goal_state': self.goal_state,
            'obstacles': self.obstacles,
            'grid_shape': self.grid_shape
        }
        return self.current_state, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """
        Exécute une étape de l'environnement.
        
        Args:
            action: Action à exécuter (0-3)
            
        Returns:
            Tuple (observation, reward, terminated, truncated, info)
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Action invalide: {action}")
        
        self.step_count += 1
        
        # État suivant selon la matrice de transition
        next_state_probs = self.P[self.current_state, action]
        next_state = self.np_random.choice(self.n_states, p=next_state_probs)
        
        reward = self.R[self.current_state, action]
        
        # Mettre à jour l'état
        prev_state = self.current_state
        self.current_state = next_state
        
        # Conditions de terminaison
        terminated = (self.current_state == self.goal_state)
        truncated = self.step_count >= self.max_steps
        
        info = {
            'goal_state': self.goal_state,
            'step_count': self.step_count,
            'prev_state': prev_state,
            'success': terminated
        }
        
        return self.current_state, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Affiche l'état courant de l'environnement.
        
        Returns:
            Image numpy si mode 'rgb_array', None sinon
        """
        if self.render_mode is None:
            return None
            
        grid = np.zeros(self.grid_shape, dtype=str)
        
        # Remplir la grille
        for state in range(self.n_states):
            row, col = self._state_to_coords(state)
            if state == self.current_state:
                grid[row, col] = 'A'  # Agent
            elif state == self.goal_state:
                grid[row, col] = 'G'  # Goal
            elif state in self.obstacles:
                grid[row, col] = 'X'  # Obstacle
            else:
                grid[row, col] = '.'  # Case vide
        
        if self.render_mode == 'human':
            print(f"Step: {self.step_count}")
            for row in grid:
                print(' '.join(row))
            print()
            
        elif self.render_mode == 'matplotlib':
            self._render_matplotlib(grid)
        
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array(grid)
            
        return None
    
    def _render_matplotlib(self, grid: np.ndarray):
        """Rendu avec Matplotlib."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()  # Mode interactif
            
        self.ax.clear()
        
        # Créer une grille colorée
        color_grid = np.zeros(self.grid_shape)
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                state = self._coords_to_state(i, j)
                if state == self.current_state:
                    color_grid[i, j] = 0.8  # Agent - bleu clair
                elif state == self.goal_state:
                    color_grid[i, j] = 0.4  # Goal - vert
                elif state in self.obstacles:
                    color_grid[i, j] = 0.0  # Obstacle - noir
                else:
                    color_grid[i, j] = 1.0  # Vide - blanc
        
        self.ax.imshow(color_grid, cmap='viridis', vmin=0, vmax=1)
        
        # Ajouter les annotations
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                text = grid[i, j]
                color = 'white' if text in ['X', 'A'] else 'black'
                self.ax.text(j, i, text, ha='center', va='center',
                           color=color, fontsize=20, fontweight='bold')
        
        self.ax.set_xticks(np.arange(-0.5, self.grid_shape[1], 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.grid_shape[0], 1), minor=True)
        self.ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        self.ax.set_title(f'GridWorld - Step: {self.step_count}')
        
        plt.tight_layout()
        plt.pause(0.1)
    
    def _render_rgb_array(self, grid: np.ndarray) -> np.ndarray:
        """Rendu en array RGB."""
        # Créer une image RGB
        cell_size = 50
        img = np.ones((self.grid_shape[0] * cell_size, 
                      self.grid_shape[1] * cell_size, 3), dtype=np.uint8) * 255
        
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                state = self._coords_to_state(i, j)
                y_start = i * cell_size
                x_start = j * cell_size
                
                if state == self.current_state:
                    img[y_start:y_start+cell_size, x_start:x_start+cell_size] = [0, 0, 255]  # Bleu
                elif state == self.goal_state:
                    img[y_start:y_start+cell_size, x_start:x_start+cell_size] = [0, 255, 0]  # Vert
                elif state in self.obstacles:
                    img[y_start:y_start+cell_size, x_start:x_start+cell_size] = [0, 0, 0]  # Noir
        
        return img
    
    def close(self):
        """Ferme l'environnement et libère les ressources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def get_transition_matrix(self) -> np.ndarray:
        """Retourne la matrice de transition P[s, a, s']."""
        return self.P.copy()
    
    def get_reward_matrix(self) -> np.ndarray:
        """Retourne la matrice de récompense R[s, a]."""
        return self.R.copy()
    
    def get_optimal_path(self, policy: np.ndarray, max_steps: int = 100) -> List[int]:
        """
        Simule un épisode avec une politique donnée.
        
        Args:
            policy: Politique (n_states,) ou (n_states, n_actions)
            max_steps: Nombre maximal de pas
            
        Returns:
            Liste des états visités
        """
        if policy.ndim == 2:
            policy = np.argmax(policy, axis=1)
        
        state = self.initial_state
        path = [state]
        
        for _ in range(max_steps):
            if state == self.goal_state or state in self.obstacles:
                break
            
            action = policy[state]
            state = self._get_next_state(state, action)
            path.append(state)
        
        return path