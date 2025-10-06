"""
Fonctions de visualisation pour le framework GridWorld.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
# import seaborn as sns


def plot_learning_curve(history: Dict[str, List[float]], 
                       title: str = "Courbe d'apprentissage",
                       figsize: tuple = (12, 4)) -> plt.Figure: #type: ignore
    """
    Trace la courbe d'apprentissage avec les récompenses et la convergence.
    
    Args:
        history: Historique des métriques
        title: Titre du graphique
        figsize: Taille de la figure
        
    Returns:
        Figure matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Courbe des récompenses
    if 'rewards' in history:
        rewards = history['rewards']
        axes[0].plot(rewards, alpha=0.6, linewidth=1)
        
        # Moyenne mobile
        window = max(1, len(rewards) // 20)
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(rewards)), moving_avg, 
                       linewidth=2, label=f'Moyenne ({window} épisodes)')
        
        axes[0].set_xlabel('Épisode')
        axes[0].set_ylabel('Récompense')
        axes[0].set_title('Récompenses par épisode')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Courbe de convergence
    if 'convergence' in history:
        convergence = history['convergence']
        axes[1].plot(convergence, linewidth=2)
        axes[1].set_xlabel('Itération')
        axes[1].set_ylabel('Valeur max')
        axes[1].set_title('Convergence de la fonction de valeur')
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_value_function(V: np.ndarray, grid_shape: tuple,
                       title: str = "Fonction de valeur V(s)",
                       cmap: str = "viridis") -> plt.Figure: #type: ignore
    """
    Visualise la fonction de valeur V(s) sous forme de heatmap.
    
    Args:
        V: Fonction de valeur shape (n_states,)
        grid_shape: Dimensions de la grille (lignes, colonnes)
        title: Titre du graphique
        cmap: Colormap
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Reshape en grille 2D
    value_grid = V.reshape(grid_shape)
    
    # Heatmap
    im = ax.imshow(value_grid, cmap=cmap)
    
    # Annotations
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            text = ax.text(j, i, f'{value_grid[i, j]:.2f}',
                          ha="center", va="center", color="w",
                          fontweight='bold')
    
    ax.set_title(title)
    ax.set_xlabel('Colonne')
    ax.set_ylabel('Ligne')
    plt.colorbar(im, ax=ax)
    
    return fig


def plot_policy(policy: np.ndarray, grid_shape: tuple,
               action_names: List[str] = None, # type: ignore
               title: str = "Politique optimale") -> plt.Figure: #type: ignore
    """
    Visualise la politique optimale.
    
    Args:
        policy: Politique shape (n_states,) ou (n_states, n_actions)
        grid_shape: Dimensions de la grille
        action_names: Noms des actions
        title: Titre du graphique
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Si politique est une distribution, prendre l'action optimale
    if policy.ndim == 2:
        policy_actions = np.argmax(policy, axis=1)
    else:
        policy_actions = policy
    
    # Reshape en grille 2D
    policy_grid = policy_actions.reshape(grid_shape)
    
    # Noms d'actions par défaut
    if action_names is None:
        action_names = ['↑', '→', '↓', '←']  # Flèches
    
    # Créer une grille de texte
    policy_text = np.empty(grid_shape, dtype=object)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            policy_text[i, j] = action_names[policy_grid[i, j]]
    
    # Heatmap avec annotations
    im = ax.imshow(policy_grid, cmap='Set3')
    
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            ax.text(j, i, policy_text[i, j], 
                   ha="center", va="center", color="black",
                   fontsize=16, fontweight='bold')
    
    ax.set_title(title)
    ax.set_xlabel('Colonne')
    ax.set_ylabel('Ligne')
    
    # Créer une colorbar avec les noms d'actions
    cbar = plt.colorbar(im, ax=ax, ticks=range(len(action_names)))
    cbar.ax.set_yticklabels(action_names)
    
    return fig


def plot_q_function(Q: np.ndarray, grid_shape: tuple,
                   action_names: List[str] = None, # type: ignore
                   title: str = "Fonction Q(s,a)") -> plt.Figure: # type: ignore
    """
    Visualise la fonction Q sous forme de sous-graphiques par action.
    
    Args:
        Q: Fonction Q shape (n_states, n_actions)
        grid_shape: Dimensions de la grille
        action_names: Noms des actions
        title: Titre du graphique
        
    Returns:
        Figure matplotlib
    """
    n_actions = Q.shape[1]
    
    if action_names is None:
        action_names = [f'Action {i}' for i in range(n_actions)]
    
    fig, axes = plt.subplots(1, n_actions, figsize=(5*n_actions, 5))
    
    if n_actions == 1:
        axes = [axes]
    
    vmin = np.min(Q)
    vmax = np.max(Q)
    
    for a in range(n_actions):
        # Reshape pour cette action
        q_grid = Q[:, a].reshape(grid_shape)
        
        im = axes[a].imshow(q_grid, cmap='RdYlBu', vmin=vmin, vmax=vmax) # type: ignore
        
        # Annotations
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                axes[a].text(j, i, f'{q_grid[i, j]:.2f}',  # type: ignore
                           ha="center", va="center", color="black" if abs(q_grid[i, j]) < (vmax-vmin)/2 else "white")
        
        axes[a].set_title(f'{action_names[a]}') # type: ignore
        axes[a].set_xlabel('Colonne') # type: ignore
        axes[a].set_ylabel('Ligne') # type: ignore
        plt.colorbar(im, ax=axes[a]) # type: ignore
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def visualize_gridworld(env, agent=None, figsize: tuple = (10, 8)) -> plt.Figure:  # type: ignore
    """
    Visualisation complète du GridWorld avec l'agent.
    
    Args:
        env: Environnement GridWorld
        agent: Agent (optionnel)
        figsize: Taille de la figure
        
    Returns:
        Figure matplotlib
    """
    fig = plt.figure(figsize=figsize)
    
    if agent is None:
        # Juste l'environnement
        env.render()
        return fig
    
    # Configuration de la grille
    n_plots = 3 if hasattr(agent, 'Q') else 2
    gs = plt.GridSpec(2, n_plots) # type: ignore
    
    # 1. État courant
    ax1 = fig.add_subplot(gs[0, 0])
    current_grid = np.zeros(env.grid_shape)
    for state in range(env.n_states):
        row, col = env._state_to_coords(state)
        if state == env.current_state:
            current_grid[row, col] = 0.8  # Agent
        elif state == env.goal_state:
            current_grid[row, col] = 0.4  # Goal
        elif state in env.obstacles:
            current_grid[row, col] = 0.0  # Obstacle
        else:
            current_grid[row, col] = 1.0  # Vide
    
    ax1.imshow(current_grid, cmap='viridis')
    ax1.set_title('État courant')
    
    # 2. Fonction de valeur
    ax2 = fig.add_subplot(gs[0, 1])
    V = agent.get_value_function().reshape(env.grid_shape)
    im2 = ax2.imshow(V, cmap='hot')
    ax2.set_title('Fonction de valeur V(s)')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Politique
    ax3 = fig.add_subplot(gs[1, 0])
    policy = agent.get_policy().reshape(env.grid_shape)
    action_names = ['↑', '→', '↓', '←']
    policy_text = np.empty(env.grid_shape, dtype=object)
    for i in range(env.grid_shape[0]):
        for j in range(env.grid_shape[1]):
            policy_text[i, j] = action_names[policy[i, j]]
    
    im3 = ax3.imshow(policy, cmap='Set3')
    for i in range(env.grid_shape[0]):
        for j in range(env.grid_shape[1]):
            ax3.text(j, i, policy_text[i, j], 
                    ha="center", va="center", color="black",
                    fontsize=12, fontweight='bold')
    ax3.set_title('Politique optimale')
    
    if n_plots == 3:
        # 4. Fonction Q (max par état)
        ax4 = fig.add_subplot(gs[1, 1])
        Q_max = np.max(agent.get_q_function(), axis=1).reshape(env.grid_shape)
        im4 = ax4.imshow(Q_max, cmap='RdYlBu')
        ax4.set_title('Max Q(s,a) par état')
        plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    return fig