"""
Outils d'analyse de convergence et de sensibilité pour les agents RL.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from gridworld_framework.agents.base_agent import BaseAgent
from gridworld_framework.core.gridworld_env import GridWorldEnv


def analyze_convergence(agent: BaseAgent, 
                       metric: str = 'Q',
                       window_size: int = 10) -> Dict[str, Any]:
    """
    Analyse la convergence d'un agent.
    
    Args:
        agent: Agent entraîné
        metric: Métrique à analyser ('Q', 'V', 'rewards')
        window_size: Taille de la fenêtre pour détecter la convergence
        
    Returns:
        Dictionnaire avec les métriques de convergence
    """
    if metric == 'Q' and agent.convergence_history:
        values = agent.convergence_history
    elif metric == 'V' and agent.v_history:
        values = [np.max(v) for v in agent.v_history]
    elif metric == 'rewards' and agent.rewards_history:
        values = agent.rewards_history
    else:
        return {'error': 'Pas de données disponibles'}
    
    if len(values) < window_size:
        return {'error': 'Pas assez de données'}
    
    # Calculer les différences
    diffs = np.abs(np.diff(values))
    
    # Moyenne mobile des différences
    moving_avg = np.convolve(diffs, np.ones(window_size)/window_size, mode='valid')
    
    # Trouver le point de convergence (quand les changements deviennent < 1% du max)
    threshold = np.max(diffs) * 0.01
    convergence_points = np.where(moving_avg < threshold)[0]
    
    if len(convergence_points) > 0:
        convergence_episode = convergence_points[0] + window_size
        converged = True
    else:
        convergence_episode = len(values)
        converged = False
    
    # Vitesse de convergence (pente moyenne avant convergence)
    if convergence_episode > 1:
        convergence_speed = (values[convergence_episode-1] - values[0]) / convergence_episode
    else:
        convergence_speed = 0
    
    # Stabilité après convergence
    if converged and convergence_episode < len(values) - window_size:
        post_convergence = values[convergence_episode:]
        stability = np.std(post_convergence) / (np.mean(np.abs(post_convergence)) + 1e-8)
    else:
        stability = np.inf
    
    return {
        'converged': converged,
        'convergence_episode': convergence_episode,
        'convergence_speed': convergence_speed,
        'stability': stability,
        'final_value': values[-1],
        'max_value': np.max(values),
        'total_episodes': len(values),
        'convergence_ratio': convergence_episode / len(values) if len(values) > 0 else 1.0
    }


def test_gridworld_scalability(agent_class, 
                               grid_sizes: List[Tuple[int, int]] = None,
                               n_episodes: int = 1000,
                               **agent_kwargs) -> Dict[str, List]:
    """
    Test la scalabilité d'un algorithme sur différentes tailles de grille.
    
    Args:
        agent_class: Classe de l'agent à tester
        grid_sizes: Liste des tailles de grille à tester
        n_episodes: Nombre d'épisodes par test
        **agent_kwargs: Arguments pour l'agent
        
    Returns:
        Résultats des tests de scalabilité
    """
    if grid_sizes is None:
        grid_sizes = [(4, 4), (5, 5), (6, 6), (8, 8), (10, 10)]
    
    results = {
        'grid_sizes': [],
        'n_states': [],
        'convergence_episodes': [],
        'final_rewards': [],
        'success_rates': [],
        'training_times': [],
        'convergence_speeds': []
    }
    
    for grid_size in grid_sizes:
        print(f"\n🔬 Test sur grille {grid_size[0]}x{grid_size[1]}...")
        
        n_states = grid_size[0] * grid_size[1]
        goal_state = n_states - 1
        
        # Créer l'environnement
        env = GridWorldEnv(
            grid_shape=grid_size,
            initial_state=0,
            goal_state=goal_state
        )
        
        # Créer et entraîner l'agent
        agent = agent_class(env, **agent_kwargs)
        
        import time
        start_time = time.time()
        history = agent.train(n_episodes=n_episodes, verbose=False)
        training_time = time.time() - start_time
        
        # Analyser la convergence
        conv_analysis = analyze_convergence(agent, metric='Q')
        
        # Évaluer l'agent
        eval_results = agent.evaluate(n_episodes=100)
        
        # Stocker les résultats
        results['grid_sizes'].append(f"{grid_size[0]}x{grid_size[1]}")
        results['n_states'].append(n_states)
        results['convergence_episodes'].append(conv_analysis.get('convergence_episode', n_episodes))
        results['final_rewards'].append(eval_results['mean_reward'])
        results['success_rates'].append(eval_results['success_rate'])
        results['training_times'].append(training_time)
        results['convergence_speeds'].append(conv_analysis.get('convergence_speed', 0))
        
        print(f"  ✓ États: {n_states}, Convergence: {conv_analysis.get('convergence_episode', 'N/A')}, "
              f"Succès: {eval_results['success_rate']:.1f}%, Temps: {training_time:.2f}s")
    
    return results


def test_hyperparameter_sensitivity(agent_class,
                                   env,
                                   param_name: str,
                                   param_values: List[float],
                                   n_episodes: int = 1000,
                                   n_trials: int = 5,
                                   **fixed_kwargs) -> Dict[str, List]:
    """
    Test la sensibilité d'un agent à un hyperparamètre.
    
    Args:
        agent_class: Classe de l'agent
        env: Environnement
        param_name: Nom de l'hyperparamètre à tester
        param_values: Valeurs à tester
        n_episodes: Nombre d'épisodes par test
        n_trials: Nombre d'essais par valeur (pour moyenne)
        **fixed_kwargs: Autres paramètres fixes
        
    Returns:
        Résultats des tests de sensibilité
    """
    results = {
        'param_values': param_values,
        'mean_rewards': [],
        'std_rewards': [],
        'mean_success_rates': [],
        'std_success_rates': [],
        'mean_convergence': [],
        'std_convergence': []
    }
    
    for param_value in param_values:
        print(f"\n🧪 Test {param_name} = {param_value}...")
        
        trial_rewards = []
        trial_success = []
        trial_convergence = []
        
        for trial in range(n_trials):
            # Créer l'agent avec le paramètre testé
            kwargs = fixed_kwargs.copy()
            kwargs[param_name] = param_value
            
            agent = agent_class(env, **kwargs)
            
            # Entraîner
            history = agent.train(n_episodes=n_episodes, verbose=False)
            
            # Évaluer
            eval_results = agent.evaluate(n_episodes=100)
            
            # Convergence
            conv_analysis = analyze_convergence(agent, metric='Q')
            
            trial_rewards.append(eval_results['mean_reward'])
            trial_success.append(eval_results['success_rate'])
            trial_convergence.append(conv_analysis.get('convergence_episode', n_episodes))
        
        # Moyennes et écarts-types
        results['mean_rewards'].append(np.mean(trial_rewards))
        results['std_rewards'].append(np.std(trial_rewards))
        results['mean_success_rates'].append(np.mean(trial_success))
        results['std_success_rates'].append(np.std(trial_success))
        results['mean_convergence'].append(np.mean(trial_convergence))
        results['std_convergence'].append(np.std(trial_convergence))
        
        print(f"  ✓ Reward: {np.mean(trial_rewards):.2f} ± {np.std(trial_rewards):.2f}, "
              f"Succès: {np.mean(trial_success):.1f}% ± {np.std(trial_success):.1f}%")
    
    return results


def plot_convergence_comparison(agents_results: Dict[str, Dict],
                               figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Compare la convergence de plusieurs agents.
    
    Args:
        agents_results: Dict {nom_agent: historique}
        figsize: Taille de la figure
        
    Returns:
        Figure matplotlib
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Courbes de récompenses
    for name, history in agents_results.items():
        if 'rewards' in history:
            rewards = history['rewards']
            # Moyenne mobile
            window = max(1, len(rewards) // 20)
            if len(rewards) >= window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                axes[0].plot(range(window-1, len(rewards)), moving_avg, 
                           label=name, linewidth=2)
    
    axes[0].set_xlabel('Épisode')
    axes[0].set_ylabel('Récompense moyenne')
    axes[0].set_title('Convergence des récompenses')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Convergence de Q
    for name, history in agents_results.items():
        if 'convergence' in history:
            axes[1].plot(history['convergence'], label=name, linewidth=2)
    
    axes[1].set_xlabel('Épisode')
    axes[1].set_ylabel('Q max')
    axes[1].set_title('Convergence de Q(s,a)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Taux de succès cumulé
    for name, history in agents_results.items():
        if 'success_rate' in history:
            success = history['success_rate']
            # Taux de succès cumulé
            cumulative_success = np.cumsum(success) / np.arange(1, len(success) + 1) * 100
            axes[2].plot(cumulative_success, label=name, linewidth=2)
    
    axes[2].set_xlabel('Épisode')
    axes[2].set_ylabel('Taux de succès (%)')
    axes[2].set_title('Taux de succès cumulé')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 105])
    
    plt.tight_layout()
    return fig


def plot_scalability_results(results: Dict[str, List],
                            figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Visualise les résultats de scalabilité.
    
    Args:
        results: Résultats de test_gridworld_scalability
        figsize: Taille de la figure
        
    Returns:
        Figure matplotlib
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    n_states = results['n_states']
    
    # 1. Épisodes de convergence vs taille
    axes[0].plot(n_states, results['convergence_episodes'], 
                marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Nombre d\'états')
    axes[0].set_ylabel('Épisodes de convergence')
    axes[0].set_title('Scalabilité de la convergence')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Taux de succès vs taille
    axes[1].plot(n_states, results['success_rates'], 
                marker='s', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Nombre d\'états')
    axes[1].set_ylabel('Taux de succès (%)')
    axes[1].set_title('Performance vs taille de grille')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])
    
    # 3. Temps d'entraînement vs taille
    axes[2].plot(n_states, results['training_times'], 
                marker='^', linewidth=2, markersize=8, color='red')
    axes[2].set_xlabel('Nombre d\'états')
    axes[2].set_ylabel('Temps (secondes)')
    axes[2].set_title('Temps d\'entraînement')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_sensitivity_analysis(results: Dict[str, List],
                             param_name: str,
                             figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Visualise l'analyse de sensibilité.
    
    Args:
        results: Résultats de test_hyperparameter_sensitivity
        param_name: Nom du paramètre testé
        figsize: Taille de la figure
        
    Returns:
        Figure matplotlib
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    param_values = results['param_values']
    
    # 1. Récompenses
    axes[0].errorbar(param_values, results['mean_rewards'], 
                    yerr=results['std_rewards'],
                    marker='o', linewidth=2, markersize=8, capsize=5)
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel('Récompense moyenne')
    axes[0].set_title(f'Sensibilité: Récompenses vs {param_name}')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Taux de succès
    axes[1].errorbar(param_values, results['mean_success_rates'], 
                    yerr=results['std_success_rates'],
                    marker='s', linewidth=2, markersize=8, capsize=5, color='green')
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel('Taux de succès (%)')
    axes[1].set_title(f'Sensibilité: Succès vs {param_name}')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Convergence
    axes[2].errorbar(param_values, results['mean_convergence'], 
                    yerr=results['std_convergence'],
                    marker='^', linewidth=2, markersize=8, capsize=5, color='purple')
    axes[2].set_xlabel(param_name)
    axes[2].set_ylabel('Épisodes de convergence')
    axes[2].set_title(f'Sensibilité: Convergence vs {param_name}')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_convergence_report(agent: BaseAgent,
                               save_path: Optional[str] = None) -> str:
    """
    Génère un rapport détaillé de convergence.
    
    Args:
        agent: Agent entraîné
        save_path: Chemin pour sauvegarder le rapport (optionnel)
        
    Returns:
        Rapport sous forme de string
    """
    # Analyse de convergence
    conv_q = analyze_convergence(agent, metric='Q')
    conv_rewards = analyze_convergence(agent, metric='rewards')
    
    # Générer le rapport
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║        RAPPORT D'ANALYSE DE CONVERGENCE                      ║
╚══════════════════════════════════════════════════════════════╝

Agent: {agent.name}
Total d'épisodes: {agent.total_episodes}
Total de steps: {agent.total_steps}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERGENCE DE Q(s,a)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Convergé: {'✓ OUI' if conv_q.get('converged') else '✗ NON'}
Épisode de convergence: {conv_q.get('convergence_episode', 'N/A')}
Ratio de convergence: {conv_q.get('convergence_ratio', 0)*100:.1f}%
Vitesse de convergence: {conv_q.get('convergence_speed', 0):.6f}
Stabilité post-convergence: {conv_q.get('stability', np.inf):.6f}
Valeur finale de Q max: {conv_q.get('final_value', 0):.4f}
Valeur maximale de Q: {conv_q.get('max_value', 0):.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERGENCE DES RÉCOMPENSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Convergé: {'✓ OUI' if conv_rewards.get('converged') else '✗ NON'}
Épisode de convergence: {conv_rewards.get('convergence_episode', 'N/A')}
Ratio de convergence: {conv_rewards.get('convergence_ratio', 0)*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERFORMANCE GLOBALE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    if agent.rewards_history:
        last_100 = agent.rewards_history[-100:]
        report += f"Récompense moyenne (100 derniers): {np.mean(last_100):.2f} ± {np.std(last_100):.2f}\n"
    
    if agent.success_history:
        last_100_success = agent.success_history[-100:]
        report += f"Taux de succès (100 derniers): {np.mean(last_100_success)*100:.1f}%\n"
    
    if agent.episode_lengths:
        last_100_lengths = agent.episode_lengths[-100:]
        report += f"Longueur moyenne d'épisode: {np.mean(last_100_lengths):.1f} ± {np.std(last_100_lengths):.1f}\n"
    
    report += "\n╚══════════════════════════════════════════════════════════════╝\n"
    
    # Sauvegarder si demandé
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Rapport sauvegardé: {save_path}")
    
    return report