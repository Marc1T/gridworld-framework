"""
Test Q-learning avec différentes tailles de grille.
Analyse l'impact de la taille du GridWorld sur l'apprentissage.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from gridworld_framework.core.gridworld_env import GridWorldEnv # type: ignore
from gridworld_framework.agents.q_learning import QLearningAgent # type: ignore
from gridworld_framework.utils.visualization import plot_learning_curve, plot_policy # type: ignore


def create_grid_config(size):
    """
    Crée une configuration de grille pour une taille donnée.
    
    Args:
        size: Taille de la grille (size x size)
        
    Returns:
        Dict avec la configuration
    """
    n_states = size * size
    
    # Goal en bas à droite
    goal_state = n_states - 1
    
    # Obstacles : environ 15% des cases
    n_obstacles = max(1, int(n_states * 0.15))
    
    # Éviter l'état initial et le goal pour les obstacles
    possible_obstacles = [s for s in range(n_states) if s != 0 and s != goal_state]
    obstacles = np.random.choice(possible_obstacles, n_obstacles, replace=False)
    
    return {
        'grid_shape': (size, size),
        'initial_state': 0,
        'goal_state': goal_state,
        'obstacles': obstacles.tolist(),
        'goal_fixed': True
    }


def test_qlearning_grid_sizes(grid_sizes=[4, 6, 8, 10], n_episodes=2000):
    """
    Test Q-learning sur différentes tailles de grille.
    
    Args:
        grid_sizes: Liste des tailles de grille à tester
        n_episodes: Nombre d'épisodes d'entraînement
        
    Returns:
        Dict avec résultats pour chaque taille
    """
    results = {}
    
    for size in grid_sizes:
        print(f"\n{'='*50}")
        print(f"TEST AVEC GRILLE {size}x{size}")
        print(f"{'='*50}")
        
        # 1. Créer la configuration
        config = create_grid_config(size)
        print(f"Configuration: Goal={config['goal_state']}, Obstacles={config['obstacles']}")
        
        # 2. Créer l'environnement
        env = GridWorldEnv(**config, render_mode=None)
        
        # 3. Créer et entraîner l'agent Q-learning
        agent = QLearningAgent(
            env,
            gamma=0.99,
            learning_rate=0.1,
            epsilon=1.0,
            epsilon_decay=0.999,
            epsilon_min=0.01
        )
        
        print("Début de l'entraînement...")
        history = agent.train(n_episodes=n_episodes, verbose=True, max_steps=size*10)
        
        # 4. Évaluation finale
        eval_results = agent.evaluate(n_episodes=100)
        
        # 5. Sauvegarder les résultats
        results[size] = {
            'config': config,
            'agent': agent,
            'history': history,
            'eval_results': eval_results,
            'env': env
        }
        
        print(f"\nRésultats pour {size}x{size}:")
        print(f"  - Reward moyen: {eval_results['mean_reward']:.3f}")
        print(f"  - Taux de succès: {np.mean([r > 0 for r in agent.rewards_history[-100:]])*100:.1f}%")
        print(f"  - Q max final: {np.max(agent.Q):.3f}")
        print(f"  - Epsilon final: {agent.epsilon:.3f}")
    
    return results


def plot_grid_size_comparison(results):
    """
    Trace la comparaison des performances par taille de grille.
    
    Args:
        results: Résultats de test_qlearning_grid_sizes()
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # 1. Courbes d'apprentissage superposées
    ax = axes[0]
    for size, data in results.items():
        rewards = data['history']['rewards']
        # Moyenne mobile pour lisser
        window = max(1, len(rewards) // 50)
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), moving_avg, 
                   label=f'{size}x{size}', linewidth=2)
    
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Récompense (moyenne mobile)')
    ax.set_title('Courbes d\'apprentissage par taille de grille')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Performance finale
    ax = axes[1]
    sizes = list(results.keys())
    mean_rewards = [results[size]['eval_results']['mean_reward'] for size in sizes]
    success_rates = [np.mean([r > 0 for r in results[size]['agent'].rewards_history[-100:]]) * 100 
                    for size in sizes]
    
    x = np.arange(len(sizes))
    width = 0.35
    
    ax.bar(x - width/2, mean_rewards, width, label='Reward moyen', alpha=0.7)
    ax.bar(x + width/2, success_rates, width, label='Taux de succès (%)', alpha=0.7)
    
    ax.set_xlabel('Taille de grille')
    ax.set_ylabel('Performance')
    ax.set_title('Performance finale par taille de grille')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{size}x{size}' for size in sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Vitesse de convergence
    ax = axes[2]
    convergence_speeds = []
    
    for size, data in results.items():
        rewards = data['history']['rewards']
        # Définir la convergence comme atteignant 80% du reward max final
        target_reward = data['eval_results']['mean_reward'] * 0.8
        convergence_episode = len(rewards)
        
        for i, reward in enumerate(rewards):
            if reward >= target_reward and i > 100:  # Éviter les faux positifs précoces
                convergence_episode = i
                break
        
        convergence_speeds.append(convergence_episode)
    
    ax.bar([f'{size}x{size}' for size in sizes], convergence_speeds, alpha=0.7)
    ax.set_xlabel('Taille de grille')
    ax.set_ylabel('Épisodes jusqu\'à convergence')
    ax.set_title('Vitesse de convergence')
    ax.grid(True, alpha=0.3)
    
    # 4. Complexité (nombre d'états vs performance)
    ax = axes[3]
    n_states = [size * size for size in sizes]
    ax.scatter(n_states, mean_rewards, s=100, alpha=0.7)
    
    for i, size in enumerate(sizes):
        ax.annotate(f'{size}x{size}', (n_states[i], mean_rewards[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Nombre d\'états')
    ax.set_ylabel('Reward moyen final')
    ax.set_title('Complexité vs Performance')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def visualize_agent_evolution(results, grid_size=6, n_steps=20):
    """
    Visualise l'évolution d'un agent pendant l'apprentissage.
    
    Args:
        results: Résultats des tests
        grid_size: Taille de grille à visualiser
        n_steps: Nombre d'étapes à montrer
    """
    if grid_size not in results:
        print(f"Taille {grid_size} non trouvée dans les résultats")
        return
    
    data = results[grid_size]
    env = data['env']
    agent = data['agent']
    
    # Créer un environnement pour la visualisation
    viz_env = GridWorldEnv(**data['config'], render_mode='matplotlib')
    
    print(f"\nVisualisation de l'agent sur grille {grid_size}x{grid_size}")
    print("Début de la simulation...")
    
    # Réinitialiser et exécuter
    state, _ = viz_env.reset()
    total_reward = 0
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for step in range(n_steps):
        # Choisir l'action (exploitation seulement)
        action = agent.act(state, explore=False)
        next_state, reward, terminated, truncated, _ = viz_env.step(action)
        
        # Rendu
        viz_env.render()
        
        # Mettre à jour l'état
        total_reward += reward
        state = next_state
        
        print(f"Step {step}: État {state} → Action {action} → Reward: {reward}")
        
        if terminated:
            print(f"🎉 Goal atteint à l'étape {step}!")
            break
        if truncated:
            print(f"⏰ Timeout à l'étape {step}")
            break
    
    print(f"Reward total: {total_reward}")
    
    # Visualisations supplémentaires
    plot_policy(agent.get_policy(), env.grid_shape, 
                title=f"Politique optimale - Grille {grid_size}x{grid_size}")
    
    plot_learning_curve(data['history'], 
                       f"Courbe d'apprentissage - Grille {grid_size}x{grid_size}")
    
    plt.show()
    
    viz_env.close()


def run_complete_analysis():
    """
    Exécute l'analyse complète.
    """
    print("ANALYSE Q-LEARNING AVEC DIFFÉRENTES TAILLES DE GRILLE")
    print("=" * 60)
    
    # 1. Tests avec différentes tailles
    grid_sizes = [4, 6, 8, 10]
    results = test_qlearning_grid_sizes(grid_sizes, n_episodes=1500)
    
    # 2. Comparaison graphique
    print("\nGénération des graphiques de comparaison...")
    plot_grid_size_comparison(results)
    
    # 3. Visualisation détaillée pour une taille moyenne
    print("\nVisualisation détaillée...")
    visualize_agent_evolution(results, grid_size=6)
    
    # 4. Résumé final
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    
    for size in grid_sizes:
        data = results[size]
        eval_results = data['eval_results']
        success_rate = np.mean([r > 0 for r in data['agent'].rewards_history[-100:]]) * 100
        
        print(f"\nGrille {size}x{size}:")
        print(f"  • Reward moyen: {eval_results['mean_reward']:.3f}")
        print(f"  • Taux de succès: {success_rate:.1f}%")
        print(f"  • Longueur moyenne: {eval_results['mean_length']:.1f} steps")
        print(f"  • État final Q max: {np.max(data['agent'].Q):.3f}")
    
    return results

def run_analysis():
    """
    Analyse demandée.
    """
    print("ANALYSE Q-LEARNING AVEC DIFFÉRENTES TAILLES DE GRILLE")
    print("=" * 60)
    
    # 1. Tests avec différentes tailles
    grid_sizes = [4, 6, 8, 10]
    # results = test_qlearning_grid_sizes(grid_sizes, n_episodes=1500)
    
    # 2. Comparaison graphique
    # print("\nGénération des graphiques de comparaison...")
    # plot_grid_size_comparison(results)
    
    # 3. Visualisation détaillée pour une taille moyenne
    print("\nVisualisation détaillée...")
    results = test_qlearning_grid_sizes([10], n_episodes=1500)
    visualize_agent_evolution(results, grid_size=10, n_steps=50)
    

    return results

if __name__ == "__main__":
    # Exécuter l'analyse complète
    results = run_analysis()