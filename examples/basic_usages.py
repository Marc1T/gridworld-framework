"""
Exemples d'utilisation basique du framework GridWorld RL.

Ce module montre comment utiliser les différents composants du framework
pour créer des environnements, entraîner des agents et visualiser les résultats.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from gridworld_framework.core.gridworld_env import GridWorldEnv
from gridworld_framework.agents.random_agent import RandomAgent
from gridworld_framework.agents.policy_iteration import PolicyIterationAgent
from gridworld_framework.agents.value_iteration import ValueIterationAgent
from gridworld_framework.agents.monte_carlo import MonteCarloAgent
from gridworld_framework.agents.q_learning import QLearningAgent
from gridworld_framework.utils.visualization import (
    plot_learning_curve, 
    plot_value_function,
    plot_q_function,
    plot_policy,
    visualize_gridworld
)
from gridworld_framework.utils.metrics import compare_agents


def demo_random_agent():
    """
    Démonstration de l'agent aléatoire comme baseline.
    
    Cet exemple montre comment:
    1. Créer un environnement GridWorld
    2. Initialiser un agent aléatoire
    3. Exécuter une marche aléatoire
    4. Visualiser les résultats
    """
    print("=== DÉMO AGENT ALÉATOIRE ===")
    
    # 1. Créer l'environnement
    env = GridWorldEnv(
        grid_shape=(4, 4),
        initial_state=0,
        goal_state=15,
        obstacles=[5, 7, 11],
        render_mode='human'
    )
    
    # 2. Créer l'agent aléatoire
    agent = RandomAgent(env)
    
    # 3. Exécuter une marche aléatoire
    print("Exécution d'une marche aléatoire...")
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(20):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        print(f"Step {step}: État {state} -> Action {action} -> État {next_state}, Reward: {reward}")
        env.render()
        
        total_reward += reward
        state = next_state
        steps += 1
        
        if terminated or truncated:
            break
    
    print(f"Episode terminé après {steps} steps, reward total: {total_reward}")
    
    # 4. Évaluation
    results = agent.evaluate(n_episodes=10)
    print(f"Performance moyenne sur 10 épisodes: {results}")
    
    env.close()
    return env, agent


def demo_policy_iteration():
    """
    Démonstration de Policy Iteration.
    
    Cet exemple montre comment:
    1. Entraîner un agent avec Policy Iteration
    2. Visualiser la fonction de valeur et la politique
    3. Évaluer la performance
    """
    print("\n=== DÉMO POLICY ITERATION ===")
    
    # 1. Créer l'environnement
    env = GridWorldEnv(
        grid_shape=(4, 4),
        initial_state=0,
        goal_state=15,
        obstacles=[5, 7, 11]
    )
    
    # 2. Créer et entraîner l'agent
    agent = PolicyIterationAgent(env, gamma=0.99, theta=1e-6)
    history = agent.train(verbose=True)
    
    # 3. Visualiser les résultats
    fig1 = plot_learning_curve(history, "Policy Iteration - Courbe d'apprentissage")
    
    fig2 = plot_value_function(
        agent.get_value_function(), 
        env.grid_shape,
        "Policy Iteration - Fonction de valeur V(s)"
    )
    
    fig3 = plot_policy(
        agent.get_policy(),
        env.grid_shape,
        title="Policy Iteration - Politique optimale"
    )
    
    # 4. Évaluation
    results = agent.evaluate(n_episodes=100)
    print(f"Performance de Policy Iteration: {results}")
    
    # 5. Visualisation complète
    fig4 = visualize_gridworld(env, agent)
    
    plt.show()
    
    return env, agent, history


def demo_value_iteration():
    """
    Démonstration de Value Iteration.
    
    Cet exemple montre comment:
    1. Entraîner un agent avec Value Iteration  
    2. Comparer avec Policy Iteration
    3. Visualiser la convergence
    """
    print("\n=== DÉMO VALUE ITERATION ===")
    
    # 1. Créer l'environnement
    env = GridWorldEnv(
        grid_shape=(4, 4),
        initial_state=0,
        goal_state=15,
        obstacles=[5, 7, 11]
    )
    
    # 2. Créer et entraîner l'agent
    agent = ValueIterationAgent(env, gamma=0.99, theta=1e-6)
    history = agent.train(verbose=True)
    
    # 3. Visualiser les résultats
    fig1 = plot_learning_curve(history, "Value Iteration - Courbe d'apprentissage")
    
    fig2 = plot_value_function(
        agent.get_value_function(),
        env.grid_shape,
        "Value Iteration - Fonction de valeur V(s)"
    )
    
    # 4. Évaluation
    results = agent.evaluate(n_episodes=100)
    print(f"Performance de Value Iteration: {results}")
    
    plt.show()
    
    return env, agent, history


def demo_monte_carlo():
    """
    Démonstration de Monte Carlo.
    
    Cet exemple montre comment:
    1. Entraîner un agent avec Monte Carlo
    2. Observer l'exploration vs exploitation
    3. Visualiser l'apprentissage en ligne
    """
    print("\n=== DÉMO MONTE CARLO ===")
    
    # 1. Créer l'environnement
    env = GridWorldEnv(
        grid_shape=(4, 4),
        initial_state=0,
        goal_state=15,
        obstacles=[5, 7, 11]
    )
    
    # 2. Créer et entraîner l'agent
    agent = MonteCarloAgent(
        env, 
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        first_visit=True
    )
    
    history = agent.train(n_episodes=1000, verbose=True)
    
    # 3. Visualiser les résultats
    fig1 = plot_learning_curve(history, "Monte Carlo - Courbe d'apprentissage")
    
    fig2 = plot_value_function(
        agent.get_value_function(),
        env.grid_shape, 
        "Monte Carlo - Fonction de valeur V(s)"
    )
    
    # 4. Évaluation
    results = agent.evaluate(n_episodes=100)
    print(f"Performance de Monte Carlo: {results}")
    
    plt.show()
    
    return env, agent, history


def demo_q_learning():
    """
    Démonstration de Q-learning.
    
    Cet exemple montre comment:
    1. Entraîner un agent avec Q-learning
    2. Observer la décroissance d'epsilon
    3. Visualiser la fonction Q
    """
    print("\n=== DÉMO Q-LEARNING ===")
    
    # 1. Créer l'environnement
    env = GridWorldEnv(
        grid_shape=(4, 4),
        initial_state=0,
        goal_state=15, 
        obstacles=[5, 7, 11]
    )
    
    # 2. Créer et entraîner l'agent
    agent = QLearningAgent(
        env,
        gamma=0.99,
        learning_rate=0.1,
        epsilon=1.0,
        epsilon_decay=0.998,
        epsilon_min=0.01
    )
    
    history = agent.train(n_episodes=1000, verbose=True)
    
    # 3. Visualiser les résultats
    fig1 = plot_learning_curve(history, "Q-learning - Courbe d'apprentissage")
    
    fig2 = plot_q_function(
        agent.get_q_function(),
        env.grid_shape,
        title="Q-learning - Fonction Q(s,a)"
    )
    
    # 4. Évaluation
    results = agent.evaluate(n_episodes=100)
    print(f"Performance de Q-learning: {results}")
    
    plt.show()
    
    return env, agent, history


def compare_all_agents():
    """
    Compare tous les algorithmes sur le même environnement.
    
    Cet exemple montre comment:
    1. Comparer les performances de différents algorithmes
    2. Générer un rapport comparatif
    3. Visualiser les différences entre les politiques
    """
    print("\n=== COMPARAISON DE TOUS LES AGENTS ===")
    
    # 1. Créer l'environnement (le même pour tous)
    env = GridWorldEnv(
        grid_shape=(4, 4),
        initial_state=0,
        goal_state=15,
        obstacles=[5, 7, 11]
    )
    
    # 2. Créer tous les agents
    agents = {
        'Random': RandomAgent(env),
        'PolicyIteration': PolicyIterationAgent(env, gamma=0.99),
        'ValueIteration': ValueIterationAgent(env, gamma=0.99),
        'MonteCarlo': MonteCarloAgent(env, gamma=0.99, epsilon=0.1),
        'QLearning': QLearningAgent(env, gamma=0.99, learning_rate=0.1, epsilon=0.1)
    }
    
    # 3. Entraîner les agents (sauf Random)
    print("Entraînement des agents...")
    
    # Policy Iteration
    print("Policy Iteration...")
    agents['PolicyIteration'].train(verbose=False)
    
    # Value Iteration  
    print("Value Iteration...")
    agents['ValueIteration'].train(verbose=False)
    
    # Monte Carlo
    print("Monte Carlo...")
    agents['MonteCarlo'].train(n_episodes=500, verbose=False)
    
    # Q-learning
    print("Q-learning...")
    agents['QLearning'].train(n_episodes=500, verbose=False)
    
    # 4. Comparer les performances
    print("\nComparaison des performances...")
    results = compare_agents(agents, env, n_episodes=100)
    
    # 5. Afficher les résultats
    print("\n=== RÉSULTATS DE LA COMPARAISON ===")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # 6. Visualiser les politiques
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, agent) in enumerate(agents.items()):
        if idx >= len(axes):
            break
            
        policy = agent.get_policy().reshape(env.grid_shape)
        action_names = ['↑', '→', '↓', '←']
        policy_text = np.empty(env.grid_shape, dtype=object)
        
        for i in range(env.grid_shape[0]):
            for j in range(env.grid_shape[1]):
                policy_text[i, j] = action_names[policy[i, j]]
        
        axes[idx].imshow(policy, cmap='Set3')
        for i in range(env.grid_shape[0]):
            for j in range(env.grid_shape[1]):
                axes[idx].text(j, i, policy_text[i, j], 
                             ha="center", va="center", color="black",
                             fontsize=12, fontweight='bold')
        
        axes[idx].set_title(f'{name}\nReward: {results[name]["mean_reward"]:.2f}')
        axes[idx].set_xlabel('Colonne')
        axes[idx].set_ylabel('Ligne')
    
    # Cacher le dernier axe si nécessaire
    if len(agents) < len(axes):
        for idx in range(len(agents), len(axes)):
            axes[idx].set_visible(False)
    
    plt.suptitle("Comparaison des politiques optimales")
    plt.tight_layout()
    plt.show()
    
    return env, agents, results


if __name__ == "__main__":
    # Exécuter toutes les démos
    print("DÉMARRAGE DES DÉMONSTRATIONS GRIDWORLD RL")
    print("=" * 50)
    
    # Démo agent aléatoire
    env_random, agent_random = demo_random_agent()
    
    # Démo algorithmes planifiés
    env_pi, agent_pi, history_pi = demo_policy_iteration()
    env_vi, agent_vi, history_vi = demo_value_iteration()
    
    # Démo algorithmes d'apprentissage
    env_mc, agent_mc, history_mc = demo_monte_carlo()
    env_ql, agent_ql, history_ql = demo_q_learning()
    
    # Comparaison finale
    env_comp, agents_comp, results_comp = compare_all_agents()
    
    print("\n=== TOUTES LES DÉMONSTRATIONS TERMINÉES ===")