"""
Exemple complet d'utilisation du framework GridWorld.

Ce script montre comment:
1. Créer un environnement
2. Entraîner différents agents
3. Visualiser les résultats
4. Analyser la convergence
5. Tester la scalabilité
6. Faire des analyses de sensibilité
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Imports du framework
from gridworld_framework.core.gridworld_env import GridWorldEnv
from gridworld_framework.agents.q_learning import QLearningAgent
from gridworld_framework.agents.value_iteration import ValueIterationAgent
from gridworld_framework.agents.policy_iteration import PolicyIterationAgent
from gridworld_framework.agents.monte_carlo import MonteCarloAgent

from gridworld_framework.utils.visualization import (
    plot_learning_curve,
    plot_value_function,
    plot_policy,
    plot_q_function
)

from gridworld_framework.utils.convergence_analysis import (
    analyze_convergence,
    test_gridworld_scalability,
    test_hyperparameter_sensitivity,
    plot_convergence_comparison,
    plot_scalability_results,
    plot_sensitivity_analysis,
    generate_convergence_report
)


def example_1_basic_training():
    """Exemple 1: Entraînement basique d'un agent Q-Learning."""
    print("\n" + "="*60)
    print("EXEMPLE 1: Entraînement basique de Q-Learning")
    print("="*60)
    
    # Créer l'environnement
    env = GridWorldEnv(
        grid_shape=(5, 5),
        initial_state=0,
        goal_state=24,
        obstacles=[7, 12, 13]
    )
    
    # Créer l'agent
    agent = QLearningAgent(
        env,
        gamma=0.99,
        learning_rate=0.1,
        epsilon=1.0,
        epsilon_decay=0.995
    )
    
    # Entraîner
    print("\n🚀 Début de l'entraînement...")
    history = agent.train(n_episodes=1000, verbose=True)
    
    # Évaluer
    print("\n📊 Évaluation de l'agent...")
    eval_results = agent.evaluate(n_episodes=100)
    print(f"Récompense moyenne: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    # print(f"Taux de succès: {eval_results['success_rate']:.1f}%")
    
    # Visualiser
    print("\n📈 Création des visualisations...")
    
    # Courbes d'apprentissage
    fig1 = plot_learning_curve(history, title="Q-Learning - Courbe d'apprentissage")
    plt.savefig("results/plots/qlearning_learning_curve.png", dpi=150, bbox_inches='tight')
    
    # Fonction de valeur
    fig2 = plot_value_function(agent.V, env.grid_shape)
    plt.savefig("results/plots/qlearning_value_function.png", dpi=150, bbox_inches='tight')
    
    # Politique
    fig3 = plot_policy(agent.policy, env.grid_shape, action_names=['↑', '→', '↓', '←'])
    plt.savefig("results/plots/qlearning_policy.png", dpi=150, bbox_inches='tight')
    
    # Fonction Q
    fig4 = plot_q_function(agent.Q, env.grid_shape, action_names=['↑', '→', '↓', '←'])
    plt.savefig("results/plots/qlearning_q_function.png", dpi=150, bbox_inches='tight')
    
    # Sauvegarder l'agent
    agent.save("results/checkpoints/qlearning_final.npz")
    
    print("✓ Exemple 1 terminé!")
    plt.show()


def example_2_compare_algorithms():
    """Exemple 2: Comparer plusieurs algorithmes."""
    print("\n" + "="*60)
    print("EXEMPLE 2: Comparaison d'algorithmes")
    print("="*60)
    
    # Créer l'environnement
    env = GridWorldEnv(grid_shape=(4, 4), initial_state=0, goal_state=15)
    
    agents = {}
    histories = {}
    
    # 1. Value Iteration
    print("\n📚 Entraînement: Value Iteration")
    vi_agent = ValueIterationAgent(env, gamma=0.99)
    histories['Value Iteration'] = vi_agent.train(verbose=False)
    agents['Value Iteration'] = vi_agent
    
    # 2. Policy Iteration
    print("📚 Entraînement: Policy Iteration")
    pi_agent = PolicyIterationAgent(env, gamma=0.99)
    histories['Policy Iteration'] = pi_agent.train(verbose=False)
    agents['Policy Iteration'] = pi_agent
    
    # 3. Q-Learning
    print("📚 Entraînement: Q-Learning")
    q_agent = QLearningAgent(env, gamma=0.99, learning_rate=0.1)
    histories['Q-Learning'] = q_agent.train(n_episodes=500, verbose=False)
    agents['Q-Learning'] = q_agent
    
    # 4. Monte Carlo
    print("📚 Entraînement: Monte Carlo")
    mc_agent = MonteCarloAgent(env, gamma=0.99)
    histories['Monte Carlo'] = mc_agent.train(n_episodes=500, verbose=False)
    agents['Monte Carlo'] = mc_agent
    
    # Comparer
    print("\n📊 Évaluation des agents...")
    for name, agent in agents.items():
        results = agent.evaluate(n_episodes=100)
        # print(f"{name:20s} - Succès: {results['success_rate']:5.1f}%, "
        #       f"Reward: {results['mean_reward']:6.2f}")
    
    # Visualiser la comparaison
    fig = plot_convergence_comparison(histories, figsize=(18, 5))
    plt.savefig("results/plots/algorithms_comparison.png", dpi=150, bbox_inches='tight')
    
    print("✓ Exemple 2 terminé!")
    plt.show()


def example_3_scalability_test():
    """Exemple 3: Test de scalabilité."""
    print("\n" + "="*60)
    print("EXEMPLE 3: Test de scalabilité de Q-Learning")
    print("="*60)
    
    # Tester sur différentes tailles
    results = test_gridworld_scalability(
        agent_class=QLearningAgent,
        grid_sizes=[(4,4), (5,5), (6,6), (8,8)],
        n_episodes=500,
        gamma=0.99,
        learning_rate=0.1,
        epsilon_decay=0.995
    )
    
    # Visualiser
    fig = plot_scalability_results(results)
    plt.savefig("results/plots/scalability_analysis.png", dpi=150, bbox_inches='tight')
    
    print("\n📊 Résumé de la scalabilité:")
    for i, size in enumerate(results['grid_sizes']):
        print(f"Grille {size}: {results['n_states'][i]} états, "
              f"Convergence: {results['convergence_episodes'][i]} épisodes, "
              f"Succès: {results['success_rates'][i]:.1f}%")
    
    print("✓ Exemple 3 terminé!")
    plt.show()


def example_4_sensitivity_analysis():
    """Exemple 4: Analyse de sensibilité."""
    print("\n" + "="*60)
    print("EXEMPLE 4: Analyse de sensibilité du learning rate")
    print("="*60)
    
    # Créer l'environnement
    env = GridWorldEnv(grid_shape=(5, 5), initial_state=0, goal_state=24)
    
    # Tester différents learning rates
    learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    results = test_hyperparameter_sensitivity(
        agent_class=QLearningAgent,
        env=env,
        param_name='learning_rate',
        param_values=learning_rates,
        n_episodes=500,
        n_trials=3,  # 3 essais par valeur
        gamma=0.99,
        epsilon_decay=0.995
    )
    
    # Visualiser
    fig = plot_sensitivity_analysis(results, param_name='Learning Rate')
    plt.savefig("results/plots/sensitivity_learning_rate.png", dpi=150, bbox_inches='tight')
    
    print("\n📊 Meilleur learning rate:")
    best_idx = np.argmax(results['mean_success_rates'])
    print(f"α = {learning_rates[best_idx]} "
          f"(Succès: {results['mean_success_rates'][best_idx]:.1f}%)")
    
    print("✓ Exemple 4 terminé!")
    plt.show()


def example_5_detailed_convergence_analysis():
    """Exemple 5: Analyse détaillée de convergence."""
    print("\n" + "="*60)
    print("EXEMPLE 5: Analyse détaillée de convergence")
    print("="*60)
    
    # Créer et entraîner l'agent
    env = GridWorldEnv(grid_shape=(6, 6), initial_state=0, goal_state=35)
    agent = QLearningAgent(env, gamma=0.99, learning_rate=0.1)
    
    print("\n🚀 Entraînement...")
    agent.train(n_episodes=1000, verbose=False)
    
    # Analyser la convergence
    print("\n🔬 Analyse de convergence...")
    conv_analysis = analyze_convergence(agent, metric='Q')
    
    print(f"\nRésultats:")
    print(f"  Convergé: {'✓ OUI' if conv_analysis['converged'] else '✗ NON'}")
    print(f"  Épisode de convergence: {conv_analysis['convergence_episode']}")
    print(f"  Vitesse: {conv_analysis['convergence_speed']:.6f}")
    print(f"  Stabilité: {conv_analysis['stability']:.6f}")
    
    # Générer un rapport complet
    report = generate_convergence_report(agent, save_path="results/convergence_report.txt")
    print(report)
    
    print("✓ Exemple 5 terminé!")


def example_6_checkpoint_and_resume():
    """Exemple 6: Sauvegarde et reprise d'entraînement."""
    print("\n" + "="*60)
    print("EXEMPLE 6: Checkpoints et reprise")
    print("="*60)
    
    env = GridWorldEnv(grid_shape=(5, 5), initial_state=0, goal_state=24)
    
    # Entraîner partiellement
    print("\n🚀 Entraînement phase 1 (500 épisodes)...")
    agent = QLearningAgent(env, gamma=0.99, learning_rate=0.1)
    agent.train(n_episodes=500, verbose=False)
    
    # Sauvegarder
    agent.save("results/checkpoints/qlearning_checkpoint_500.npz")
    print(f"✓ Agent sauvegardé après {agent.total_episodes} épisodes")
    
    # Évaluation intermédiaire
    eval1 = agent.evaluate(n_episodes=100)
    print(f"Performance phase 1: Succès = {eval1['success_rate']:.1f}%")
    
    # Charger et continuer
    print("\n🔄 Reprise de l'entraînement...")
    agent2 = QLearningAgent(env, gamma=0.99, learning_rate=0.1)
    agent2.load("results/checkpoints/qlearning_checkpoint_500.npz")
    
    print(f"✓ Agent chargé: {agent2.total_episodes} épisodes déjà effectués")
    
    # Continuer l'entraînement
    print("\n🚀 Entraînement phase 2 (500 épisodes supplémentaires)...")
    agent2.train(n_episodes=500, verbose=False)
    
    # Évaluation finale
    eval2 = agent2.evaluate(n_episodes=100)
    print(f"Performance phase 2: Succès = {eval2['success_rate']:.1f}%")
    print(f"Amélioration: {eval2['success_rate'] - eval1['success_rate']:+.1f}%")
    
    print("✓ Exemple 6 terminé!")


def example_7_stochastic_environment():
    """Exemple 7: Environnement stochastique."""
    print("\n" + "="*60)
    print("EXEMPLE 7: Environnement stochastique")
    print("="*60)
    
    # Environnement déterministe
    env_det = GridWorldEnv(
        grid_shape=(5, 5),
        initial_state=0,
        goal_state=24,
        stochastic=False
    )
    
    # Environnement stochastique (10% de bruit)
    env_stoch = GridWorldEnv(
        grid_shape=(5, 5),
        initial_state=0,
        goal_state=24,
        stochastic=True,
        noise=0.1
    )
    
    results = {}
    
    # Entraîner sur déterministe
    print("\n🎯 Entraînement: Environnement déterministe")
    agent_det = QLearningAgent(env_det, gamma=0.99, learning_rate=0.1)
    history_det = agent_det.train(n_episodes=500, verbose=False)
    results['Déterministe'] = history_det
    eval_det = agent_det.evaluate(n_episodes=100)
    
    # Entraîner sur stochastique
    print("🎲 Entraînement: Environnement stochastique")
    agent_stoch = QLearningAgent(env_stoch, gamma=0.99, learning_rate=0.1)
    history_stoch = agent_stoch.train(n_episodes=500, verbose=False)
    results['Stochastique'] = history_stoch
    eval_stoch = agent_stoch.evaluate(n_episodes=100)
    
    # Comparer
    print("\n📊 Comparaison:")
    print(f"Déterministe  - Succès: {eval_det['success_rate']:.1f}%, "
          f"Reward: {eval_det['mean_reward']:.2f}")
    print(f"Stochastique  - Succès: {eval_stoch['success_rate']:.1f}%, "
          f"Reward: {eval_stoch['mean_reward']:.2f}")
    
    # Visualiser
    fig = plot_convergence_comparison(results)
    plt.savefig("results/plots/stochastic_comparison.png", dpi=150, bbox_inches='tight')
    
    print("✓ Exemple 7 terminé!")
    plt.show()


def run_all_examples():
    """Exécute tous les exemples."""
    # Créer les dossiers nécessaires
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("   FRAMEWORK GRIDWORLD - EXEMPLES COMPLETS")
    print("="*60)
    
    try:
        example_1_basic_training()
    except Exception as e:
        print(f"❌ Erreur dans exemple 1: {e}")
    
    try:
        example_2_compare_algorithms()
    except Exception as e:
        print(f"❌ Erreur dans exemple 2: {e}")
    
    try:
        example_3_scalability_test()
    except Exception as e:
        print(f"❌ Erreur dans exemple 3: {e}")
    
    try:
        example_4_sensitivity_analysis()
    except Exception as e:
        print(f"❌ Erreur dans exemple 4: {e}")
    
    try:
        example_5_detailed_convergence_analysis()
    except Exception as e:
        print(f"❌ Erreur dans exemple 5: {e}")
    
    try:
        example_6_checkpoint_and_resume()
    except Exception as e:
        print(f"❌ Erreur dans exemple 6: {e}")
    
    try:
        example_7_stochastic_environment()
    except Exception as e:
        print(f"❌ Erreur dans exemple 7: {e}")
    
    print("\n" + "="*60)
    print("   TOUS LES EXEMPLES TERMINÉS!")
    print("="*60)
    print("\n📁 Résultats sauvegardés dans:")
    print("   - results/plots/")
    print("   - results/checkpoints/")
    print("   - results/convergence_report.txt")


if __name__ == "__main__":
    # Vous pouvez exécuter un exemple spécifique:
    # example_1_basic_training()
    # example_2_compare_algorithms()
    # example_3_scalability_test()
    # example_4_sensitivity_analysis()
    # example_5_detailed_convergence_analysis()
    # example_6_checkpoint_and_resume()
    example_7_stochastic_environment()
    
    # Ou tous les exemples:
    # run_all_examples()