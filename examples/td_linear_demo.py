"""
Démonstration de l'agent TD(0) avec approximation linéaire.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from gridworld_framework.core.gridworld_env import GridWorldEnv
from gridworld_framework.agents.td_linear import TDLinearAgent
from gridworld_framework.utils.visualization import plot_learning_curve, plot_value_function


def demo_td_linear():
    """
    Démonstration de TD(0) avec approximation linéaire.
    """
    print("=== DÉMO TD(0) AVEC APPROXIMATION LINÉAIRE ===")
    
    # 1. Créer un environnement
    env = GridWorldEnv(
        grid_shape=(6, 6),
        initial_state=0,
        goal_state=35,
        obstacles=[7, 8, 13, 14, 19, 20],
        render_mode=None
    )
    
    # 2. Créer l'agent TD(0) linéaire
    agent = TDLinearAgent(
        env,
        feature_dim=15,        # Nombre de features
        learning_rate=0.05,    # Taux d'apprentissage pour les poids
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.997,
        epsilon_min=0.01
    )
    
    print(f"Dimension des features: {agent.feature_dim}")
    print(f"Dimension des poids w: {agent.w.shape}")
    print(f"Norme initiale de w: {np.linalg.norm(agent.w):.4f}")
    
    # 3. Entraînement
    print("\nDébut de l'entraînement TD(0)...")
    history = agent.train(n_episodes=2000, verbose=True)
    
    # 4. Visualisation des résultats
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Courbe d'apprentissage
    axes[0, 0].plot(history['rewards'], alpha=0.6, linewidth=1)
    window = 50
    if len(history['rewards']) >= window:
        moving_avg = np.convolve(history['rewards'], np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(history['rewards'])), moving_avg, 
                       linewidth=2, label=f'Moyenne ({window} épisodes)', color='red')
    axes[0, 0].set_xlabel('Épisode')
    axes[0, 0].set_ylabel('Récompense')
    axes[0, 0].set_title('Récompenses par épisode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Erreurs TD
    axes[0, 1].plot(history['td_errors'], alpha=0.6, linewidth=1)
    axes[0, 1].set_xlabel('Épisode')
    axes[0, 1].set_ylabel('Erreur TD moyenne')
    axes[0, 1].set_title('Erreur TD par épisode')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Importance des features
    feature_importance = agent.get_feature_importance()
    axes[1, 0].bar(range(len(feature_importance)), feature_importance)
    axes[1, 0].set_xlabel('Index de feature')
    axes[1, 0].set_ylabel('Importance')
    axes[1, 0].set_title('Importance des features')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Évolution des poids
    axes[1, 1].plot(history['convergence'], linewidth=2)
    axes[1, 1].set_xlabel('Épisode')
    axes[1, 1].set_ylabel('Norme des poids ||w||')
    axes[1, 1].set_title('Évolution des poids')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 5. Fonction de valeur apprise
    V_approx = agent.get_value_function()
    plot_value_function(V_approx, env.grid_shape, 
                       "TD(0) Linéaire - Fonction de valeur approximée")
    
    # 6. Comparaison avec la vraie fonction de valeur (si disponible)
    try:
        # Essayer de calculer la vraie fonction de valeur avec Value Iteration
        from gridworld_framework.agents.value_iteration import ValueIterationAgent
        vi_agent = ValueIterationAgent(env, gamma=0.99)
        vi_agent.train(verbose=False)
        V_true = vi_agent.get_value_function()
        
        # Erreur d'approximation
        mse = np.mean((V_approx - V_true)**2)
        print(f"\nErreur quadratique moyenne vs Value Iteration: {mse:.6f}")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Vraie fonction de valeur
        im1 = axes[0].imshow(V_true.reshape(env.grid_shape), cmap='viridis')
        axes[0].set_title('Vraie fonction de valeur (Value Iteration)')
        plt.colorbar(im1, ax=axes[0])
        
        # Fonction de valeur approximée
        im2 = axes[1].imshow(V_approx.reshape(env.grid_shape), cmap='viridis')
        axes[1].set_title('Fonction de valeur approximée (TD(0) Linéaire)')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Comparaison avec Value Iteration non disponible: {e}")
    
    # 7. Évaluation finale
    results = agent.evaluate(n_episodes=100)
    print(f"\nPerformance finale:")
    print(f"  - Reward moyen: {results['mean_reward']:.3f}")
    print(f"  - Longueur moyenne: {results['mean_length']:.1f}")
    print(f"  - Poids finaux ||w||: {np.linalg.norm(agent.w):.4f}")
    print(f"  - Epsilon final: {agent.epsilon:.4f}")
    
    return env, agent, history

"""
Test simple du modèle TD(0) Linéaire
"""

def test_td_linear_basic():
    """Test basique avec visualisation pas-à-pas"""
    
    print("🧪 TEST TD(0) LINÉAIRE - DÉMARRAGE")
    print("=" * 50)
    
    # 1. Créer un environnement simple
    env = GridWorldEnv(
        grid_shape=(4, 4),
        initial_state=0,
        goal_state=15,
        obstacles=[5, 7],
        render_mode='human'
    )
    
    # 2. Créer l'agent TD(0)
    agent = TDLinearAgent(
        env,
        feature_dim=10,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.8,  # Beaucoup d'exploration au début
        epsilon_decay=0.995
    )
    
    print("✅ Environnement et agent créés")
    print(f"📊 Dimension features: {agent.feature_dim}")
    print(f"🎯 Poids initiaux: {agent.w}")
    print(f"📈 Norme des poids: {np.linalg.norm(agent.w):.4f}")
    
    # 3. TEST: Vérifier les features
    print("\n🔍 TEST DES FEATURES:")
    for state in [0, 5, 15]:  # Début, obstacle, goal
        features = agent.features[state]
        value = agent.value(state)
        print(f"État {state}: V̂(s) = {value:.4f}")
        print(f"  Features: {[f'{f:.3f}' for f in features[:5]]}...")
    
    # 4. TEST: Un épisode d'entraînement avec logging
    print("\n🎮 TEST D'UN ÉPISODE COMPLET:")
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(20):
        # Afficher l'état avant action
        old_value = agent.value(state)
        print(f"\n--- Step {step} ---")
        print(f"État: {state}, V̂(s) = {old_value:.4f}")
        
        # Choisir action
        action = agent.act(state)
        action_names = ['↑', '→', '↓', '←']
        print(f"Action: {action} ({action_names[action]})")
        
        # Exécuter l'action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Afficher résultat
        print(f"→ État suivant: {next_state}, Reward: {reward}")
        print(f"Terminé: {terminated}, Truncated: {truncated}")
        
        # Afficher valeurs avant mise à jour
        next_value = agent.value(next_state) if not done else 0
        td_target = reward + agent.gamma * next_value
        td_error = td_target - old_value
        print(f"V̂(s): {old_value:.4f}, V̂(s'): {next_value:.4f}")
        print(f"TD target: {td_target:.4f}, TD error: {td_error:.4f}")
        
        # Mise à jour
        old_w = agent.w.copy()
        agent.update(state, action, reward, next_state, done)
        
        # Afficher changement des poids
        w_change = np.linalg.norm(agent.w - old_w)
        print(f"Δ poids: {w_change:.6f}")
        
        total_reward += reward
        state = next_state
        
        if done:
            print(f"🎉 Épisode terminé à l'étape {step}!")
            break
    
    print(f"\n📊 RÉSUMÉ ÉPISODE:")
    print(f"Reward total: {total_reward}")
    print(f"Poids finaux norm: {np.linalg.norm(agent.w):.4f}")
    print(f"Epsilon final: {agent.epsilon:.4f}")
    
    return env, agent

if __name__ == "__main__":
    # env, agent, history = demo_td_linear()
    env, agent = test_td_linear_basic()
