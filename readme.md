# 🎮 GridWorld Reinforcement Learning Framework

Framework complet pour l'apprentissage et l'expérimentation d'algorithmes de Reinforcement Learning sur des environnements GridWorld.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-green)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📋 Table des matières

- [Caractéristiques](#-caractéristiques)
- [Installation](#-installation)
- [Structure du projet](#-structure-du-projet)
- [Démarrage rapide](#-démarrage-rapide)
- [Algorithmes implémentés](#-algorithmes-implémentés)
- [Analyses et visualisations](#-analyses-et-visualisations)
- [Exemples](#-exemples)
- [Documentation](#-documentation)

---

## ✨ Caractéristiques

### 🎯 **Environnements**
- GridWorld 2D personnalisable
- Support des obstacles
- Environnements déterministes et stochastiques
- Compatible API Gymnasium
- Multiples modes de rendu

### 🤖 **Algorithmes**
- **Programmation Dynamique**: Value Iteration, Policy Iteration
- **Monte Carlo**: First-Visit, Every-Visit
- **Temporal Difference**: Q-Learning, SARSA (à venir)
- **Avancé**: Double Q-Learning

### 📊 **Analyses**
- **Convergence**: Analyse détaillée de la convergence
- **Scalabilité**: Tests sur différentes tailles de grille
- **Sensibilité**: Analyse des hyperparamètres
- **Comparaison**: Benchmark entre algorithmes

### 💾 **Fonctionnalités**
- Sauvegarde/chargement d'agents (checkpoints)
- Visualisations interactives
- Rapports automatiques
- Tracking complet des métriques

---

## 🚀 Installation

### Prérequis
```bash
Python >= 3.8
```

### Installation des dépendances
```bash
pip install numpy matplotlib gymnasium
```

### Installation du framework
```bash
git clone https://github.com/Marc1T/gridworld-framework.git
cd gridworld_framework
pip install -e .
```

---

## 📁 Structure du projet

```
gridworld_framework/
├── core/                          # Cœur du framework
│   ├── __init__.py
│   ├── gridworld_env.py          # Environnement GridWorld
│   └── mdp.py                    # Classe MDP
│
├── agents/                        # Agents RL
│   ├── __init__.py
│   ├── base_agent.py             # Classe abstraite
│   ├── random_agent.py           # Agent aléatoire
│   ├── value_iteration.py        # Value Iteration
│   ├── policy_iteration.py       # Policy Iteration
│   ├── monte_carlo.py            # Monte Carlo
│   └── q_learning.py             # Q-Learning
│
├── utils/                         # Utilitaires
│   ├── __init__.py
│   ├── visualization.py          # Visualisations
│   ├── metrics.py                # Métriques
│   └── convergence_analysis.py   # Analyses de convergence
│
├── examples/                      # Exemples
│   ├── basic_usage.py
│   ├── compare_algorithms.py
│   └── example_complete.py
│
├── tests/                         # Tests unitaires
│   ├── test_env.py
│   └── test_agents.py
│
├── results/                       # Résultats (créé automatiquement)
│   ├── plots/
│   ├── checkpoints/
│   └── reports/
│
├── README.md
├── requirements.txt
└── setup.py
```

---

## 🎮 Démarrage rapide

### 1. Créer un environnement

```python
from gridworld_framework.core.gridworld_env import GridWorldEnv

env = GridWorldEnv(
    grid_shape=(5, 5),        # Grille 5x5
    initial_state=0,          # Départ en haut à gauche
    goal_state=24,            # Objectif en bas à droite
    obstacles=[7, 12, 13],    # États obstacles
    step_penalty=-0.01,       # Pénalité par pas
    goal_reward=1.0           # Récompense pour le but
)
```

### 2. Entraîner un agent

```python
from gridworld_framework.agents.q_learning import QLearningAgent

# Créer l'agent
agent = QLearningAgent(
    env,
    gamma=0.99,              # Facteur d'actualisation
    learning_rate=0.1,       # Taux d'apprentissage
    epsilon=1.0,             # Exploration initiale
    epsilon_decay=0.995      # Décroissance de l'exploration
)

# Entraîner
history = agent.train(n_episodes=1000, verbose=True)

# Évaluer
results = agent.evaluate(n_episodes=100)
print(f"Taux de succès: {results['success_rate']:.1f}%")
```

### 3. Visualiser les résultats

```python
from gridworld_framework.utils.visualization import (
    plot_learning_curve,
    plot_value_function,
    plot_policy
)

# Courbe d'apprentissage
plot_learning_curve(history)

# Fonction de valeur
plot_value_function(agent.V, env.grid_shape)

# Politique optimale
plot_policy(agent.policy, env.grid_shape, action_names=['↑','→','↓','←'])
```

### 4. Sauvegarder/charger

```python
# Sauvegarder
agent.save("checkpoints/my_agent.npz")

# Charger
agent.load("checkpoints/my_agent.npz")
```

---

## 🤖 Algorithmes implémentés

### 📚 Programmation Dynamique

#### **Value Iteration**
Calcule directement la fonction de valeur optimale.

```python
from gridworld_framework.agents.value_iteration import ValueIterationAgent

agent = ValueIterationAgent(env, gamma=0.99, theta=1e-6)
agent.train()
```

**Équation de Bellman optimale:**
```
V(s) = max_a Σ P(s'|s,a)[R(s,a) + γV(s')]
```

#### **Policy Iteration**
Alterne entre évaluation et amélioration de politique.

```python
from gridworld_framework.agents.policy_iteration import PolicyIterationAgent

agent = PolicyIterationAgent(env, gamma=0.99)
agent.train()
```

### 🎲 Monte Carlo

Apprend à partir d'épisodes complets.

```python
from gridworld_framework.agents.monte_carlo import MonteCarloAgent

agent = MonteCarloAgent(
    env, 
    gamma=0.99,
    epsilon=1.0,
    first_visit=True  # First-Visit ou Every-Visit
)
agent.train(n_episodes=5000)
```

### ⚡ Temporal Difference

#### **Q-Learning** (Off-Policy)
Apprend la fonction Q optimale directement.

```python
from gridworld_framework.agents.q_learning import QLearningAgent

agent = QLearningAgent(
    env,
    gamma=0.99,
    learning_rate=0.1,
    epsilon=1.0,
    use_double_q=False  # Double Q-Learning si True
)
agent.train(n_episodes=1000)
```

**Mise à jour Q-Learning:**
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

---

## 📊 Analyses et visualisations

### 1. Analyse de convergence

```python
from gridworld_framework.utils.convergence_analysis import (
    analyze_convergence,
    generate_convergence_report
)

# Analyser
analysis = analyze_convergence(agent, metric='Q')
print(f"Convergé: {analysis['converged']}")
print(f"Épisode: {analysis['convergence_episode']}")

# Rapport détaillé
report = generate_convergence_report(agent, save_path="report.txt")
print(report)
```

### 2. Test de scalabilité

```python
from gridworld_framework.utils.convergence_analysis import (
    test_gridworld_scalability,
    plot_scalability_results
)

# Tester sur différentes tailles
results = test_gridworld_scalability(
    agent_class=QLearningAgent,
    grid_sizes=[(4,4), (5,5), (6,6), (8,8), (10,10)],
    n_episodes=500,
    gamma=0.99
)

# Visualiser
plot_scalability_results(results)
```

### 3. Analyse de sensibilité

```python
from gridworld_framework.utils.convergence_analysis import (
    test_hyperparameter_sensitivity,
    plot_sensitivity_analysis
)

# Tester learning rates
results = test_hyperparameter_sensitivity(
    agent_class=QLearningAgent,
    env=env,
    param_name='learning_rate',
    param_values=[0.01, 0.05, 0.1, 0.3, 0.5, 0.9],
    n_episodes=500,
    n_trials=5
)

# Visualiser
plot_sensitivity_analysis(results, param_name='Learning Rate')
```

### 4. Comparaison d'algorithmes

```python
from gridworld_framework.utils.convergence_analysis import (
    plot_convergence_comparison
)

# Entraîner plusieurs agents
agents_histories = {
    'Value Iteration': vi_agent.get_history(),
    'Q-Learning': q_agent.get_history(),
    'Monte Carlo': mc_agent.get_history()
}

# Comparer
plot_convergence_comparison(agents_histories)
```

---

## 💡 Exemples

### Exemple 1: Environnement simple

```python
# Grille 4x4 classique
env = GridWorldEnv(grid_shape=(4, 4))
agent = QLearningAgent(env)
agent.train(n_episodes=500)
agent.evaluate(n_episodes=100)
```

### Exemple 2: Avec obstacles

```python
# Grille avec obstacles
env = GridWorldEnv(
    grid_shape=(6, 6),
    obstacles=[8, 9, 14, 15, 20, 21]
)
agent = QLearningAgent(env)
agent.train(n_episodes=1000)
```

### Exemple 3: Environnement stochastique

```python
# 10% de bruit (actions aléatoires)
env = GridWorldEnv(
    grid_shape=(5, 5),
    stochastic=True,
    noise=0.1
)
agent = QLearningAgent(env, learning_rate=0.1)
agent.train(n_episodes=2000)
```

### Exemple 4: Benchmark complet

Voir `examples/example_complete.py` pour un exemple complet incluant:
- Entraînement de plusieurs algorithmes
- Tests de scalabilité
- Analyses de sensibilité
- Visualisations complètes
- Sauvegarde des résultats

```bash
python examples/example_complete.py
```

---

## 📖 Documentation

### Classes principales

#### **GridWorldEnv**
```python
GridWorldEnv(
    grid_shape=(4, 4),           # Dimensions
    initial_state=0,             # État de départ
    goal_state=15,               # But
    obstacles=[],                # Liste d'obstacles
    stochastic=False,            # Transitions stochastiques
    noise=0.0,                   # Niveau de bruit
    step_penalty=-0.01,          # Coût par action
    goal_reward=1.0,             # Récompense du but
    obstacle_penalty=-1.0,       # Pénalité obstacle
    render_mode=None             # 'human', 'matplotlib', 'rgb_array'
)
```

#### **BaseAgent**
Méthodes communes à tous les agents:
- `train(n_episodes, max_steps, verbose)`
- `evaluate(n_episodes, render)`
- `act(state, explore)`
- `save(filepath)`
- `load(filepath)`
- `get_policy()`
- `get_value_function()`
- `get_q_function()`

---

## 🔬 Analyses disponibles

### Métriques de performance
- Récompense moyenne et écart-type
- Taux de succès
- Longueur moyenne des épisodes
- Vitesse de convergence
- Stabilité post-convergence

### Visualisations
- Courbes d'apprentissage
- Fonctions de valeur (V et Q)
- Politiques optimales
- Comparaisons entre algorithmes
- Tests de scalabilité
- Analyses de sensibilité

---

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer de nouvelles fonctionnalités
- Améliorer la documentation
- Ajouter des exemples

N'hésitez pas à :

1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Ouvrir une Pull Request
   
---

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---
## 🙏 Remerciements

- Inspiré par [Gymnasium](https://gymnasium.farama.org/)
- Basé sur les principes de [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) de Sutton et Barto

---

**Bon apprentissage ! 🚀**

---