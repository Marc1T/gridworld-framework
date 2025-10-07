# GridWorld Reinforcement Learning Framework

Un framework modulaire pour l'apprentissage par renforcement basé sur des environnements GridWorld, inspiré de Gymnasium.

## 🚀 Fonctionnalités

- **Environnement GridWorld** compatible avec l'API Gymnasium
- **Multiples algorithmes RL** implémentés :
  - 🎯 Policy Iteration
  - 💰 Value Iteration  
  - 🎲 Monte Carlo
  - 🤖 Q-learning
  - 🎪 Agent aléatoire (baseline)
- **Visualisations complètes** :
  - Courbes d'apprentissage
  - Fonctions de valeur V(s) et Q(s,a)
  - Politiques optimales
  - États du GridWorld en temps réel
- **Interface conviviale** pour configurer les environnements
- **Structure modulaire** et extensible

## 📦 Installation

### Installation directe

```bash
git clone https://github.com/votre-username/gridworld-rl-framework.git
cd gridworld-rl-framework
pip install -e .
```

### Dépendances

```bash
pip install numpy matplotlib seaborn gymnasium
```

## 🎯 Utilisation rapide

### Exemple basique

```python
from gridworld_framework import GridWorldEnv, QLearningAgent
from gridworld_framework.utils.visualization import plot_learning_curve

# Créer l'environnement
env = GridWorldEnv(
    grid_shape=(4, 4),
    initial_state=0,
    goal_state=15,
    obstacles=[5, 7, 11],
    render_mode='human'
)

# Créer et entraîner l'agent
agent = QLearningAgent(env, gamma=0.99, learning_rate=0.1)
history = agent.train(n_episodes=1000)

# Visualiser les résultats
plot_learning_curve(history)
```

### Exécuter les démonstrations

```python
from gridworld_framework.examples import compare_all_agents

# Comparer tous les algorithmes
env, agents, results = compare_all_agents()
```

## 📚 Algorithmes implémentés

### 🎯 Policy Iteration
Algorithme de planification qui alterne entre l'évaluation et l'amélioration de politique.

```python
from gridworld_framework import PolicyIterationAgent

agent = PolicyIterationAgent(env, gamma=0.99, theta=1e-6)
agent.train()
```

### 💰 Value Iteration  
Algorithme de planification qui calcule directement la fonction de valeur optimale.

```python
from gridworld_framework import ValueIterationAgent

agent = ValueIterationAgent(env, gamma=0.99, theta=1e-6)
agent.train()
```

### 🎲 Monte Carlo
Méthode d'apprentissage sans modèle basée sur des épisodes complets.

```python
from gridworld_framework import MonteCarloAgent

agent = MonteCarloAgent(
    env, 
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995
)
agent.train(n_episodes=1000)
```

### 🤖 Q-learning
Algorithme TD off-policy populaire pour l'apprentissage sans modèle.

```python
from gridworld_framework import QLearningAgent

agent = QLearningAgent(
    env,
    gamma=0.99,
    learning_rate=0.1,
    epsilon=1.0,
    epsilon_decay=0.998
)
agent.train(n_episodes=1000)
```

## 🎨 Visualisation

### Courbes d'apprentissage
```python
from gridworld_framework.utils.visualization import plot_learning_curve

history = agent.train(n_episodes=1000)
plot_learning_curve(history)
```

### Fonction de valeur et politique
```python
from gridworld_framework.utils.visualization import (
    plot_value_function,
    plot_policy,
    plot_q_function
)

# Fonction de valeur V(s)
plot_value_function(agent.get_value_function(), env.grid_shape)

# Politique optimale
plot_policy(agent.get_policy(), env.grid_shape)

# Fonction Q(s,a)
plot_q_function(agent.get_q_function(), env.grid_shape)
```

### Visualisation complète
```python
from gridworld_framework.utils.visualization import visualize_gridworld

visualize_gridworld(env, agent)
```

## ⚙️ Configuration de l'environnement

### Grille personnalisée
```python
env = GridWorldEnv(
    grid_shape=(5, 5),           # Taille de la grille
    initial_state=0,             # État initial
    goal_state=24,               # État objectif
    goal_fixed=True,             # Objectif fixe ou aléatoire
    obstacles=[6, 12, 18],       # États obstacles
    render_mode='matplotlib'     # Mode de rendu
)
```

### Matrices de transition personnalisées
```python
# Accéder aux matrices MDP
P = env.get_transition_matrix()  # P[s, a, s']
R = env.get_reward_matrix()      # R[s, a]

# Modifier les matrices si nécessaire
P[0, 0, 1] = 1.0  # Transition certaine
```

## 📊 Métriques et évaluation

### Évaluation simple
```python
results = agent.evaluate(n_episodes=100)
print(f"Reward moyen: {results['mean_reward']:.2f}")
```

### Comparaison d'agents
```python
from gridworld_framework.utils.metrics import compare_agents

agents = {
    'Q-learning': q_agent,
    'Monte Carlo': mc_agent,
    'Policy Iteration': pi_agent
}

results = compare_agents(agents, env, n_episodes=100)
```

## 🏗️ Structure du projet

```
gridworld_framework/
├── core/                 # Cœur du framework
│   ├── gridworld_env.py  # Environnement GridWorld
│   └── mdp.py           # Classe MDP
├── agents/              # Implémentations des algorithmes
│   ├── base_agent.py    # Classe de base
│   ├── random_agent.py
│   ├── policy_iteration.py
│   ├── value_iteration.py
│   ├── monte_carlo.py
│   └── q_learning.py
├── utils/               # Utilitaires
│   ├── visualization.py # Fonctions de visualisation
│   └── metrics.py       # Métriques et comparaisons
├── examples/            # Exemples d'utilisation
│   └── basic_usage.py
└── tests/               # Tests unitaires
```

## 🔧 Extension du framework

### Ajouter un nouvel agent
```python
from gridworld_framework.agents.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        # Initialisation personnalisée
    
    def act(self, state, explore=True):
        # Implémentation de la sélection d'action
        pass
    
    def update(self, state, action, reward, next_state, done):
        # Implémentation de la mise à jour
        pass
```

### Créer un environnement personnalisé
```python
from gridworld_framework.core.gridworld_env import GridWorldEnv

class MyCustomGridWorld(GridWorldEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Personnalisations
    
    def _setup_mdp(self):
        # Logique MDP personnalisée
        pass
```

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- Inspiré par [Gymnasium](https://gymnasium.farama.org/)
- Basé sur les principes de [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) de Sutton et Barto
```

## 15. Fichier requirements.txt

**requirements.txt**
```txt
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
gymnasium>=0.28.0
```

## Résumé final

Voici la structure complète que nous avons créée :

```
gridworld_framework/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── gridworld_env.py
│   └── mdp.py
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── random_agent.py
│   ├── policy_iteration.py
│   ├── value_iteration.py
│   ├── monte_carlo.py
│   └── q_learning.py
├── utils/
│   ├── __init__.py
│   ├── visualization.py
│   └── metrics.py
├── examples/
│   ├── __init__.py
│   └── basic_usage.py
├── setup.py
├── requirements.txt
└── README.md
```

### Comment tester le framework :

1. **Installation** :
```bash
pip install -e .
```

2. **Test basique** :
```python
from gridworld_framework.examples import demo_random_agent
env, agent = demo_random_agent()
```

3. **Test complet** :
```python
from gridworld_framework.examples import compare_all_agents
env, agents, results = compare_all_agents()
```
