# Reinforcement learning with deep learning

2020.09-2020.11

with prof. HoHyen Park



## 1. Solve MDP with calculation (Dynamic Programming)

### 1-1. Policy Iteration

- Grid world example with policy iteration

- print sum of the rewards (return value) every episode

- 6 obstacle, 7x7 grid size

  

### 1-2. Value Iteration

- Grid world example with value iteration
- print Q-function value in every state
- print sum of the rewards (return value) every episode
- 6 obstacle, 7x7 grid size



## 2. Solve MDP with Learning (Reinforcement Learning)



### 2-1. SARSA

- Solve grid world example with SARSA
- print sum of the rewards with discount every episode
- Find problem in SARSA Algorithm and solve it (.pdf file)
- apply epsilon greedy and compare with before version (.pdf file)
- Change environment with Non-deterministic grid-world and describe (.pdf file).  ( P_{ss'}^a !=1, 상태변환확률 !=1 )



### 2-2. Q-Learning

- Solve grid world example with Q-Learning
- apply epsilon greedy and compare with before version (.pdf file)
- Change environment with Non-deterministic grid-world and describe (.pdf file).  ( P_{ss'}^a !=1 )



## 3. Solve MDP when dimension is big (function approximation)

가치기반의 강화학습

### 3-1. Deep SARSA

- Solve grid world with Deep SARSA (use deeplearning layer fcn)



### 3-2. DQN

- Solve cartpole with DQN
- Solve grid world with DQN

Document with compare Deep SARSA and DQN



## 4. Solve MDP with Policy approximation

정책기반의 강화학습

### 4-1. REINFORCE

- Solve cartpole with DQN
- Solve grid world with DQN



### 4-2 A2C

- Solve cartpole with DQN
- Solve grid world with DQN

Document with whole pipe line



