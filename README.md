# VALUE ITERATION ALGORITHM

## AIM:
To find an optimal policy for an agent navigating a grid-world with slippery tiles, aiming to reach a goal state while maximizing expected rewards using value iteration algorithm.

## PROBLEM STATEMENT:
The problem involves using the Value Iteration algorithm to find the best strategy for an agent in the Frozen Lake environment. The agent must navigate icy terrain, avoid hazards, and reach the goal while optimizing cumulative rewards in an uncertain environment.

## POLICY ITERATION ALGORITHM:
**Environment Setup:**
- The code begins by importing necessary libraries and setting up the Frozen Lake environment using Gym. It also initializes the initial state, goal state, and transition probabilities (P).

**Value Iteration Algorithm:**
- The core algorithm used in this code is Value Iteration.
**Value Initialization:**
- Initialize a value function (V) with zeros for each state.
**Value Iteration Loop:**
- For each state (s), calculate the Q-values for all possible actions (a) using the Bellman equation. The Q-value represents the expected cumulative reward when taking action 'a' from state 's'.
- Update the value function (V) by taking the maximum Q-value for each state.
- Check for convergence: If the maximum change in value function (V) is smaller than a threshold (theta), break the loop.
- After convergence, derive the optimal policy (pi) by selecting actions that maximize the Q-values.
**Results:**
- The code prints the optimal policy, its state-value function, the success rate of reaching the goal, and the mean undiscounted return of the optimal policy.

## VALUE ITERATION FUNCTION:
```
Developed By: Sai Eswar Kandukuri
Reg.No: 212221240020
```
```python3
desc=['FFSH','FFFH','GFHF','HFFH']
env = gym.make('FrozenLake-v1',desc=desc)
init_state = env.reset()
goal_state = 8
P = env.env.P
```
```python3
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_state,reward,done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if np.max(np.abs(V-np.max(Q,axis=1)))<theta:
        break
      V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return V, pi
```

## OUTPUT:
<img width="443" alt="1" src="https://github.com/saieswar1607/rl-value-iteration/assets/93427011/775d8589-0378-4be3-954c-a6f81ff65a68">
<br>
<img width="607" alt="2" src="https://github.com/saieswar1607/rl-value-iteration/assets/93427011/a44f6ff1-f20c-449e-9e50-266f95547e26">
<br>
<img width="430" alt="3" src="https://github.com/saieswar1607/rl-value-iteration/assets/93427011/af34f343-fe04-42ac-baba-846afa7f4ead">

## RESULT:
Thus, an optimal policy for an agent navigating a grid-world with slippery tiles, aiming to reach a goal state while maximizing expected rewards using value iteration algorithm is successfully implemented.
