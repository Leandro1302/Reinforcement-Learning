import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from utils import create_bandit, play, resultBest, resultRandom
from epsGreedy import resultEpsilon0, resultEpsilon0_1

def simulateEpsilonGreedyOptimistic(epsilon,n=2000,k=10,tau=1000):
    results = np.zeros(shape=(n,tau))
    for episode in tqdm(range(n), desc="episode"):
        bandit = create_bandit(k)
        Nt = np.zeros(k)
        Racc = np.zeros(k)
        Qt = np.zeros(k)
        for arm in range(k):
            Nt[arm] = 1
            Racc[arm] = 3.0
            Qt[arm] = 3.0
        for t in range(tau):
            arm = np.argmax(Qt)
            if (np.random.uniform() <= epsilon):    
                arm = np.random.choice(range(k))
            Nt[arm] += 1
            reward = play(bandit, arm)
            Racc[arm] += reward
            Qt[arm] = Racc[arm] / Nt[arm]
            results[episode][t] = reward
    return results

resultEpsilon0Optimistic = simulateEpsilonGreedyOptimistic(0.0)
resultEpsilon0_1Optimistic = simulateEpsilonGreedyOptimistic(0.1)

plt.plot(resultEpsilon0.mean(axis=0))
plt.plot(resultEpsilon0_1.mean(axis=0))
plt.plot(resultEpsilon0Optimistic.mean(axis=0))
plt.plot(resultEpsilon0_1Optimistic.mean(axis=0))
plt.plot(resultBest.mean(axis=0))
plt.legend(['greedy (epsilon=0)', 'epsilon-greedy 0.1', 'greedy (epsilon=0) optimistic', 'epsilon-greedy 0.1 optimistic', 'best'])