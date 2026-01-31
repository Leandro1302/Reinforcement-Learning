import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from utils import create_bandit, play, resultBest, resultRandom


def simulateEpsilonGreedy(epsilon,n=2000,k=10,tau=1000):
    results = np.zeros(shape=(n,tau))
    for episode in tqdm(range(n), desc="episode"):
        bandit = create_bandit(k)
        Nt = np.zeros(k)
        Racc = np.zeros(k)
        Qt = np.zeros(k)
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

resultEpsilon0 = simulateEpsilonGreedy(0.0)
resultEpsilon0_1 = simulateEpsilonGreedy(0.1)
resultEpsilon0_01 = simulateEpsilonGreedy(0.01)
resultEpsilon0_2 = simulateEpsilonGreedy(0.2)

plt.plot(resultRandom.mean(axis=0))
plt.plot(resultEpsilon0.mean(axis=0))
plt.plot(resultEpsilon0_01.mean(axis=0))
plt.plot(resultEpsilon0_1.mean(axis=0))
plt.plot(resultEpsilon0_2.mean(axis=0))
plt.plot(resultBest.mean(axis=0))
plt.legend(['random', 'greedy (epsilon=0)', 'epsilon-greedy 0.01', 'epsilon-greedy 0.1', 'epsilon-greedy 0.2', 'best'])