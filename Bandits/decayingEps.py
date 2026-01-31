import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from utils import create_bandit, play, resultBest
from epsGreedy import resultEpsilon0_1
from epsGreedy_optInit import resultEpsilon0_1Optimistic


def simulateDecayingEpsilon(epsilon,decay_factor,n=2000,k=10,tau=1000,optimistic=False):
    results = np.zeros(shape=(n,tau))
    for episode in tqdm(range(n), desc="episode"):
        bandit = create_bandit(k)
        Nt = np.zeros(k)
        Racc = np.zeros(k)
        Qt = np.zeros(k)
        if optimistic:
            for arm in range(k):
                Nt[arm] = 1
                Racc[arm] = 3.0
                Qt[arm] = 3.0
        for t in range(tau):
            arm = np.argmax(Qt)
            if (np.random.uniform() <= epsilon * np.exp(-t/decay_factor) ):    
                arm = np.random.choice(range(k))
            Nt[arm] += 1
            reward = play(bandit, arm)
            Racc[arm] += reward
            Qt[arm] = Racc[arm] / Nt[arm]
            results[episode][t] = reward
    return results

resultEpsilon0_1Decay1000 = simulateDecayingEpsilon(0.1, 1000)
resultEpsilon0_1Decay1000Optimistic = simulateDecayingEpsilon(0.1, 1000, optimistic=True)

plt.plot(resultEpsilon0_1.mean(axis=0))
plt.plot(resultEpsilon0_1Optimistic.mean(axis=0))
plt.plot(resultEpsilon0_1Decay1000.mean(axis=0))
plt.plot(resultEpsilon0_1Decay1000Optimistic.mean(axis=0))
plt.plot(resultBest.mean(axis=0))
plt.legend(['epsilon-greedy 0.1', 'epsilon-greedy 0.1 optimistic', 'decaying eps-greedy 0.1 factor=1000', 'decaying eps-greedy 0.1 factor=1000 optimistic', 'best'])