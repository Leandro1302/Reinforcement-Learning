import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from utils import create_bandit, play, resultBest
from epsGreedy import resultEpsilon0_1

def simulateUCB(c,n=2000,k=10,tau=1000):
    results = np.zeros(shape=(n,tau))
    for episode in tqdm(range(n), desc="episode"):
        bandit = create_bandit(k)
        Nt = np.zeros(k)
        Racc = np.zeros(k)
        Qt = np.zeros(k)
        UCB = np.zeros(k)
        for arm in range(k):
            UCB[arm] = 3.0
        
        for t in range(tau):
            arm = np.argmax(UCB)
            
            Nt[arm] += 1
            reward = play(bandit, arm)
            Racc[arm] += reward
            Qt[arm] = Racc[arm] / Nt[arm]
            UCB[arm] = Qt[arm] + c * np.sqrt(np.log(t+1) / Nt[arm])
            results[episode][t] = reward
    return results

resultUCB_1 = simulateUCB(1.0)
resultUCB_2 = simulateUCB(2.0)
resultUCB_3 = simulateUCB(3.0)
plt.plot(resultEpsilon0_1.mean(axis=0))
plt.plot(resultUCB_1.mean(axis=0))
plt.plot(resultUCB_2.mean(axis=0))
plt.plot(resultUCB_3.mean(axis=0))
plt.plot(resultBest.mean(axis=0))
plt.legend(['epsilon-greedy 0.1', 'UCB c=1.0', 'UCB c=2.0', 'UCB c=3.0','best'])