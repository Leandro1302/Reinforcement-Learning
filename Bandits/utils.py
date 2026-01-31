import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

def create_bandit(k):
    return np.random.normal(loc=0.0, scale=1.0, size=(k))

def play(bandit, arm):
    return np.random.normal(loc=bandit[arm], scale=1.0)

def simulateBest(n=2000,k=10,tau=1000):
    results = np.zeros(shape=(n,tau))
    for episode in tqdm(range(n), desc="episode"):
        bandit = create_bandit(k)
        for t in range(tau):
            #arm = np.random.choice(range(k))
            reward = np.max(bandit)#play(bandit, arm)
            results[episode][t] = reward
    return results

resultBest = simulateBest()

def simulateRandom(n=2000,k=10,tau=1000):
    results = np.zeros(shape=(n,tau))
    for episode in tqdm(range(n), desc="episode"):
        bandit = create_bandit(k)
        for t in range(tau):
            arm = np.random.choice(range(k))
            reward = play(bandit, arm)
            results[episode][t] = reward
    return results

resultRandom = simulateRandom()