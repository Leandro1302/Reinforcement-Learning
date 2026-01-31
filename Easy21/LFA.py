import random 
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from tqdm.notebook import tqdm
from utils import createTable, drawFirstCard, step, plotA, plotV, evaluateQ,  epsilonGreedy, computeMSE, createFeatures
from MCControl import optimalQ

features = createFeatures() 
featureDim = 3 * 6 * 2

def phi(state, action):
    lfa = np.zeros(featureDim)

    d, p = state

    for i, feature in enumerate(features):
        di,pi,ai = feature
        if di[0] <= d and d <= di[1] and pi[0] <= p and p <= pi[1] and action == ai:
            lfa[i] = 1.0
    
    return lfa 

def computeQ(state, action, theta):
    return np.dot(phi(state, action), theta) 

def computeQTable(theta):
    Q = dict()
    for di in range(1, 10+1):
        for pi in range(1, 21+1):
            for ai in ['hit', 'stick']:
                si = (di,pi)
                Q[(si,ai)] = computeQ(si, ai, theta)
    return Q


def fixedEpsilonGreedy(epsilon, state, theta):
    if random.uniform(0,1) <= epsilon:
        return random.choice(['stick','hit'])
    
    if computeQ(state, 'hit', theta) > computeQ(state, 'stick', theta):
        return 'hit'
    else:
        return 'stick'
    

def TDLambdaControlLFA(n, lambd, optimalQ): # SARSA(lambda) con LFA
    epsilon = 0.05 # fixed
    alpha = 0.01 # fixed
    
    # init theta randomly
    mu, sigma = 0.0, 0.0
    theta = np.random.normal(mu, sigma, featureDim) # theta = 0
    
    mse = []
       
    # run n episodes    
    for k in tqdm(range(n), desc="Performing LFA TD(lambda) Control for lambda=" + str(lambd)): 

        # initialize k-th episode
        E = np.zeros(featureDim)
                
        s = (drawFirstCard(), drawFirstCard())
        a = fixedEpsilonGreedy(epsilon, s, theta)
        
        # sample k-th episode
        while not s == 'terminal':            
            sp, r = step(s, a)
           
            if not sp == 'terminal':
                ap = fixedEpsilonGreedy(epsilon, sp, theta)
                tdError = r + computeQ(sp, ap, theta) - computeQ(s, a, theta)
            else:
                ap = None
                tdError = r - computeQ(s, a, theta)
            
            E = lambd * E + phi(s, a)
            theta = theta + alpha * tdError * E
                
            a = ap
            s = sp
        
        if optimalQ:
            mse.append(computeMSE(computeQTable(theta), optimalQ))
            
    return theta, mse

def evalTdLFA(n):
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mses = []
    for lambd in lambdas:
        theta, mse = TDLambdaControlLFA(n, lambd, optimalQ)
        mses.append(mse)
        
    fig = plt.figure(figsize=(10,10))
    
    lc = fig.add_subplot(111)       
    lc.plot(mses[0])
    lc.plot(mses[-1])
    lc.set_xlabel('episode')
    lc.set_ylabel('mse(Q,Q*)')
    lc.legend(['lambda=0.0', 'lambda=1.0'])
    
    fig2 = plt.figure(figsize=(10,10))
    mp = fig2.add_subplot(111)
    mp.plot(lambdas, [mse[-1] for mse in mses])
    mp.set_xlabel('lambda')
    mp.set_ylabel('mse(Q,Q*)')

evalTdLFA(10000)
theta, mse = TDLambdaControlLFA(1_000_000, 0.3, None)
plotV(computeQTable(theta))
plotA(computeQTable(theta))
evaluateQ(100000, computeQTable(theta))