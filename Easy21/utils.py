import random 
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from tqdm.notebook import tqdm

def drawCard():
    number = random.randint(1, 10)
    color = 'red' if random.random() <= 1/3 else 'black'
    return (number, color)

def drawFirstCard(): # always black
    number = random.randint(1, 10)
    return number

def step(state, action):
    dealerCard, playerSum = state
    if action == 'hit':
        cardNumber, cardColor = drawCard()
        if cardColor == 'black':
            playerSum += cardNumber
        else:
            playerSum -= cardNumber
        if playerSum < 1 or playerSum > 21:
            return ('terminal', -1)
        return ((dealerCard, playerSum), 0)
    else:
        dealerSum = dealerCard
        while dealerSum < 17:
            cardNumber, cardColor = drawCard()
            if cardColor == 'black':
                dealerSum += cardNumber
            else:
                dealerSum -= cardNumber
            if dealerSum < 1 or dealerSum > 21:
                return ('terminal', 1)
        if (playerSum > dealerSum):
            return ('terminal', 1)
        elif (playerSum < dealerSum):
            return ('terminal', -1)
        else:
            return ('terminal', 0)

def createTable():
    T = dict()
    for d in range(1, 10+1):
        for p in range(1, 21+1):
            for a in ['hit', 'stick']:
                T[((d,p),a)] = 0
    #T['terminal','hit'] = 0
    #T['terminal','stick'] = 0
    return T

def plotV(Q):

    def V(d, p):
        return max(Q[(d, p), 'hit'], Q[(d, p), 'stick'])

    fig = plt.figure(figsize=plt.figaspect(0.25)*4)
    ax = fig.add_subplot(111, projection='3d')
    
    dealer_showing = np.arange(1, 10+1)
    player_score = np.arange(1, 21+1)
    dealer_showing, player_score = np.meshgrid(dealer_showing, player_score)

    z = np.ndarray(shape=(21, 10))
    for d in range(1, 10+1):
        for p in range(1, 21+1):
            z[p-1][d-1] = V(d, p)
    
    surf = ax.plot_surface(dealer_showing, player_score, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #surf = ax.plot_wireframe(dealer_showing, player_score, z, color='black', linewidth=1, antialiased=True)
    
    ax.set_xlabel('dealerCard')
    ax.set_ylabel('playerSum')
    ax.set_zlabel('V*')
    ax.view_init(10, -45)
    #ax.view_init(90, 0)
    
    #ax.set_zlim(-1.0, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    plt.xticks(np.arange(1, 11))
    plt.yticks(np.arange(1, 22))

    plt.show()  


def plotA(Q):
    def A(d, p):
        return 1 if Q[(d, p), 'hit'] > Q[(d, p), 'stick'] else 0

    a = np.ndarray(shape=(21, 10))
    for d in range(1, 10+1):
        for p in range(1, 21+1):
            a[p-1][d-1] = A(d, p)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
            
    ax.imshow(a, aspect=10/21, extent=[1,10,1,21], cmap=cm.binary)
    ax.set_xlabel('dealerCard')
    ax.set_ylabel('playerSum')

def greedy(state, Q):
    if Q[(state, 'hit')] > Q[(state, 'stick')]:
        return 'hit'
    return 'stick'


def epsilonGreedy(state, N0, N, Q):
    epsilon = N0 / (N0 + N[(state, 'hit')] + N[(state, 'stick')])
            
    if random.random() <= epsilon:
        action = random.choice(['hit','stick'])
    else:
        if Q[(state,'hit')] > Q[(state, 'stick')]:
            action = 'hit'
        else:
            action = 'stick'
        
    return action
    
def evaluateQ(n, Q):
    outcomes = {-1: 0, 0: 0, 1: 0}
    # run n episodes    
    for k in tqdm(range(n), desc="Performing simulations"): 
        state = (drawFirstCard(), drawFirstCard())
        while not state == 'terminal':
            action = greedy(state, Q)
            state, reward = step(state, action)
            if state == 'terminal':
                outcomes[reward] += 1
    return outcomes

def computeMSE(Q, Qp):
    result = 0
    for d in range(1, 10+1):
        for p in range(1, 21+1):
            for a in ['hit', 'stick']:
                result += (Q[((d,p),a)] - Qp[((d,p),a)]) ** 2
    return result / (10 * 21 * 2)


def createFeatures():
    features = []
    
    dealerIntervalls = [(1,4), (4,7), (7,10)]
    playerIntervalls = [(1,6), (4,9), (7,12), (10,15), (13,18), (16,21)]
    actions = ["hit", "stick"]
    
    for di in dealerIntervalls:
        for pi in playerIntervalls:
            for action in actions:
                features.append((di, pi, action))

    return features

