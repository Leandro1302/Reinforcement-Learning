from tqdm.notebook import tqdm
from utils import createTable, drawFirstCard, step, plotA, plotV, evaluateQ,  epsilonGreedy, computeMSE
from MCControl import optimalQ


def QLearning(n, optimalQ):
    N0 = 100 
    N = createTable()
    Q = createTable()
    
    mse = []
       
    # run n episodes    
    for k in tqdm(range(n), desc="Performing Q-Learning"): 
        
        # initialize k-th episode
        s = (drawFirstCard(), drawFirstCard())
        a = epsilonGreedy(s, N0, N, Q)
        N[(s,a)] += 1
        
        # sample k-th episode
        while not s == 'terminal':            
            sp, r = step(s, a)
        
            if not sp == 'terminal':
                ap = epsilonGreedy(sp, N0, N, Q)
                N[(sp,ap)] += 1
                qError = r + max(Q[(sp,'hit')], Q[(sp, 'stick')]) - Q[(s, a)] # r + max(∀a Q(s',a)) - Q(s,a)
            else:
                ap = None
                qError = r - Q[(s, a)]
            
            alpha = (1 / (N[(s, a)]))
            Q[(s, a)] += alpha * qError
            
            s = sp
            a = ap
        
        if optimalQ:
            mse.append(computeMSE(Q, optimalQ))
            
    return Q, mse



Q, mse = QLearning(1_000_000, optimalQ)
plotV(Q)
plotA(Q)
evaluateQ(1_000_000, Q)