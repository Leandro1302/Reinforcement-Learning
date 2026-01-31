from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from utils import createTable, drawFirstCard, step, plotA, plotV, evaluateQ,  epsilonGreedy, computeMSE
from MCControl import optimalQ

def TDLambdaControl(n,lambd, optimalQ):
    N0 = 100
    
    N = createTable()
    Q = createTable()
    
    mse = []
       
    # run n episodes    
    for k in tqdm(range(n), desc="Performing TD(lambda) Control for lambda=" + str(lambd)): 
        # initialize k-th episode
        E = createTable()
                
        s = (drawFirstCard(), drawFirstCard())
        a = epsilonGreedy(s, N0, N, Q)
        N[(s,a)] += 1
        
        # sample k-th episode
        while not s == 'terminal':            
            sp, r = step(s, a)
        
            if not sp == 'terminal':
                ap = epsilonGreedy(sp, N0, N, Q)
                N[(sp,ap)] += 1
                tdError = r + Q[(sp,ap)] - Q[(s, a)]
            else:
                ap = None
                tdError = r - Q[(s, a)]
            E[(s,a)] += 1
            
            for di in range(1, 10+1):
                for pi in range(1, 21+1):
                    for ai in ['hit', 'stick']:
                        si = (di,pi)
                        alpha = (1 / (N[(si, ai)] + 1))
                        Q[(si, ai)] += alpha * tdError * E[(si, ai)]
                        E[(si, ai)] *= lambd
            
            s = sp
            a = ap
        
        if optimalQ:
            mse.append(computeMSE(Q, optimalQ))
            
    return Q, mse

def evalTdLambda(n):
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mses = []
    for lambd in lambdas:
        Q, mse = TDLambdaControl(n, lambd, optimalQ)
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


evalTdLambda(1000)
tdl_0_2_Q, _ = TDLambdaControl(100_000, 0.2, None)
plotV(tdl_0_2_Q)
plotA(tdl_0_2_Q)
evaluateQ(100000, tdl_0_2_Q)
