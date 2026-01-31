from tqdm.notebook import tqdm
from utils import createTable, drawFirstCard, step, plotA, plotV, evaluateQ, epsilonGreedy

def MCControl(n):
    N0 = 100
    
    N = createTable()
    Q = createTable()
    
    # run n episodes
    for k in tqdm(range(n), desc="Performing Monte-Carlo Control"): 
        state = (drawFirstCard(), drawFirstCard())
        
        # sample k-th episode
        episode = []
        while not state == 'terminal':
            action = epsilonGreedy(state, N0, N, Q)
                    
            nextState, reward = step(state, action)
           
            episode.append((state, action, reward))
            
            state = nextState
                
        # update Q-values using MC control from k-th episode
        for (state, action, reward) in episode:
            N[(state, action)] += 1
            Gt = episode[-1][2]
            alpha = 1 / (N[(state, action)] + 0)
            Q[(state, action)] += alpha * (Gt - Q[(state, action)])
    return Q


optimalQ = MCControl(10_000_000)

plotV(optimalQ)
plotA(optimalQ)

evaluateQ(100000, optimalQ)

optimalQ = MCControl(10_000_000)