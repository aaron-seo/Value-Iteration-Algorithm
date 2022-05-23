import numpy as np
import drawHeatMap as hm
import rewardTable as rt
import transitionTable as tt

def expect(xDistribution, function):
    expectation=sum([function(x)*px for x, px in xDistribution.items()])
    return expectation

def getSPrimeRDistributionFull(s, action, transitionTable, rewardTable):
    reward=lambda sPrime: rewardTable[s][action][sPrime]
    p=lambda sPrime: transitionTable[s][action][sPrime]
    sPrimeRDistribution={(sPrime, reward(sPrime)): p(sPrime) for sPrime in transitionTable[s][action].keys()}
    return sPrimeRDistribution
    
def updateQFull(s, a, Q, getSPrimeRDistribution, gamma):
    def reward_function(s_prime_r):
        s_prime, r = s_prime_r
        Q_s_prime_a_prime = [Q[s_prime][a_prime] for a_prime in Q[s_prime].keys()]
        return r + gamma * max(Q_s_prime_a_prime)

    Qas = expect(getSPrimeRDistribution(s, a), reward_function) 

    return Qas

def qValueIteration(Q, updateQ, stateSpace, actionSpace, convergenceTolerance):
    QOld = Q.copy()
    QNew = Q.copy()

    Q_delta = np.Inf
    while not (Q_delta < convergenceTolerance):
        Q_delta = 0
        for s in stateSpace:
            for a in actionSpace:
                q = QOld.copy()
                QNew[s][a] = updateQ(s, a, QOld)
                Q_delta = max(Q_delta, abs(q[s][a] - QNew[s][a]))

    return QNew

def getPolicyFull(Q, roundingTolerance):
    max_Q = max(Q.values())

    max_actions = []
    for a in Q.keys():
        if abs(Q[a] - max_Q) < roundingTolerance:
            max_actions.append(a)

    policy = {a : 1 / len(max_actions) for a in max_actions}

    return policy


def viewDictionaryStructure(d, levels, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(levels[indent]) + ": "+ str(key))
        if isinstance(value, dict):
            viewDictionaryStructure(value, levels, indent+1)
        else:
            print('\t' * (indent+1) + str(levels[indent+1])+ ": " + str(value))

def main():
    
    minX, maxX, minY, maxY=(0, 3, 0, 2)

    
    
    actionSpace=[(0,1), (0,-1), (1,0), (-1,0)]
    stateSpace=[(i,j) for i in range(maxX+1) for j in range(maxY+1) if (i, j) != (1, 1)]
    Q={s:{a: 0 for a in actionSpace} for s in stateSpace}
    
    normalCost=-0.04
    trapDict={(3,1):-1}
    bonusDict={(3,0):1}
    blockList=[(1,1)]
    
    p=0.8
    transitionProbability={'forward':p, 'left':(1-p)/2, 'right':(1-p)/2, 'back':0}
    transitionProbability={move: p for move, p in transitionProbability.items() if transitionProbability[move]!=0}
    
    transitionTable=tt.createTransitionTable(minX, minY, maxX, maxY, trapDict, bonusDict, blockList, actionSpace, transitionProbability)
    rewardTable=rt.createRewardTable(transitionTable, normalCost, trapDict, bonusDict)

    
    print(getSPrimeRDistributionFull((3, 2), (-1, 0), transitionTable, rewardTable))
    print()
    print(getSPrimeRDistributionFull((2, 0), (1, 0), transitionTable, rewardTable))

    '''
    
    levelsReward  = ["state", "action", "next state", "reward"]
    levelsTransition  = ["state", "action", "next state", "probability"]
    
    viewDictionaryStructure(transitionTable, levelsTransition)
    viewDictionaryStructure(rewardTable, levelsReward)

    '''
        
    getSPrimeRDistribution=lambda s, action: getSPrimeRDistributionFull(s, action, transitionTable, rewardTable)
    gamma = 0.8       
    updateQ=lambda s, a, Q: updateQFull(s, a, Q, getSPrimeRDistribution, gamma)
    convergenceTolerance = 10e-7
    QNew=qValueIteration(Q, updateQ, stateSpace, actionSpace, convergenceTolerance)
    
    roundingTolerance= 10e-7
    getPolicy=lambda Q: getPolicyFull(Q, roundingTolerance)
    policy={s:getPolicy(QNew[s]) for s in stateSpace}

    V={s: max(QNew[s].values()) for s in stateSpace}
    
    VDrawing=V.copy()
    VDrawing[(1, 1)]=0
    VDrawing={k: v for k, v in sorted(VDrawing.items(), key=lambda item: item[0])}
    policyDrawing=policy.copy()
    policyDrawing[(1, 1)]={(1, 0): 1.0}
    policyDrawing={k: v for k, v in sorted(policyDrawing.items(), key=lambda item: item[0])}

    hm.drawFinalMap(VDrawing, policyDrawing, trapDict, bonusDict, blockList, normalCost)
    

    
    
    
if __name__=='__main__': 
    main()
