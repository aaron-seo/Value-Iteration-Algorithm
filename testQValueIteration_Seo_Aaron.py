import sys
sys.path.append("../src/")

import unittest
from ddt import ddt, data, unpack
import qValueIteration_Seo_Aaron as target_code 
import transitionTable as tt
import rewardTable as rt

# initializations

minX, maxX, minY, maxY=(0, 3, 0, 2)

actionSpace=[(0,1), (0,-1), (1,0), (-1,0)]
stateSpace=[(i,j) for i in range(maxX+1) for j in range(maxY+1) if (i, j) != (1, 1)]
Q={s:{a: 0 for a in actionSpace} for s in stateSpace}

normalCost=-0.04
trapDict={(3,1):-1}
bonusDict={(3,0):1}
blockList=[(1,1)]

p=0.6
transitionProbability={'forward':p, 'left':(1-p)/2, 'right':(1-p)/2, 'back':0}
transitionProbability={move: p for move, p in transitionProbability.items() if transitionProbability[move]!=0}

transitionTable=tt.createTransitionTable(minX, minY, maxX, maxY, trapDict, bonusDict, blockList, actionSpace, transitionProbability)
rewardTable=rt.createRewardTable(transitionTable, normalCost, trapDict, bonusDict)

@ddt
class TestQValueIteration(unittest.TestCase):
    def assertDictAlmostEqual(self, calculated_dict, expected_dict, places = 7):
        for key in calculated_dict.keys():
            self.assertAlmostEqual(calculated_dict[key], expected_dict[key], places = places)

    @data(((3, 2), (-1, 0), transitionTable, rewardTable, {((0, 0), -0.04): 0, ((0, 1), -0.04): 0, ((0, 2), -0.04): 0, ((1, 0), -0.04): 0, ((1, 2), -0.04): 0, ((2, 0), -0.04): 0, ((2, 1), -0.04): 0, ((2, 2), -0.04): 0.6, ((3, 0), 1): 0, ((3, 1), -1): 0.2, ((3, 2), -0.04): 0.2}), ((2, 0), (1, 0), transitionTable, rewardTable, {((0, 0), -0.04): 0, ((0, 1), -0.04): 0, ((0, 2), -0.04): 0, ((1, 0), -0.04): 0, ((1, 2), -0.04): 0, ((2, 0), -0.04): 0.2, ((2, 1), -0.04): 0.2, ((2, 2), -0.04): 0, ((3, 0), 1): 0.6, ((3, 1), -1): 0, ((3, 2), -0.04): 0}))
    @unpack
    def test_getSPrimeRDistributionFull(self, s, action, transitionTable, rewardTable, expected_result):
        calculated_result = target_code.getSPrimeRDistributionFull(s, action, transitionTable, rewardTable)
        self.assertDictAlmostEqual(calculated_result, expected_result, places = 4)

    @data(((3, 2), (-1, 0), Q, lambda s, action: target_code.getSPrimeRDistributionFull(s, action, transitionTable, rewardTable), 0.5, -0.232), ((2, 0), (1, 0), Q, lambda s, action: target_code.getSPrimeRDistributionFull(s, action, transitionTable, rewardTable), 0.8, 0.584))
    @unpack
    def test_updateQFull(self, state, action, Q, getSPrimeRDistribution, gamma, expected_result):
        calculated_result = target_code.updateQFull(state, action, Q, getSPrimeRDistribution, gamma)
        self.assertAlmostEqual(calculated_result, expected_result, places = 4)

    @data((Q[(3, 2)], 0.1, {(0, 1) : 0.25, (0, -1) : 0.25, (1, 0) : 0.25, (-1, 0) : 0.25}), (Q[(3, 2)], 0.05, {(0, 1) : 0.25, (0, -1) : 0.25, (1, 0) : 0.25, (-1, 0) : 0.25}))
    @unpack
    def test_getPolicyFull(self, Q, roundingTolerance, expected_result):
        calculated_result = target_code.getPolicyFull(Q, roundingTolerance)
        self.assertDictAlmostEqual(calculated_result, expected_result, places = 4)

    def teardown(self):
        pass

if __name__ == "__main__":
    unittest.main(verbosity = 2)
