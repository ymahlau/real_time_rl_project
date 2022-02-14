import math
import unittest

from src.agents.sac import SAC
from src.envs.probe_envs import ConstRewardEnv, PredictableRewardEnv, TwoActionsTwoStates
from src.utils.utils import evaluate_policy


class TestSAC(unittest.TestCase):

    def test_value_function(self):
        delta = 0.05

        env = ConstRewardEnv()
        sac = SAC(env, alpha=0.2, lr_val=0.0003, replay_size=1, batch_size=1, hidden_size=256, num_hidden=2)
        sac.learn(iterations=100)
        self.assertAlmostEqual(1, sac.value([0, 0]).item(), delta=delta)

        env = PredictableRewardEnv()
        sac = SAC(env, alpha=0.2, lr_val=0.0003, replay_size=1, batch_size=1, hidden_size=256, num_hidden=2)
        sac.learn(iterations=100)
        self.assertAlmostEqual(1, sac.value([1, 0]).item(), delta=delta)
        self.assertAlmostEqual(-1, sac.value([-1, 0]).item(), delta=delta)

    def test_policy(self):
        delta = 0.2

        # Check if optimal policy can be adopted
        env = TwoActionsTwoStates()
        sac = SAC(env, alpha=0.2, gamma=1, lr_pol=0.003, lr_val=0.03, replay_size=200, batch_size=16,
                  hidden_size=256, num_hidden=2)
        sac.learn(iterations=1000)
        avg = evaluate_policy(sac.policy.act, env, trials=1000, rtmdp_ob=False)
        self.assertAlmostEqual(2, avg, delta=delta)
        self.assertAlmostEqual(2, sac.value([0, 0]).item(), delta=2*delta)
        self.assertAlmostEqual(2, sac.value([1, 0]).item(), delta=2*delta)
        self.assertAlmostEqual(1, sac.value([2, 1]).item(), delta=delta)
        self.assertAlmostEqual(1, sac.value([3, 0]).item(), delta=delta)

        # Check if random policy is adopted when entropy is valued extremely highly
        env = TwoActionsTwoStates()
        sac = SAC(env, alpha=10, gamma=1, lr_pol=0.003, lr_val=0.03, replay_size=200, batch_size=16,
                  hidden_size=256, num_hidden=2)
        sac.learn(iterations=1000)
        avg = evaluate_policy(sac.policy.act, env, trials=1000, rtmdp_ob=False)

        self.assertAlmostEqual(1.5, avg, delta=delta)
        self.assertAlmostEqual(2 * 10 * math.log(2, math.e), sac.value([0, 0]).item(), delta=10)
        self.assertAlmostEqual(1 * 10 * math.log(2, math.e), sac.value([3, 0]).item(), delta=10)
