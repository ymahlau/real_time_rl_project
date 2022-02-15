import unittest
import math

from src.agents import PolicyValueNetwork
from src.agents.rtac import RTAC
from src.envs.probe_envs import ConstRewardEnv, PredictableRewardEnv, TwoActionsTwoStates
from src.utils.utils import evaluate_policy
from src.utils.wrapper import RTMDP


class TestRTAC(unittest.TestCase):

    def test_value_const_env(self):
        delta = 0.01

        env = RTMDP(ConstRewardEnv(), 0)
        network = PolicyValueNetwork(2, 1)
        rtac = RTAC(env, network, lr=0.01, buffer_size=2, batch_size=2)
        rtac.train(num_episodes=100)
        self.assertAlmostEqual(1, rtac.network.value_network([0, 1]).item(), delta=delta)

    def test_value_predictable_env(self):
        env = RTMDP(PredictableRewardEnv(), 0)
        rtac = RTAC(env, lr=0.01, buffer_size=1, batch_size=1, hidden_size=256, num_hidden=2, shared_parameters=False)
        rtac.train(training_time=1)
        self.assertAlmostEqual(1, rtac.network.value_network([1, 1]).item(), delta=delta)
        self.assertAlmostEqual(-1, rtac.network.value_network([-1, 1]).item(), delta=delta)

    def test_policy(self):
        delta = 0.2

        # Check if optimal policy can be adopted
        env = RTMDP(TwoActionsTwoStates(), 0)
        rtac = RTAC(env, buffer_size=100, batch_size=8, hidden_size=256, num_hidden=2, shared_parameters=False)
        rtac.train(training_time=5)
        avg = evaluate_policy(rtac.network.policy_network.act, env, trials=1000)

        self.assertAlmostEqual(2, avg, delta=delta)
        self.assertAlmostEqual(2, rtac.network.value_network([0, 1, 0]).item(), delta=delta)
        self.assertAlmostEqual(2, rtac.network.value_network([1, 1, 0]).item(), delta=delta)
        self.assertAlmostEqual(1, rtac.network.value_network([2, 0, 1]).item(), delta=delta)
        self.assertAlmostEqual(1, rtac.network.value_network([3, 1, 0]).item(), delta=delta)

        # Check if random policy is adopted when entropy is valued extremely highly
        env = RTMDP(TwoActionsTwoStates(), 0)
        rtac = RTAC(env, buffer_size=200, batch_size=16, hidden_size=256, num_hidden=2, shared_parameters=False)
        rtac.train(training_time=5)
        avg = evaluate_policy(rtac.network.policy_network.act, env, trials=1000)

        self.assertAlmostEqual(1.5, avg, delta=delta)
        self.assertAlmostEqual(2 * 100 * math.log(2, math.e), rtac.network.value_network([0, 1, 0]).item(), delta=10)
        self.assertAlmostEqual(1 * 100 * math.log(2, math.e), rtac.network.value_network([3, 1, 0]).item(), delta=10)
