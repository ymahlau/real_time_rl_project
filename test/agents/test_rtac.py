import math
import unittest

from src.agents.rtac import RTAC
from src.envs.probe_envs import ConstRewardEnv, PredictableRewardEnv, TwoActionsTwoStates
from src.utils.wrapper import RTMDP


class TestRTAC(unittest.TestCase):

    def test_value_const_env(self):
        env = RTMDP(ConstRewardEnv(), 0)
        initial_state = env.reset()

        rtac = RTAC(env, lr=0.01, buffer_size=10, batch_size=5)
        rtac.train(num_steps=1000)
        initial_state_approx = rtac.get_value(initial_state).item()
        self.assertAlmostEqual(1, initial_state_approx, places=5)

    def test_value_predictable_env(self):
        env = RTMDP(PredictableRewardEnv(), initial_action=0)
        rtac = RTAC(env, lr=0.01, buffer_size=100, batch_size=10)
        rtac.train(num_steps=1000)

        pos_state_approx = rtac.get_value(([1], 0)).item()
        neg_state_approx = rtac.get_value(([-1], 0)).item()

        self.assertAlmostEqual(1, pos_state_approx, places=5)
        self.assertAlmostEqual(-1, neg_state_approx, places=5)

    def test_policy_two_states_two_actions(self):
        delta = 0.2
        env = RTMDP(TwoActionsTwoStates(), initial_action=0)
        eval_env = RTMDP(TwoActionsTwoStates(), initial_action=0)
        rtac = RTAC(env, eval_env=eval_env, lr=0.01, buffer_size=100, batch_size=10)
        rtac.train(num_steps=4000)
        avg = rtac.evaluate()

        self.assertAlmostEqual(2, avg, delta=delta)

        self.assertAlmostEqual(2, rtac.get_value(([0], 0)).item(), delta=delta)
        self.assertAlmostEqual(2, rtac.get_value(([1], 0)).item(), delta=delta)
        self.assertAlmostEqual(1, rtac.get_value(([2], 1)).item(), delta=delta)
        self.assertAlmostEqual(1, rtac.get_value(([3], 0)).item(), delta=delta)

    def test_policy(self):
        # Check if random policy is adopted when entropy is valued extremely highly
        env = RTMDP(TwoActionsTwoStates(), initial_action=0)
        eval_env = RTMDP(TwoActionsTwoStates(), initial_action=0)
        rtac = RTAC(env, eval_env=eval_env, lr=0.01, buffer_size=100, batch_size=10, entropy_scale=100)
        rtac.train(num_steps=4000)
        avg = rtac.evaluate()

        self.assertAlmostEqual(1.5, avg, delta=0.2)
        self.assertAlmostEqual(2 * 100 * math.log(2, math.e), rtac.get_value(([0], 0)).item(), delta=30)
        self.assertAlmostEqual(1 * 100 * math.log(2, math.e), rtac.get_value(([3], 0)).item(), delta=30)

    def test_normalization_simple(self):
        env = RTMDP(ConstRewardEnv(), initial_action=0)
        network_kwargs = {'normalized': True, 'pop_art_factor': 0.5}
        rtac = RTAC(env, entropy_scale=1, lr=0.01, buffer_size=1, batch_size=1, network_kwargs=network_kwargs)

        rtac.train(num_steps=5000)
        normalized_value = rtac.get_value(([0], 0))
        unnormalized_value = rtac.network.unnormalize(normalized_value)
        self.assertAlmostEqual(0, normalized_value.item(), places=2)
        self.assertAlmostEqual(1, unnormalized_value.item(), places=4)

        self.assertAlmostEqual(0.001, rtac.network.scale.data.item(), places=4)
        self.assertAlmostEqual(1, rtac.network.shift.data.item(), places=4)

    def test_normalization_two_states(self):
        delta = 0.2
        env = RTMDP(PredictableRewardEnv(), initial_action=0)
        network_kwargs = {'normalized': True, 'pop_art_factor': 0.1}
        rtac = RTAC(env, entropy_scale=0.2, lr=0.01, buffer_size=100, batch_size=100, network_kwargs=network_kwargs)
        rtac.train(num_steps=5000)

        normalized_value_pos = rtac.get_value(([1], 0))
        unnormalized_value_pos = rtac.network.unnormalize(normalized_value_pos)
        normalized_value_neg = rtac.get_value(([-1], 0))
        unnormalized_value_neg = rtac.network.unnormalize(normalized_value_neg)

        self.assertAlmostEqual(1, normalized_value_pos.item(), delta=delta)
        self.assertAlmostEqual(1, unnormalized_value_pos.item(), places=2)
        self.assertAlmostEqual(-1, normalized_value_neg.item(), delta=delta)
        self.assertAlmostEqual(-1, unnormalized_value_neg.item(), places=2)

        self.assertAlmostEqual(1, rtac.network.scale.data.item(), delta=delta)
        self.assertAlmostEqual(0, rtac.network.shift.data.item(), delta=delta)
