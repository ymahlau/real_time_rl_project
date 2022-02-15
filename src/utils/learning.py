import gym

from src.agents import ActorCritic


def learn(
        env: gym.Env,
        agent: ActorCritic,
        discount: float = 0.99):
    pass

def evaluate_agent(
        env: gym.Env,
        agent: ActorCritic,
        discount: float = 0.99):
    pass
