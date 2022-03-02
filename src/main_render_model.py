import time
from typing import Union

from src.agents.rtac import RTAC
from src.agents.sac import SAC
from src.envs.custom_lunar_lander import CustomLunarLander
from src.utils.wrapper import RTMDP


def render_agent(agent: Union[SAC, RTAC]):
    env = agent.env
    while True:
        done = False
        num_steps = 0
        reward_sum = 0

        state = env.reset()
        while not done:
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            reward_sum += reward
            num_steps += 1
            time.sleep(0.01)

        print(f'Finished with reward {reward_sum} after {num_steps} steps')


def main():
    env = CustomLunarLander(step_size=0.2)
    # env = RTMDP(CustomLunarLander(step_size=0.5), 0)
    network_kwargs = {'num_layers': 2, 'hidden_size': 512, 'double_value': False, 'normalized': False}
    agent = SAC(env, network_kwargs=network_kwargs, use_target=False, use_device=False, entropy_scale=0.1)
    # agent = RTAC(env, network_kwargs=network_kwargs, use_target=False, use_device=False)
    agent.load_network('../model_data/CustomLunarLander/SAC-S51-T20')
    render_agent(agent)

if __name__ == '__main__':
    main()
