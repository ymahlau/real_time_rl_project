import gym
import keyboard
from gym.utils.play import play

from src.envs.custom_lunar_lander import CustomLunarLander

up_pressed = False
left_pressed = False
right_pressed = False

def evaluate_keyboard_input() -> int:
    if keyboard.is_pressed('up'):
        return 2

    if keyboard.is_pressed('left'):
        return 1

    if keyboard.is_pressed('right'):
        return 3

    return 0

def play_custom_lunar_lander():
    env = CustomLunarLander(step_size=0.1)
    while True:
        cum_reward = 0
        num_steps = 0
        done = False
        env.reset()

        while not done:
            env.render()
            action = evaluate_keyboard_input()
            next_state, reward, done, info = env.step(action)

            # stats
            cum_reward += reward
            num_steps += 1

        print(f'Finished LunarLander with reward {cum_reward} after {num_steps} steps')


def play_lunar_lander():
    # play normal lunar_lander
    env = gym.make('LunarLander-v2')
    keys_to_action = {(ord('a'),): 1, (ord('d'),): 3, (ord('w'),): 2}
    play(env, zoom=2, keys_to_action=keys_to_action)


if __name__ == '__main__':
    # play_lunar_lander()

    play_custom_lunar_lander()
