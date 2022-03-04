# Real Time Reinforcement Learning

Implementation of Real-Time Markov Decision Process (RTMDP)
and Real-Time Actor-Critic (RTAC) as presented in the following paper:

https://arxiv.org/abs/1911.04448

Additionally, a variation of the LunarLander environment with a variable physic
step size is included for experiments.

## Installation
You need to either create an environment or update an existing environment.
**After** creating an environment you have to activate it:
### Create environment
```
conda env create -f environment.yml
```

### Update environment (if env exists)
```
conda env update -f environment.yml --prune
```

### Activate environment
```
conda activate real-time-rl
```

## Basic Usage

The experiments presented by us can be replicated by calling main on 
the command line. Secondly, the interface of ActorCritic agents can also be called 
directly. For the possible command line arguments take a look [here](src/main.py)

### Environment Wrappers
Firstly, create an environment to be solved.
```
env = gym.make('CartPole-v1')
env2 = CustomLunarLander(step_size=0.1)
```

Optionally, one can use wrappers to modify the environment. The RTMDP wrapper converts
a given environment into the real-time version. The PreviousActionWrapper adds the last
action to the state space without introducing the feed-through mechanism of the actions.

```
real_time_env = RTMDP(env)
extended_env = PreviousActionWrapper(env)
```

### Agents
The two agents to be used are SAC and RTAC. Both have the same interface with some
optional arguments (which differ between the interfaces).

```
sac = SAC(env, seed=0)
sac.train()
avg_reward = sac.evaluate()

rtac = RTAC(env, seed=0)
rtac.train()
avg_reward = rtac.evaluate()
```

It is also possible to track data during the training process:
```
rtac = RTAC(env)
performance_list = rtac.train(track_stats=True)
```

The neural network used by the agents can be specified using a keyword argument
dictionary:
```
network_kwargs = {'num_layers': 3, 'hidden_size': 128, 'normalized': True}
rtac = RTAC(env, network_kwargs=network_kwargs)
```
