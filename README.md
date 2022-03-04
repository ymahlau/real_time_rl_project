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
optional arguments (which differ between the interfaces). If one wants to evaluate
the performance of the agent, a separate evaluation environment has to be given.
This is necessary, because when evaluating at specific time steps during training
the agent might be in the middle of an episode.

```
sac = SAC(env, eval_env=eval_env, seed=0)
sac.train()
avg_reward = sac.evaluate()

rtac = RTAC(env, eval_env=eval_env, seed=0)
rtac.train()
avg_reward = rtac.evaluate()
```

It is also possible to track data during the training process:
```
rtac = RTAC(env, eval_env=eval_env)
performance_list = rtac.train(track_stats=True)
```

The neural network used by the agents can be specified using a keyword argument
dictionary:
```
network_kwargs = {'num_layers': 3, 'hidden_size': 128, 'normalized': True}
rtac = RTAC(env, network_kwargs=network_kwargs)
```

## Experiments
For our Experiments, we used the default settings of the sac and rtac
agents given in the original paper. For the exact values one may look at the
implementations in the source code.

We used our consumer desktop computers for the experiments. The exact
specifications of our three computers can be found below:
1. Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz, 32GB RAM
2. Intel(R) Core(TM) i5-9600K CPU @ 3.70GHz 3.70 GHz, 32GB RAM
3. Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz 1.50 GHz 16GB RAM

We did not record exact runtimes of our experiments as the runtimes for
RTAC and SAC are the same (they do basically the same computation). On
average an experiment on CartPole for all five seeds took 30-40 min. The
experiments on LunarLander took much longer with 3-4h for all five seeds. To speed
up the computation we ran the experiments on multiple cpu-cores simultaneously. This is
already included in the runtimes given above, and therefore it is difficult to
determine the runtime of a single experiment. In total, we ran 11 experiments
on CartPole and 14 experiments on LunarLander resulting in a total runtime of
multiple days.



