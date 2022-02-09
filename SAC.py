from vacuum import VacuumEnv
from policyNetwork import PolicyNetwork
from utilities import ReplayBuffer

env = VacuumEnv(10)
pol = PolicyNetwork(env.observation_space,env.action_space).act #Callable: State -> Int
replay = ReplayBuffer(10)

done = False
state = env.reset()
while not done:
    action = pol(state)
    next_state,reward,done,_ = env.step(action)
    replay.add_data( (state,action,reward,next_state,done) )
    state = next_state
    
    #Retrieve samples like this
    if replay.capacity_reached():
        sample = replay.sample(10)
        #print(sample)
    
print("Done")
