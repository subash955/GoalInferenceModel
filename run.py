
import environment as env
import valueIteration as vi
import model1 as model
import numpy as np


#Executes the model
def main(goals,gamma,b,maxrow,maxcol,barriers,sequence):
    goal_prob = {g:1/len(goals) for g in goals}
    states = [(x,y) for x in range(1,maxcol+1) for y in range(1,maxrow+1)]
    transition, actions, TransitionProbability = env.TestingEnvironment(states, barriers,maxrow,maxcol)
    rewards = {g:env.InitializeReward(states,g) for g in goals}
    value = {g:vi.ComputeValueFunction(states,actions,rewards[g],transition,gamma,TransitionProbability) for g in goals}
    policy = {g:vi.ComputePolicy(states,actions,value[g],rewards[g],transition,gamma,b,TransitionProbability) for g in goals}
    return model.GoalInference(sequence, goals, goal_prob, policy,transition,TransitionProbability)

#Tests
def TestPrediction(goals,gamma,b,maxrow,maxcol,barriers,sequence,result):
    if np.argmax(np.asarray(main(goals,gamma,b,maxrow,maxcol,barriers,sequence))) == result:
       return "Prediction test passed"
    return "Prediction test failed"

def TestSingleState():
    if main([(1,1)],0.9,1,1,1,[],[(1,1)]) == [1.0]:
        return "Single state test passed"
    return "Single state test failed"

def TestBarriers(state,next_state,maxrow,maxcol,action):
    barriers = [(state,next_state)]
    states = [(x,y) for x in range(1,maxcol+1) for y in range(1,maxrow+1)]
    transition, actions, TransitionProbability = env.TestingEnvironment(states, barriers,maxrow,maxcol)
    if transition(state,action) == state:
        return "Barriers test passed"
    return "Barriers test failed"

def TestValueIteration(gamma,p,state,next_state,goal,barriers,maxrow,maxcol):
    states = [(x,y) for x in range(1,maxcol+1) for y in range(1,maxrow+1)]
    rewards = env.InitializeReward(states,goal)
    transition, actions, TransitionProbability = env.TestingEnvironment(states, barriers,maxrow,maxcol)
    value = vi.ComputeValueFunction(states,actions,rewards,transition,gamma,TransitionProbability)
    if value[state] == value[next_state]:
        return "Value Iteration test passed"
    return "Value Iteration test failed"
def TestPolicy(gamma,p,state,goal,barriers,maxrow,maxcol,beta,expected):
    states = [(x,y) for x in range(1,maxcol+1) for y in range(1,maxrow+1)]
    rewards = env.InitializeReward(states,goal)
    transition, actions, TransitionProbability = env.TestingEnvironment(states, barriers,maxrow,maxcol)
    value = vi.ComputeValueFunction(states,actions,rewards,transition,gamma,TransitionProbability)
    policy = vi.ComputePolicy(states,actions,value,rewards,transition,gamma,beta,TransitionProbability)
    if policy[state][np.argmax(np.asarray([y for x,y in policy[state]]))][0] in expected:
        return "Policy test passed"
    return "Policy test failed"

print(TestPrediction([(1,1),(4,1),(3,1)],0.9,1,5,7,[],[(4,3),(4,2),(4,1),(3,1),(2,1)],0))
print(TestPrediction([(4,4),(2,1),(1,1)],0.9,1,5,5,[],[(1,2),(2,2),(2,3),(3,3),(3,4),(3,4),(3,5)],0))
print(TestSingleState())
print(TestBarriers((1,1),(2,1),4,4,"right"))
print(TestValueIteration(0.9,1,(1,1),(3,3),(3,1),[],3,3))
print(TestPolicy(0.9,1,(1,1),(3,3),[],3,3,1,["up","right"]))
