import numpy as np

def ComputeValueFunction(states,actions,reward,transition,gamma,TransitionProbability):
    ValueFunc = {s:0 for s in states}
    for s in states:
        ValueFunc[s] = 0
    error = 1
    while error > 0:
        error = 0
        for s in states:
            value = ValueFunc[s]
            ValueFunc[s] = UpdateValue(s,ValueFunc,actions(s),reward,transition,gamma,TransitionProbability)
            error = max(error, abs(value - ValueFunc[s]))
    return ValueFunc


def UpdateValue(s,ValueFunc,actions,reward,transition,gamma,TransitionProbability):
    NewValue = -1000
    for a in actions.keys():    
        expected =  TransitionProbability(transition(s,a),s,a)*(gamma*ValueFunc[transition(s,a)]+reward[s]) + (sum([TransitionProbability(transition(s,b),s,a)*(ValueFunc[transition(s,b)]+reward[transition(s,b)]) for b in actions.keys() if b != a]))
        if expected > NewValue:
            NewValue = expected
    return NewValue

def OptimalAction(s,ValueFunc,actions,beta,reward,transition,gamma,TransitionProbability):
    out = [(a,np.exp(beta*(gamma*TransitionProbability(transition(s,a),s,a)*ValueFunc[transition(s,a)]+reward[s]) + (sum([TransitionProbability(transition(s,b),s,a)*(ValueFunc[transition(s,b)]+reward[transition(s,b)]) for b in actions.keys() if b != a])))) for a in actions.keys()]
    return [(x,y/sum([y for x,y in out])) for x,y in out]
        
def ComputePolicy(states,actions, ValueFunc,reward,transition,gamma,beta,TransitionProbability):
    return {s:OptimalAction(s,ValueFunc,actions(s),beta,reward,transition,gamma,TransitionProbability) for s in states}