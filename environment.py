import numpy as np

def InitializeReward(states,goal):
    return {s:(-1+2*int(s==goal))for s in states}


def TestingEnvironment(states,barriers,rows,cols):
    actionspace = {'up' : (0,1), 'down' : (0,-1), 'left' : (-1,0), 'right' : (1,0),'none' :(0,0)}
    barriers = barriers
    states = states
    maxrow = rows
    maxcol = cols

    def transition (state,action):
        next_state = tuple(np.asarray(state) + np.asarray(actionspace[action]))
        if (state,next_state) in barriers or (next_state,state) in barriers:
            return state
        return next_state

    def actions(new_state):
        actions = {}
        if new_state[0] > 1:
            actions['left'] = (-1,0)
        if new_state[0]  < maxcol:
            actions['right'] = (1,0)
        if new_state[1] > 1:
            actions['down'] = (0,-1)
        if new_state[1]  < maxrow:
            actions['up'] = (0,1)
        actions['none'] = (0,0)
        return actions

    def TransitionProbability(s_next,s,a):
        return int(transition(s,a) == s_next)
    return transition, actions, TransitionProbability

