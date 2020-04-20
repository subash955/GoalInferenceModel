import numpy as np

def GoalInference(sequence, goals, goal_prob, policy,transition,TransitionProbability):
    goal_probabilities = [(g,1/len(g))for g in goals]
    total = 0
    for i,g in enumerate(goals):
        p_seq = 1
        for (j,s) in enumerate(sequence[:-1]):
            p_snext = 0
            for a,action_prob in policy[g][s]:
                p_snext += TransitionProbability(sequence[j+1],s,a)*action_prob
            p_seq *= p_snext
        p_seq*= goal_prob[g]
        total += p_seq
        goal_probabilities[i]= ((g,p_seq))


    return [y/total for x,y in goal_probabilities]