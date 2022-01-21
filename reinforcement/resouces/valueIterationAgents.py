# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            counter = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    counter[state] = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
            self.values = counter


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0.
        Lambda = self.discount
        for sPrime, T in self.mdp.getTransitionStatesAndProbs(state, action):
            R = self.mdp.getReward(state, action, sPrime)
            value += T * (R + Lambda * self.values[sPrime])
        return value


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        values = [self.computeQValueFromValues(state, action) for action in actions]
        maxIndex = values.index(max(values))
        return actions[maxIndex]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        S = len(states)
        for i in range(self.iterations):
            turn = i % S   # start updating states from 0 to S-1, then repeat
            state = states[turn]
            if not self.mdp.isTerminal(state) and self.mdp.getPossibleActions(state):
                self.values[state] = self.computeQValueFromValues(state, self.getPolicy(state))

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = util.Counter()

        for state in states:
            if not self.mdp.isTerminal(state):
                predecessors[state] = set()

        for state in states:    # step 1: find all the predecessors of the 'state'
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    T = self.mdp.getTransitionStatesAndProbs(state, action)
                    for nextState, prob in T:
                        if not self.mdp.isTerminal(nextState) and prob != 0:
                            predecessors[nextState].add(state)

        priority_queue = util.PriorityQueue()  # step 2: create an empty priority queue to keep each state's priority

        for state in states:    # step 3: push states into the priority queue with the priority of -diff
            if not self.mdp.isTerminal(state):
                Q = self.computeQValueFromValues(state, self.getPolicy(state))
                diff = abs(Q - self.values[state])
                priority_queue.push(state, -diff)

        for i in range(self.iterations):  # step 4: do iterations
            if priority_queue.isEmpty():
                return

            state = priority_queue.pop()  # note that the 'state' would not be a terminal state as we checked when we created the predecessors

            self.values[state] = self.computeQValueFromValues(state, self.getPolicy(state))

            for predecessor in predecessors[state]:  # update each predecessor's priority of the state
                Q = self.computeQValueFromValues(predecessor, self.getPolicy(predecessor))
                diff = abs(Q - self.values[predecessor])

                if diff > self.theta:
                    priority_queue.update(predecessor, -diff)
