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


class Predict:
    def __init__(self, next_state, list_of_states):
        self.next_state = next_state
        self.list_of_states = list_of_states


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        k = 0
        while k < self.iterations:
            updated_values = self.values.copy()

            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    updated_values[state] = self.find_max_q_value(state)

            k += 1
            self.values = updated_values

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
        # util.raiseNotDefined()

        result = 0
        for next_state, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            # Q[k+1](s, a) = alpha * [R(s, a, s') + discount * V(s')]
            result += probability * (self.mdp.getReward(state, action, next_state) +
                                     self.discount * self.getValue(next_state)
                                     )
        return result

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        max_q_value = self.find_max_q_value(state)
        best_action = None

        # We have to get the maximum q-state from all q-states, and after that we have to return its action
        for action in self.mdp.getPossibleActions(state):
            if max_q_value == self.computeQValueFromValues(state, action):
                best_action = action

        return best_action

    def find_max_q_value(self, state):
        # We have to get the maximum q-state from all q-states
        max_q_value = float('-inf')

        for action in self.mdp.getPossibleActions(state):
            if self.getQValue(state, action) > max_q_value:
                max_q_value = self.getQValue(state, action)

        return max_q_value

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

    def __init__(self, mdp, discount=0.9, iterations=1000):
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

        k = 0
        while k < self.iterations:
            index = k % len(self.mdp.getStates())
            state = self.mdp.getStates()[index]
            k += 1

            if self.computeActionFromValues(state):
                self.values[state] = self.computeQValueFromValues(state, self.computeActionFromValues(state))


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def find_max_q_value(self, state):
        # We have to get the maximum q-state from all q-states
        max_q_value = float('-inf')

        for action in self.mdp.getPossibleActions(state):
            if self.computeQValueFromValues(state, action) > max_q_value:
                max_q_value = self.computeQValueFromValues(state, action)

        return max_q_value

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        p_queue = util.PriorityQueue()

        predicts_list = []
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for next_state, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        found = False

                        for predict in predicts_list:
                            if predict.next_state == next_state:
                                predict.list_of_states.append(state)
                                found = True

                        if not found:
                            new_predict = Predict(next_state, [state])
                            predicts_list.append(new_predict)

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                p_queue.update(state, -abs(self.find_max_q_value(state) - self.values[state]))

        k = 0
        while k < self.iterations:
            if p_queue.isEmpty():
                break
            k += 1

            state = p_queue.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = self.find_max_q_value(state)

            for p in predicts_list:
                if not self.mdp.isTerminal(p.next_state):
                    diff = abs(self.find_max_q_value(p.next_state) - self.values[p.next_state])

                    if diff > self.theta:
                        p_queue.update(p.next_state, -diff)
