# multiAgents.py
# --------------
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
import random
import util

from game import Agent
from game import Directions
from util import manhattanDistance


def check_terminate(currentGameState, depth=None):
    if depth is None:
        return currentGameState.isWin() or currentGameState.isLose()
    else:
        return depth == 0 or currentGameState.isWin() or currentGameState.isLose()


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if successorGameState.getNumFood() == 0:
            return float('inf')

        nearest_ghost_distance = float('inf')

        for ghost in newGhostStates:
            if ghost.scaredTimer != 0:
                continue
            else:
                distance_ghost_vs_pacman = manhattanDistance(ghost.configuration.pos, newPos)
                if distance_ghost_vs_pacman < nearest_ghost_distance:
                    nearest_ghost_distance = distance_ghost_vs_pacman

        if nearest_ghost_distance == 0:
            return float('-inf')

        food_exist_grid = currentGameState.getFood()

        if not food_exist_grid[newPos[0]][newPos[1]]:
            nearest_food_distance = float('inf')

            for x in range(food_exist_grid.width):
                for y in range(food_exist_grid.height):
                    if not food_exist_grid[x][y]:
                        continue
                    else:
                        distances_of_food_from_pacman = manhattanDistance((x, y), newPos)
                        if distances_of_food_from_pacman < nearest_food_distance:
                            nearest_food_distance = distances_of_food_from_pacman
        else:
            nearest_food_distance = 0

        return 1 / (nearest_food_distance + 0.3) - 1 / (nearest_ghost_distance - 0.6)
        # return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


def find_legal_actions_successors(gameState, agentIndex):
    legal_actions = []
    for legal_action in gameState.getLegalActions(agentIndex):
        if legal_action != Directions.STOP:
            legal_actions.append(legal_action)

    legal_actions_successors = []
    for action in legal_actions:
        legal_actions_successors.append(gameState.generateSuccessor(agentIndex, action))

    return legal_actions, legal_actions_successors


def select_random_max_index(maximum, values):
    max_indices = []
    for i in range(0, len(values)):
        if values[i] == maximum:
            max_indices.append(i)

    return random.choice(max_indices)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def calculate_value(self, agent_index, gameState, depth):
        if check_terminate(gameState, depth):
            return self.evaluationFunction(gameState)

        agent_legal_actions, agent_legal_actions_successors = find_legal_actions_successors(gameState, agent_index)

        list_of_values = []
        for state in agent_legal_actions_successors:
            new_index = (agent_index + 1) % gameState.getNumAgents()
            list_of_values.append(self.calculate_value(new_index, state, depth - 1))

        if agent_index != 0:    # for ghosts
            return min(list_of_values)
        else:   # for pacman
            return max(list_of_values)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        pacman_legal_actions, pacman_legal_actions_successors = find_legal_actions_successors(gameState, 0)

        depth_of_minimax = gameState.getNumAgents() * self.depth

        successors_values = []
        for state in pacman_legal_actions_successors:
            successors_values.append(self.calculate_value(1, state, depth_of_minimax - 1))

        maximum_value = max(successors_values)

        random_index = select_random_max_index(maximum_value, successors_values)
        return pacman_legal_actions[random_index]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def calculate_value(self, agent_index, gameState, depth, a, b):
        if a is None:
            a = float('-inf')
        if b is None:
            b = float('inf')

        if check_terminate(gameState, depth):
            return self.evaluationFunction(gameState)

        # agent_legal_actions, agent_legal_actions_successors = find_legal_actions_successors(gameState, agent_index)
        agent_legal_actions = []
        for legal_action in gameState.getLegalActions(agent_index):
            if legal_action != Directions.STOP:
                agent_legal_actions.append(legal_action)

        if agent_index != 0:    # for ghosts
            value = float('inf')
            for action in agent_legal_actions:
                successor = gameState.generateSuccessor(agent_index, action)
                new_index = (agent_index + 1) % gameState.getNumAgents()
                value = min(value, self.calculate_value(new_index, successor, depth - 1, a, b))

                if min(value, a) < a:
                    return value

                b = min(b, value)
            return value
        else:    # for pacman
            value = float('-inf')
            for action in agent_legal_actions:
                successor = gameState.generateSuccessor(agent_index, action)
                new_index = (agent_index + 1) % gameState.getNumAgents()
                value = max(value, self.calculate_value(new_index, successor, depth - 1, a, b))

                if max(value, b) > b:
                    return value

                a = max(a, value)
            return value

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacman_legal_actions, pacman_legal_actions_successors = find_legal_actions_successors(gameState, 0)

        depth_of_minimax = gameState.getNumAgents() * self.depth

        if check_terminate(gameState, depth_of_minimax):
            return self.evaluationFunction(gameState)

        a = float('-inf')
        b = float('inf')
        max_value = float('-inf')
        best_action = None

        for action in pacman_legal_actions:
            successor = gameState.generateSuccessor(0, action)
            value = self.calculate_value(1 % gameState.getNumAgents(), successor, depth_of_minimax - 1, a, b)
            if value > max_value:
                max_value = value
                best_action = action

            a = max(a, value)

        return best_action
        # util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def calculate_value(self, agent_index, gameState, depth):
        import statistics

        if check_terminate(gameState, depth):
            return self.evaluationFunction(gameState)

        agent_legal_actions, agent_legal_actions_successors = find_legal_actions_successors(gameState, agent_index)

        list_of_values = []
        for state in agent_legal_actions_successors:
            new_index = (agent_index + 1) % gameState.getNumAgents()
            list_of_values.append(self.calculate_value(new_index, state, depth - 1))

        if agent_index != 0:
            return float(statistics.mean(list_of_values))
        else:
            return max(list_of_values)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        pacman_legal_actions, pacman_legal_actions_successors = find_legal_actions_successors(gameState, 0)

        depth_of_minimax = gameState.getNumAgents() * self.depth

        successors_values = []
        for state in pacman_legal_actions_successors:
            successors_values.append(self.calculate_value(1, state, depth_of_minimax - 1))

        maximum_value = max(successors_values)

        random_index = select_random_max_index(maximum_value, successors_values)
        return pacman_legal_actions[random_index]
        # util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if check_terminate(currentGameState):
        return currentGameState.getScore()

    ghosts, pacman = currentGameState.getGhostStates(), currentGameState.getPacmanPosition()

    distances_of_ghosts = []
    scared_timers_list = []

    for ghost in ghosts:
        distance_of_ghost_from_pacman = manhattanDistance(pacman, ghost.configuration.pos)
        distances_of_ghosts.append(distance_of_ghost_from_pacman)
        scared_timers_list.append(ghost.scaredTimer)

    distance_from_unscared_list = []
    distance_from_scared_list = []

    for distance in distances_of_ghosts:
        for timer in scared_timers_list:
            if timer != 0:
                if timer > 2:
                    distance_from_scared_list.append(distance)
            else:
                distance_from_unscared_list.append(distance)

    ghost_bonus = 0
    for distance in distance_from_scared_list:
        ghost_bonus += (300 / distance)

    ghost_penalty = 0
    for distance in distance_from_unscared_list:
        ghost_penalty += (500 / distance ** 2)

    foods_list = currentGameState.getFood().asList()

    manhattan_distance_list = []
    for food in foods_list:
        manhattan_distance_list.append((manhattanDistance(food, pacman), food))

    manhattan_nearest_food_list = []
    for distance, food in sorted(manhattan_distance_list)[:5]:
        manhattan_nearest_food_list.append(food)

    maze_nearest_food_list = []
    for food in manhattan_nearest_food_list:
        maze_nearest_food_list.append(manhattanDistance(food, pacman))

    food_bonus = 0
    for x in sorted(maze_nearest_food_list):
        food_bonus += (8.5 / x)

    return currentGameState.getScore() + ghost_bonus - ghost_penalty + food_bonus
    # util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
