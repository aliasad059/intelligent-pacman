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


from util import manhattanDistance, raiseNotDefined
from game import Directions
import random, util
from game import Agent


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

        from util import manhattanDistance as mDist
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFoodPos = (successorGameState.getFood()).asList()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPos = [ghostState.getPosition() for ghostState in newGhostStates]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        evaluatedValue = 0
        for foodDist in newFoodPos:
            dist = mDist(newPos, foodDist)
            if dist < 5:
                evaluatedValue += 6 - dist
            else:
                evaluatedValue += 1

        for ghostDist in newGhostPos:
            dist = mDist(newPos, ghostDist)
            if dist == 0:
                evaluatedValue = -1 * evaluatedValue
            elif dist < 5:
                evaluatedValue -= (6 - dist) * 1.5

        return successorGameState.getScore() + evaluatedValue + sum(newScaredTimes)


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


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

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
        agentsNum = gameState.getNumAgents()

        def isTerminalState(gameState, currentDepth, agentIndex):
            return currentDepth == self.depth or \
                   not gameState.getLegalActions(agentIndex)

        def maxValue(gameState, currentDepth, agentIndex):
            bestResult = [-999999]

            if isTerminalState(gameState, currentDepth, agentIndex):
                bestResult[0] = self.evaluationFunction(gameState)
                return bestResult

            for a in gameState.getLegalActions(agentIndex=agentIndex):
                v = bestResult[0]
                nextResult = minValue(gameState.generateSuccessor(agentIndex=agentIndex, action=a),
                                      currentDepth,
                                      agentIndex=1)
                next_v = nextResult[0]
                if v < next_v:
                    bestResult[0] = next_v
                    try:
                        bestResult[1] = a
                    except IndexError:  # for the first iteration, we haven't pushed the action before
                        bestResult.append(a)

            return bestResult

        def minValue(gameState, currentDepth, agentIndex):
            bestResult = [9999999]

            if isTerminalState(gameState, currentDepth, agentIndex):
                bestResult[0] = self.evaluationFunction(gameState)
                return bestResult

            if agentIndex < agentsNum - 1:  # then we should determine other ghosts' decision
                for a in gameState.getLegalActions(agentIndex=agentIndex):
                    v = bestResult[0]
                    nextResult = minValue(gameState.generateSuccessor(agentIndex=agentIndex, action=a),
                                          currentDepth,
                                          agentIndex=agentIndex + 1)
                    next_v = nextResult[0]
                    if next_v < v:
                        bestResult[0] = next_v
                        try:
                            bestResult[1] = a
                        except IndexError:  # for the first iteration, we haven't pushed the action before
                            bestResult.append(a)

            else:  # the next agent is pacman
                for a in gameState.getLegalActions(agentIndex=agentIndex):
                    v = bestResult[0]
                    nextResult = maxValue(gameState.generateSuccessor(agentIndex=agentIndex, action=a),
                                          currentDepth + 1,
                                          agentIndex=0)
                    next_v = nextResult[0]
                    if next_v < v:
                        bestResult[0] = next_v
                        try:
                            bestResult[1] = a
                        except IndexError:  # for the first iteration, we haven't pushed the action before
                            bestResult.append(a)

            return bestResult

        return maxValue(gameState, currentDepth=0, agentIndex=0)[1]  # second element is the best action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agentsNum = gameState.getNumAgents()

        def isTerminalState(gameState, currentDepth, agentIndex):
            return currentDepth == self.depth or \
                   not gameState.getLegalActions(agentIndex)

        def maxValue(gameState, currentDepth, agentIndex, alpha, beta):
            bestResult = [-999999]

            if isTerminalState(gameState, currentDepth, agentIndex):
                bestResult[0] = self.evaluationFunction(gameState)
                return bestResult

            for a in gameState.getLegalActions(agentIndex=agentIndex):
                v = bestResult[0]
                nextResult = minValue(gameState.generateSuccessor(agentIndex=agentIndex, action=a),
                                      currentDepth, 1, alpha, beta)
                next_v = nextResult[0]
                if v < next_v:
                    bestResult[0] = next_v
                    try:
                        bestResult[1] = a
                    except IndexError:  # for the first iteration, we haven't pushed the action before
                        bestResult.append(a)
                if bestResult[0] > beta:
                    return bestResult
                alpha = max(alpha, bestResult[0])

            return bestResult

        def minValue(gameState, currentDepth, agentIndex, alpha, beta):
            bestResult = [9999999]

            if isTerminalState(gameState, currentDepth, agentIndex):
                bestResult[0] = self.evaluationFunction(gameState)
                return bestResult

            if agentIndex < agentsNum - 1:  # then we should determine other ghosts' decision
                for a in gameState.getLegalActions(agentIndex=agentIndex):
                    v = bestResult[0]
                    nextResult = minValue(gameState.generateSuccessor(agentIndex=agentIndex, action=a),
                                          currentDepth, agentIndex + 1, alpha, beta)
                    next_v = nextResult[0]
                    if next_v < v:
                        bestResult[0] = next_v
                        try:
                            bestResult[1] = a
                        except IndexError:  # for the first iteration, we haven't pushed the action before
                            bestResult.append(a)
                    if bestResult[0] < alpha:
                        return bestResult
                    beta = min(beta, bestResult[0])

            else:  # the next agent is pacman
                for a in gameState.getLegalActions(agentIndex=agentIndex):
                    v = bestResult[0]
                    nextResult = maxValue(gameState.generateSuccessor(agentIndex=agentIndex, action=a),
                                          currentDepth + 1, 0, alpha, beta)
                    next_v = nextResult[0]
                    if next_v < v:
                        bestResult[0] = next_v
                        try:
                            bestResult[1] = a
                        except IndexError:  # for the first iteration, we haven't pushed the action before
                            bestResult.append(a)
                    if bestResult[0] < alpha:
                        return bestResult
                    beta = min(beta, bestResult[0])
            return bestResult

        # second element is the best action
        return maxValue(gameState, currentDepth=0, agentIndex=0, beta=9999999, alpha=-9999999)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
