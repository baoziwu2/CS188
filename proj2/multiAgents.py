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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        food_score = 0
        for food in newFood.asList():
            food_score += 1 / (manhattanDistance(newPos, food) + 1)
        
        ghost_score = 0
        for ghost in newGhostStates:
            distance = manhattanDistance(newPos, ghost.getPosition())
            if distance <= 1:
                ghost_score -= 10
            else:
                ghost_score += 1 / (distance + 1)
        
        return successorGameState.getScore() + food_score + ghost_score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minmax(self, gameState: GameState, agentIndex: int, depth: int):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        compare_function = max if agentIndex == 0 else min
        best_score = float('-inf') if agentIndex == 0 else float('inf')
        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        next_depth = depth + 1 if next_agent == 0 else depth

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.minmax(successor, next_agent, next_depth)
            best_score = compare_function(best_score, score)
        
        return best_score
        

    def getAction(self, gameState: GameState):
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

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        best_score = float('-inf')
        best_action = None
        for action in gameState.getLegalActions():
            successor = gameState.generateSuccessor(0, action)
            score = self.minmax(successor, 1, 0)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def next_agent_depth_tuple(self, agentIndex: int, gameState: GameState, depth: int):
        return ((agentIndex + 1) % gameState.getNumAgents(), 
                (depth + 1) if (agentIndex + 1) % gameState.getNumAgents() == 0 else depth)

    def max_value(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        best_score = float('-inf')
        next_agent, next_depth = self.next_agent_depth_tuple(0, gameState, depth)

        for action in gameState.getLegalActions():
            successor = gameState.generateSuccessor(0, action)
            score = self.min_value(successor, next_agent, next_depth, alpha, beta)
            best_score = max(best_score, score)
            if best_score > beta:
                return best_score
            alpha = max(alpha, best_score)

        return best_score

    def min_value(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        best_score = float('inf')
        next_agent, next_depth = self.next_agent_depth_tuple(agentIndex, gameState, depth)

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if next_agent == 0:
                score = self.max_value(successor, next_agent, next_depth, alpha, beta)
            else:
                score = self.min_value(successor, next_agent, next_depth, alpha, beta)
            best_score = min(best_score, score)
            if best_score < alpha:
                return best_score
            beta = min(beta, best_score)

        return best_score

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        best_score = float('-inf')
        best_action = None
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions():
            successor = gameState.generateSuccessor(0, action)
            score = self.min_value(successor, 1, 0, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_expectation(self, gameState: GameState, agentIndex: int, depth: int):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        next_depth = depth + 1 if next_agent == 0 else depth

        if agentIndex == 0:  
            best_score = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                score = self.get_expectation(successor, next_agent, next_depth)
                best_score = max(best_score, score)
            return best_score
        else:  
            total_score = 0
            legal_actions = gameState.getLegalActions(agentIndex)
            num_actions = len(legal_actions)

            for action in legal_actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = self.get_expectation(successor, next_agent, next_depth)
                total_score += score

            return total_score / num_actions if num_actions > 0 else 0

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        best_score = float('-inf')
        best_action = None

        for action in gameState.getLegalActions():
            successor = gameState.generateSuccessor(0, action)
            score = self.get_expectation(successor, 1, 0)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    Firstly we want pacman the eat all the food as soon as possible,
    so we add a score for the number of food left and the distance to the nearest food.
    Secondly the capsules are somehow useful
    so we want pacman have trend to eat the capsule by considering the distance
    meanwhile pacman will be rewarded if it defeat the ghost
    so we want it to chase the ghost when the ghost is scared
    Note that the ghost is randomly moving, it will be more dangerous when it is pretty close to pacman
    therefore we punish pacman severely when the ghost is really close
    """
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    res = currentGameState.getScore()

    res -= 5 * len(foods.asList())
    for food in foods.asList():
        res += 1 / (manhattanDistance(pos, food) + 1)

    capsule_score = 0
    for capsule in capsules:
        capsule_score += 1 / (manhattanDistance(pos, capsule) + 1)
    res += 2 * capsule_score

    for ghost in ghostStates:
        dis = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            res += 10 / (dis + 1)
        else:
            if dis <= 1:
                res -= 20
            elif dis <= 3:
                res += 5 / (dis + 1)

    return res

# Abbreviation
better = betterEvaluationFunction
