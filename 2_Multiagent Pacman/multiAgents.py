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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        currCap = currentGameState.getCapsules()
        newCap = successorGameState.getCapsules()
        score = 0

        if action == "Stop":
            return -100

        #capDist = [9999]
        #for capsule in currCap:
        #    capDist = capDist + [manhattanDistance(capsule,newPos)+1]
        
        for scare in newScaredTimes:
            score += scare*2
        
        ghostDist = [9999]
        for ghost in newGhostStates:
            ghostDist += [manhattanDistance(ghost.getPosition(),newPos)]
        
        foodDist = [9999]
        foodList = newFood.asList()
        for food in foodList:
            foodDist += [manhattanDistance(food,newPos)]
        
        #capCost = 1.0/min(capDist)
        foodCost = 1.0/min(foodDist)
        ghostCost = min(ghostDist)
        
        #score += ghostCost*(foodCost**4)*(capCost**4)
        score += ghostCost*(foodCost**4)

        score += successorGameState.getScore()
        
        #if newPos in currentGameState.getFood().asList():
        #    score = score + 1
        #print score
        return score
        #return successorGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        """
        "*** YOUR CODE HERE ***"
        def minimax(lastState, action, lastAgentIndex, depth):
            currentState = lastState.generateSuccessor(lastAgentIndex, action)
            
            if depth == 0 or currentState.isWin() or currentState.isLose():
                return self.evaluationFunction(currentState)
            else:
                numAgents = currentState.getNumAgents()
                currentAgentIndex = lastAgentIndex + 1
                if currentAgentIndex >= numAgents:
                    currentAgentIndex = 0
                    depth -= 1
                    if depth == 0:
                        return self.evaluationFunction(currentState)

                actions = currentState.getLegalActions(currentAgentIndex)
                if currentAgentIndex == 0:
                    value = max(map(lambda x: minimax(currentState, x, currentAgentIndex, depth), actions)) #Pacman returns the max of childrens
                else:
                    value = min(map(lambda x: minimax(currentState, x, currentAgentIndex, depth), actions)) #Ghost returns the min of childrens
					
                return value

        actions = gameState.getLegalActions(0)
        actionTake = "Stop"
        value = -99999
        for action in actions:
            tempValue = minimax(gameState, action, 0, self.depth)
            if tempValue > value:
                value = tempValue
                actionTake = action       
        return actionTake
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimax(alpha, beta, lastState, action, lastAgentIndex, depth):
            currentState = lastState.generateSuccessor(lastAgentIndex, action)
            if depth == 0 or currentState.isWin() or currentState.isLose():
                return self.evaluationFunction(currentState)
            else:
                numAgents = currentState.getNumAgents()
                currentAgentIndex = lastAgentIndex + 1
                if currentAgentIndex >= numAgents:
                    currentAgentIndex = 0
                    depth -= 1
                    if depth == 0:
                        return self.evaluationFunction(currentState)

                actions = currentState.getLegalActions(currentAgentIndex)
                
                if currentAgentIndex == 0:
                    value = -9999999
                    for action in actions:
                        tempValue = minimax(alpha, beta, currentState, action, currentAgentIndex, depth)
                        if tempValue > value:
                            value = tempValue
                        if value > beta:
                            break
                        if value > alpha:
                            alpha = value
                else:
                    value = 9999999
                    for action in actions:
                        tempValue = minimax(alpha, beta, currentState, action, currentAgentIndex, depth)
                        if tempValue < value:
                            value = tempValue
                        if value < alpha:
                            break
                        if value < beta:
                            beta = value
                        
                return value
        
        alpha = -9999999
        beta = 9999999
        value = -9999999
        actionTake = "Stop"
        actions = gameState.getLegalActions(0)
        for action in actions:
            tempValue = minimax(alpha, beta, gameState, action, 0, self.depth)
            if tempValue > value:
                value = tempValue
                actionTake = action
            if value > beta:
                break
            alpha = max(alpha, value)
		
        return actionTake
        #util.raiseNotDefined()

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
        def expectimax(lastState, action, lastAgentIndex, depth):
            currentState = lastState.generateSuccessor(lastAgentIndex, action)
            if depth == 0 or currentState.isWin() or currentState.isLose():
                return self.evaluationFunction(currentState)
            else:
                numAgents = currentState.getNumAgents()
                currentAgentIndex = lastAgentIndex + 1
                if currentAgentIndex >= numAgents:
                    currentAgentIndex = 0
                    depth -= 1
                    if depth == 0:
                        return self.evaluationFunction(currentState)
                
                actions = currentState.getLegalActions(currentAgentIndex)
                if currentAgentIndex == 0:
                    value = max(map(lambda x: expectimax(currentState, x, currentAgentIndex, depth), actions))
                else:
                    value = sum(map(lambda x: expectimax(currentState, x, currentAgentIndex, depth), actions))
                    value = value * 1.0 / len(actions)
                return value

        actions = gameState.getLegalActions(0)
        actionTake = "Stop"
        value = -99999
        for action in actions:
            tempValue = expectimax(gameState, action, 0, self.depth)
            if tempValue > value:
                value = tempValue
                actionTake = action       
        return actionTake
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList();
    foodNum = len(foodList)
    foodDist =  min([9999]+map(lambda x: util.manhattanDistance(pos, x), foodList))
    ghostPos = map(lambda x: x.getPosition(), currentGameState.getGhostStates())
    ghostDist = min(map(lambda x: util.manhattanDistance(pos, x), ghostPos))
    return score * 100 - foodNum * 5 - ghostDist * 1 - foodDist * 3
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

