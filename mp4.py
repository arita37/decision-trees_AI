#mp4
import sys
import time
import numpy as np
import copy
import operator

from collections import deque

BOARD_SIZE = 6

# Pretty printing
class bcolors:
    ENDC = '\033[0m'
    HEADER = '\033[95m'
    BLACK = '\033[30m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GREEN_BAK = '\033[42m'
    RED_BAK = '\033[41m'
    WHITE_BAK = '\033[47m'
    BLACK_BAK = '\033[40m'

# The moves as tuples
UP = (-1,0)
DOWN = (1,0)
LEFT = (0,-1)
RIGHT = (0,1)
MOVES = (UP, DOWN, LEFT, RIGHT)
#-------------------#

START_CELL = 's'
TERMINAL_CELLS = ('-1', '1')
WALL_CELL = 'w'
STATE_CELL = '_'
CORRECT_MOVE_PROB = 0.8
WRONG_MOVE_PROB = 0.1
DISCOUNT_FACTOR = 0.999    # Discount factor (used for policy iteration)
REWARD = -0.04     # Negative reward for each state
MAX_ITERATIONS = 1000

GAME = None
UTILITIES = None

# creates an empty utility vector, with all number to 0
def CreateEmptyUtilityVector():
    return [ [ 0 for _ in range(BOARD_SIZE) ] for _ in range(BOARD_SIZE) ]

# updates the given matrix with starting values
def PopulateInitialUtilities(ut):
    row = 0
    for x, y in zip(ut,GAME.theBoard):
        col = 0
        for c1, c2 in zip(x,y):
            if c2 == '1' or c2 == '-1':
                ut[row][col] = int(c2)
            col += 1
        row += 1

class Game(object):
    def __init__(self,theBoard):
        self.theBoard = theBoard
        self.cells = deque()
        self.GenerateCells()

    def GenerateCells(self):
        for i in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                self.cells.append(BoardCell((i,x),self.theBoard[i][x]))

    def GetBoardCellByIndex(self,idx):
        for cell in self.cells:
            if cell.index == idx:
                return cell
        return None

class BoardCell(object):
    def __init__(self, index, value, gen=True):
        self.index = index
        self.value = value
        self.neighbors = []
        
        if gen: self.LocateNeighbors()

    def __hash__(self):
        return hash(self.index)

    def __eq__(self,other):
        if self.index == other.index: return True
        else: return False

    def LocateNeighbors(self):
        print '\nWorking on: {}:{}'.format(self.index,self.value)
        for n in MOVES:
            if ((0 <= self.index[0]+n[0] < BOARD_SIZE) and \
                (0 <= self.index[1]+n[1] < BOARD_SIZE)):
                cellVal = GameWorld[self.index[0]+n[0]][self.index[1]+n[1]]
                tempCell = BoardCell((self.index[0]+n[0],self.index[1]+n[1]), cellVal, False)
                print "Adding neighbor: " + cellVal
                self.neighbors.append(tempCell)

    def GetReward(self):
        if self.value == STATE_CELL or self.value == START_CELL or self.value == WALL_CELL:
            return REWARD
        if self.value in TERMINAL_CELLS:
            return int(self.value) + REWARD

    def GenerateUtilityValue(self):
        maxUtility = None

        #if self.value in TERMINAL_CELLS:
        #    return self.GetReward()

        for n in self.neighbors:

            if n.value == WALL_CELL:
                continue

            utility = 0
            possibleStates = deque()
            possibleStates.append((n,CORRECT_MOVE_PROB))

            # 90 degree moves from our intended move == sketchy
            if (tuple(map(operator.sub, n.index, self.index)) == UP) or \
                (tuple(map(operator.sub, n.index, self.index)) == DOWN):
                sketchyMove1 = GAME.GetBoardCellByIndex(tuple(map(operator.add, self.index, LEFT)))
                sketchyMove2 = GAME.GetBoardCellByIndex(tuple(map(operator.add, self.index, RIGHT)))
            else:
                sketchyMove1 = GAME.GetBoardCellByIndex(tuple(map(operator.add, self.index, UP)))
                sketchyMove2 = GAME.GetBoardCellByIndex(tuple(map(operator.add, self.index, DOWN)))
            
            # if it's not a valid move, stay
            if sketchyMove1 and (sketchyMove1.value == STATE_CELL or sketchyMove1.value in TERMINAL_CELLS):
                    possibleStates.append((sketchyMove1,WRONG_MOVE_PROB))
            else:
                possibleStates.append((n,WRONG_MOVE_PROB))  #not a valid move -- stay put

            if sketchyMove2 and (sketchyMove2.value == STATE_CELL or sketchyMove2.value in TERMINAL_CELLS):
                possibleStates.append((sketchyMove2,WRONG_MOVE_PROB))
            else:
                possibleStates.append((n,WRONG_MOVE_PROB))  #not a valid move -- stay put

            # Bellman update pre-work
            for nextPos, prob in possibleStates:
                utility += prob * UTILITIES[nextPos.index[0]][nextPos.index[1]]
            if (maxUtility is None) or (utility > maxUtility): maxUtility = utility
            print '---> utility: {}'.format(utility)

        # Bellman update for this vector
        result = self.GetReward() + DISCOUNT_FACTOR * maxUtility
        print '---> final utility: {}'.format(utility)
        print '--> Bellman: {}'.format(result)
        return result


def ValueIteration():
    repeat = True
    iterations = 0
    control = 1

    global UTILITIES
    UTILITIES = CreateEmptyUtilityVector()
    newUv = CreateEmptyUtilityVector()
    PopulateInitialUtilities(UTILITIES)
    PopulateInitialUtilities(newUv)

    while(repeat):
        iterations += 1
        maxNorm = 0
        for state in sorted(GAME.cells):

            print '-'*75
            PrintGame()
            time.sleep(0)

            if state.value == WALL_CELL:
                continue

            v = state.GenerateUtilityValue()
            if v: maxNorm = max(maxNorm, abs(UTILITIES[state.index[0]][state.index[1]] - v))
            newUv[state.index[0]][state.index[1]] = v
            print '-----> MaxNorm: {}'.format(maxNorm)

            #print v
            #print max(maxNorm, abs(UTILITIES[state.index[0]][state.index[1]] - v))

            # update the master matrix
            UTILITIES = newUv

        # maxNorm -- pp. 654
        if maxNorm <= (1 - DISCOUNT_FACTOR)/DISCOUNT_FACTOR: repeat = False
        print '--> variance: {}'.format((1 - DISCOUNT_FACTOR)/DISCOUNT_FACTOR)

        if iterations >= MAX_ITERATIONS:
            repeat = False
            print 'reached iteration limit'

        if iterations % control == 0:
            print iterations
            for x in range(BOARD_SIZE):
                for i in range(BOARD_SIZE):
                    print '%.4f' % UTILITIES[x][i]
            control *= 2
            time.sleep(1)

    print iterations
    for x in range(BOARD_SIZE):
        for i in range(BOARD_SIZE):
            print '%.4f' % UTILITIES[x][i]
    return iterations

def PrintGame():
    for x in range(BOARD_SIZE):
        for i in range(BOARD_SIZE):
            print '%.4f   |' % UTILITIES[x][i],
        print '\n'

# just parsing the maze file
with open("maze.txt", "rtU") as f:
    # Primary data structure
    GameWorld = np.empty((BOARD_SIZE,BOARD_SIZE), dtype=object)

    x = 0
    y = 0
    terminalCells = set()

    # Populate `theProblem`
    for line in f:
        for ch in line.split():
            GameWorld[y,x] = ch
            if ch == START_CELL:
                startingNodeXY = (y,x)
                GameWorld[y,x] = STATE_CELL
                print "Start: {}".format(startingNodeXY)
            elif ch in TERMINAL_CELLS:
                terminalCells.add((y,x))
                print ch
            x += 1
        x = 0
        y += 1

x = 0
y = 0

# pretty printing if we get time
PrettyGameWorld = copy.deepcopy(GameWorld)
for line in PrettyGameWorld:
    for cell in line:
        if cell == '_':
            PrettyGameWorld[y,x] = bcolors.WHITE_BAK + '   ' + bcolors.ENDC
        elif cell == 'w':
            PrettyGameWorld[y,x] = bcolors.WHITE_BAK + bcolors.BLACK + '+++' + bcolors.ENDC
        elif cell == '-1':
            PrettyGameWorld[y,x] = bcolors.RED_BAK + ' x ' + bcolors.ENDC
        else:
            PrettyGameWorld[y,x] = bcolors.GREEN_BAK + ' ! ' + bcolors.ENDC
        x += 1
    x = 0
    y += 1

# Starting state
print '-'*24
print '\n'.join(''.join(str(cell) for cell in x) for x in PrettyGameWorld)
print '-'*24


#---- start real stuff ----#
GAME = Game(GameWorld)
print ValueIteration()
PrintGame()