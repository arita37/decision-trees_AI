#mp4_learner.py
import sys
import time
import random
import numpy as np
import copy
import operator
import math

from collections import deque

BOARD_SIZE = 6
MAZE_FILE = 'maze.txt'
POLICY_FILE = 'policy'

# Pretty printing
class bcolors:
    ENDC = '\033[0m'
    HEADER = '\033[95m'
    BLACK = '\033[30m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91;1m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GREEN_BAK = '\033[42m'
    RED_BAK = '\033[41;1m'
    WHITE_BAK = '\033[47m'
    BLACK_BAK = '\033[40;1m'
    CYAN_BAK = '\033[46m'
    YELLOW_BAK = '\033[43m'

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
DISCOUNT_FACTOR = 0.99    # Discount factor (used for policy iteration)
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
        self.currentPos = startingNodeXY
        self.cells = deque()
        self.gameMoveSequence = deque()
        self.GenerateCells()
        self.policy = dict()
        self.Q = dict()
        self.eps = .8
        self.gamma = .005

    def GenerateCells(self):
        for i in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                self.cells.append(BoardCell((i,x),self.theBoard[i][x]))

    def GetBoardCellByIndex(self,idx):
        for cell in self.cells:
            if cell.index == idx:
                return cell
        return None

    def ReadPolicyValues(self):
        with open(POLICY_FILE, 'r') as f:
            for x in range(BOARD_SIZE):
                y = 0
                for value in f:
                    self.policy[(x,y)] = value.rstrip('\n')
                    self.Q[(x,y)] = 0
                    y += 1
                    if y == BOARD_SIZE: break

    def PrintQ(self):
        prettyBoard = copy.deepcopy(self.Q)

        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                cell = prettyBoard[(x,y)]
                if float(cell) == 0:
                    prettyBoard[(x,y)] = '{}{}  {}   {}'.format(bcolors.BLACK,bcolors.BLACK_BAK,round(float(cell),2),bcolors.ENDC)
                elif float(cell) < 0 and float(cell) > -0.4:
                    prettyBoard[(x,y)] = '{}{} {}  {}'.format(bcolors.BLACK,bcolors.YELLOW_BAK,round(float(cell),2),bcolors.ENDC)
                elif float(cell) <= -0.4:
                    prettyBoard[(x,y)] = '{} {}  {}'.format(bcolors.RED_BAK,round(float(cell),2),bcolors.ENDC)
                else:
                    prettyBoard[(x,y)] = '{}{}  {}   {}'.format(bcolors.BLACK,bcolors.GREEN_BAK,round(float(cell),2),bcolors.ENDC)

        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                print '{}'.format(prettyBoard[(x,y)]).center(8) + '|',
            print '\n'

    def PrintPolicy(self):
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                print '{}'.format(self.policy[(x,y)]).center(8) + '|',
            print '\n'

    def MakeMove(self, move, val):
        r = random.random()
        nextState = (self.currentPos[0] + move[0], self.currentPos[1] + move[1])

        #print 'Attempting move from {} to {}'.format(self.currentPos, nextState)
        self.Q[self.currentPos] = val
        self.currentPos = nextState
        self.gameMoveSequence.append(nextState)

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
        #print '\nWorking on: {}:{}'.format(self.index,self.value)
        for n in MOVES:
            if ((0 <= self.index[0]+n[0] < BOARD_SIZE) and \
                (0 <= self.index[1]+n[1] < BOARD_SIZE)):
                cellVal = GameWorld[self.index[0]+n[0]][self.index[1]+n[1]]
                if cellVal == 'w': continue
                tempCell = BoardCell((self.index[0]+n[0],self.index[1]+n[1]), cellVal, False)
                #print "Adding neighbor: {}".format(tempCell.index)
                self.neighbors.append(tempCell)

def PrintGame():
    prettyBoard = copy.deepcopy(GAME.theBoard)
    y = 0
    x = 0
    for line in prettyBoard:
        for cell in line:
            if cell == '_':
                prettyBoard[y,x] = bcolors.WHITE_BAK + '   ' + bcolors.ENDC
            elif cell == 'w':
                prettyBoard[y,x] = bcolors.BLACK_BAK + bcolors.BLACK + '   ' + bcolors.ENDC
            elif cell == '-1':
                prettyBoard[y,x] = bcolors.RED_BAK + ' x ' + bcolors.ENDC
            else:
                prettyBoard[y,x] = bcolors.GREEN_BAK + ' ! ' + bcolors.ENDC
            x += 1
        x = 0
        y += 1

    for x in range(BOARD_SIZE):
        for i in range(BOARD_SIZE):
            if (x,i) == GAME.currentPos:
                print bcolors.CYAN_BAK + bcolors.BLACK + '@'.center(3) + bcolors.ENDC + ' ',
            else:
                print '{}'.format(prettyBoard[x][i]).center(8) + ' ',
        print '\n'

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# just parsing the maze file
with open(MAZE_FILE, "rtU") as f:
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


# Warning: It's messy from here down... #
#---- start real stuff ----#

GAME = Game(GameWorld)
GAME.ReadPolicyValues()

PrintGame()
GAME.PrintPolicy()

episodes = 300000

go = 1
q = None
maxQ = None
control = episodes * 0.01
lowest_rmse = (0,10)

for i in range(episodes):
    actions = GAME.GetBoardCellByIndex(GAME.currentPos).neighbors
    if random.random() > GAME.eps:
        move = random.choice(actions)
        maxQ = GAME.Q[(move.index[0],move.index[1])]
        move = (move.index[0] - GAME.currentPos[0],move.index[1] - GAME.currentPos[1])
        #print '--> False Step! '
    else:
        q = [GAME.Q[n.index] for n in actions]
        maxQ = max(q)
        count = q.count(maxQ)
        if count > 1:
            bestMove = [k for k in range(len(actions)) if q[k] == maxQ]
            move = random.choice(bestMove)
            move = (actions[move].index[0] - GAME.currentPos[0],actions[move].index[1] - GAME.currentPos[1])
        else:
            move = q.index(maxQ)
            move = (actions[move].index[0] - GAME.currentPos[0],actions[move].index[1] - GAME.currentPos[1])

    maxQ = float(GAME.policy[GAME.currentPos]) + GAME.gamma * float(maxQ)
    GAME.MakeMove(move, maxQ)
    
    print 'Actions: {}'.format(q)
    print 'Value: {}'.format(maxQ)
    print '-'*60
    print '+'*5 + '\tActual Board\t' + '+'*5
    PrintGame()
    print '-'*60
    print '+'*10 + '\tOur Current View\t' + '+'*10
    GAME.PrintQ()
    print '='*60

    if i % control == 0: 
        print 'restarting -- old: {}, new: {}'.format(control, int(control * 1.5))
        GAME.currentPos = startingNodeXY
        control = int(control * 1.4)

    policyVals = np.empty((BOARD_SIZE,BOARD_SIZE), dtype=object)
    QVals = np.empty((BOARD_SIZE,BOARD_SIZE), dtype=object)
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            policyVals[x][y] = float(GAME.policy[(x,y)])
            QVals[x][y] = float(GAME.Q[(x,y)])

    _rmse = rmse(policyVals, QVals)
    if _rmse < lowest_rmse[1]: 
        lowest_rmse = (i,_rmse)
        #print lowest_rmse
    print 'RMSE: {}'.format(_rmse)
    print 'Lowest RMSE: {} (loop#,rmse)'.format(lowest_rmse)

    if _rmse < 0.016:
        print 'Breaking on iteration {}'.format(i)
        break
    time.sleep(.1)

print '-'*60
print '+'*5 + '\tActual Board\t' + '+'*5
PrintGame()
print '-'*60
print '+'*10 + '\tOur Current View\t' + '+'*10
GAME.PrintQ()
print '='*60

