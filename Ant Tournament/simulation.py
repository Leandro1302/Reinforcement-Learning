import pickle
from copy import copy

class Simulation:
    TEAM_A = 0
    TEAM_B = 1

    def __init__(self, antA, antB, rewardSignal, params, arenaPath, logfile=None):
        self.terminated = False
        self.time = 0

        self.pheromoneDecay = 0.99
        self.pheromoneThreshold = 0.01
        self.ants = [] 

        self.maxId = [0, 0]

        # 2 Teams
        self.antA = antA
        self.antB = antB

        self.rewardSignal = rewardSignal
        self.params = params
       
        # load the arena
        file = open(arenaPath, "r")

        self.gridWidth = int(file.readline())
        self.gridHeight = int(file.readline())

        # Create the grids
        self.foodGrid = [[0 for _ in range(self.gridWidth)] for _ in range(self.gridHeight)]
        self.obstacleGrid = [[0 for _ in range(self.gridWidth)] for _ in range(self.gridHeight)]
        self.nestGrid = [[[0 for _ in range(self.gridWidth)] for _ in range(self.gridHeight)], [[0 for _ in range(self.gridWidth)] for _ in range(self.gridHeight)]]
        self.pheromoneGrid = [[[0.0 for _ in range(self.gridWidth)] for _ in range(self.gridHeight)], [[0.0 for _ in range(self.gridWidth)] for _ in range(self.gridHeight)]]
        self.antGrid = [[[0.0 for _ in range(self.gridWidth)] for _ in range(self.gridHeight)], [[0.0 for _ in range(self.gridWidth)] for _ in range(self.gridHeight)]]

        x = 0
        y = 0
        while True:
            line = file.readline()
            if not line:
                break
            for c in line:
                if c == 'A':
                    self.nestGrid[Simulation.TEAM_A][x][y] = 1
                elif c == 'B':
                    self.nestGrid[Simulation.TEAM_B][x][y] = 1
                elif c == 'a':
                    id = self.maxId[Simulation.TEAM_A]
                    self.maxId[Simulation.TEAM_A] += 1
                    self.ants.append(self.antA(self, x, y, Simulation.TEAM_A, id))
                elif c == 'b':
                    id = self.maxId[Simulation.TEAM_B]
                    self.maxId[Simulation.TEAM_B] += 1
                    self.ants.append(self.antB(self, x, y, Simulation.TEAM_B, id))
                elif c == '#':
                    self.obstacleGrid[x][y] = 1
                elif c.isdigit():
                    self.foodGrid[x][y] = int(c)
                x += 1
            y += 1
            x = 0
        file.close()

        self.foodCount = [0, 0]

        if logfile:
            self.logfile = open(logfile, 'wb')
            pickle.dump(self.gridWidth, self.logfile)
            pickle.dump(self.gridHeight, self.logfile)
        else:
            self.logfile = None

    def updatePheromones(self):
        for i in range(self.gridWidth):
            for j in range(self.gridHeight):
                self.pheromoneGrid[Simulation.TEAM_A][i][j] *= self.pheromoneDecay
                self.pheromoneGrid[Simulation.TEAM_B][i][j] *= self.pheromoneDecay
                if (self.pheromoneGrid[Simulation.TEAM_A][i][j] < self.pheromoneThreshold):  
                    self.pheromoneGrid[Simulation.TEAM_A][i][j] = 0.0
                if (self.pheromoneGrid[Simulation.TEAM_B][i][j] < self.pheromoneThreshold):  
                    self.pheromoneGrid[Simulation.TEAM_B][i][j] = 0.0
    
    def updateFoodCounts(self):
        self.foodCount = [0, 0]
        for x in range(self.gridWidth):
            for y in range(self.gridHeight):
                self.foodCount[Simulation.TEAM_A] += self.nestGrid[Simulation.TEAM_A][x][y] * self.foodGrid[x][y]
                self.foodCount[Simulation.TEAM_B] += self.nestGrid[Simulation.TEAM_B][x][y] * self.foodGrid[x][y]

    def computeFoodCount(self, team):
        result = 0
        for x in range(self.gridWidth):
            for y in range(self.gridHeight):
                result += self.nestGrid[team][x][y] * self.foodGrid[x][y]
        return result

    def updateLogFile(self):
        if not self.logfile:
            return
        
        pickle.dump(self.foodGrid, self.logfile)
        pickle.dump(self.obstacleGrid, self.logfile)
        pickle.dump(self.nestGrid, self.logfile)
        pickle.dump(self.pheromoneGrid, self.logfile)

        ants = [[],[]]
        for ant in self.ants:
            ants[ant.team].append((ant.x, ant.y, ant.hasFood))
        pickle.dump(ants, self.logfile)

        pickle.dump(self.foodCount, self.logfile)

    def hasTerminated(self):
        return self.terminated
    
    def terminate(self):
        self.terminated = True

    def tick(self):
        if self.terminated:
            return
        
         # Update ants
        for ant in self.ants:
            ant.energy = 1 # replenish energy
            ant.act()        
            
        # Update pheromones
        self.updatePheromones()

        # Update food counts
        self.updateFoodCounts()

        # Write current state to logfile
        self.updateLogFile()

        self.time += 1

    def shutdown(self):
        if self.logfile:
            self.logfile.close()