
class Ant:
    # Actions
    A_LEFT      = 0
    A_RIGHT     = 1
    A_UP        = 2
    A_DOWN      = 3
    A_TAKE      = 4
    A_DROP      = 5
    A_IDLE      = 6
    A_PHEROMONE = 7

    ACTIONS = [A_LEFT, A_RIGHT, A_UP, A_DOWN, A_TAKE, A_DROP]

    D_LEFT = 0
    D_RIGHT = 1
    D_UP = 2
    D_DOWN = 3
    D_CENTER = 4

    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # Left, Right, Up, Down, Center

    def __init__(self, simulation, x, y, team, id):
        self.x = x
        self.y = y
        self.team = team
        self.id = id
        self.simulation = simulation
    
        self.hasFood = False
        
        self.direction = (0, 0)
        self.energy = 1

        self.simulation.antGrid[self.team][self.x][self.y] += 1

    def act(self, action):
        if action in [Ant.A_LEFT, Ant.A_RIGHT, Ant.A_UP, Ant.A_DOWN]:
            self.direction = Ant.DIRECTIONS[action]
            self.move()
        elif action == Ant.A_TAKE:
            self.takeFood()
        elif action == Ant.A_DROP:
            self.dropFood()

        return self.simulation.rewardSignal(self)

    def move(self):
        if self.energy < 1:
            return
        self.energy -= 1

        if not self.direction in Ant.DIRECTIONS:
            return
        
        dx, dy = self.direction

        nx = self.x + dx
        ny = self.y + dy

        if ((nx >= 0) and (ny >= 0) and (nx < self.simulation.gridWidth) and (ny < self.simulation.gridHeight) and (self.simulation.obstacleGrid[nx][ny] == 0)):
            self.simulation.antGrid[self.team][self.x][self.y] -= 1
            self.simulation.antGrid[self.team][nx][ny] += 1
            self.x = nx
            self.y = ny

    def takeFood(self):
        if self.energy < 1:
            return
        self.energy -= 1
        
        if (self.simulation.foodGrid[self.x][self.y] > 0 and not self.hasFood):  
            self.hasFood = True
            self.simulation.foodGrid[self.x][self.y] -= 1  

            return True
        return False
    
    def dropFood(self):
        if self.energy < 1:
            return
        self.energy -= 1
        
        if (self.hasFood):
            self.simulation.foodGrid[self.x][self.y] += 1
            self.hasFood = False

    def dropPheromone(self):
        self.simulation.pheromoneGrid[self.team][self.x][self.y] = 1.0

    def senseFood(self):
        return [self.simulation.foodGrid[self.x + dx][self.y + dy] for (dx,dy) in self.DIRECTIONS]
    
    def senseObstacles(self):
        return [self.simulation.obstacleGrid[self.x + dx][self.y + dy] for (dx,dy) in self.DIRECTIONS]
    
    def senseOwnNest(self):
        return [self.simulation.nestGrid[self.team][self.x + dx][self.y + dy] for (dx,dy) in self.DIRECTIONS]
    
    def senseOtherNest(self):
        return [self.simulation.nestGrid[1-self.team][self.x + dx][self.y + dy] for (dx,dy) in self.DIRECTIONS]
    
    def senseOwnPheromone(self):
        return [self.simulation.pheromoneGrid[self.team][self.x + dx][self.y + dy] for (dx,dy) in self.DIRECTIONS]
    
    def senseOtherPheromone(self):
        return [self.simulation.pheromoneGrid[1 - self.team][self.x + dx][self.y + dy] for (dx,dy) in self.DIRECTIONS]
    
    def senseOwnAnts(self):
        return [self.simulation.antGrid[self.team][self.x + dx][self.y + dy] for (dx,dy) in self.DIRECTIONS]
    
    def senseOtherAnts(self):
        return [self.simulation.antGrid[1-self.team][self.x + dx][self.y + dy] for (dx,dy) in self.DIRECTIONS]

