from simulation import Simulation
from BaseClassAnt import Ant
import random

class RandomAnt(Ant):
    def __init__(self, simulation, x, y, team, id):
        super().__init__(simulation, x, y, team, id)

    def act(self):
        if self.hasFood and self.simulation.nestGrid[self.team][self.x][self.y] > 0:
            action = Ant.A_DROP
        elif self.simulation.foodGrid[self.x][self.y] > 0 and not self.hasFood:
            action = Ant.A_TAKE
        else:
            action = random.choice([Ant.A_LEFT, Ant.A_RIGHT, Ant.A_UP, Ant.A_DOWN])
        return super().act(action)
