from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

def compute_infected(model):
    return sum(agent.infected for agent in model.schedule.agents)
    
def compute_dead(model):
    return sum(agent.dead for agent in model.schedule.agents)

def compute_healed(model):
    return sum(agent.healed for agent in model.schedule.agents)

def get_isol_boxes(n, width, height):
    boxes = [(i*width/n, i*height/n) for i in range(1, n)]

    return boxes

class CoronaAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.infected = False
        self.dead = False
        self.healed = False
        self.transmission_prob = 0.4
        self.mortality_rate = 0.004
        self.inf_radius = 0.3
        self.prev_pos = None
        self.days_infected = 0
        self.healing_period = 14
        self.travel_prob = 0.05
        self.healthcare_potential = 0.02

    def step(self):
        inf_perc = compute_infected(self.model) / len(self.model.schedule.agents)
        if inf_perc > self.healthcare_potential:
            self.healthcare_potential *= 0.999999999999
            self.mortality_rate *= 1.000000000001

        if self.dead:
            return
        if self.days_infected > self.healing_period:
            self.healed = True
            self.infected = False
        p = self.random.random()
        if self.infected:
            self.days_infected += 1
            if p < self.mortality_rate:
                self.dead = True
                self.infected = False
                self.model.space.move_agent(self, (self.pos[0], self.model.space.height - 0.1))
        self.move()
        self.infect()

    def infect(self):
        if not self.infected or self.healed or self.dead:
            return

        neighbors = self.model.space.get_neighbors(self.pos, include_center=False, radius=self.inf_radius)

        if len(neighbors) > 1:
            for n in neighbors:
                r = self.random.random()
                if r < self.transmission_prob and not n.dead and not n.healed:
                    n.infected = True


    def move(self):
        if self.prev_pos:
            self.pos = self.prev_pos
            self.prev_pos = None
            return
        x, y = self.pos
        xd = self.random.uniform(-0.1, 0.1)
        yd = self.random.uniform(-0.1, 0.1)
        if self.random.random() < 0.0001 and not self.prev_pos:
            self.prev_pos = self.pos
            x, y = self.model.space.width / 2, self.model.space.height / 2

        nx, ny = (min(self.model.space.width-0.1, max(0, x+xd)), (min(self.model.space.height-0.1, max(0, y+yd))))
        for b in self.model.isol_boxes:
            if b[0]-self.inf_radius < nx < b[0]+self.inf_radius:
                if self.random.random() < self.travel_prob:
                    nx += xd / abs(xd) * 2 * self.inf_radius
                else:
                    nx = x

            if b[1]-self.inf_radius < ny < b[1]+self.inf_radius:
                if self.random.random() < self.travel_prob:
                    ny += yd / abs(yd) * 2 * self.inf_radius
                else: 
                    ny = y

        self.model.space.move_agent(self, (nx, ny))


class CoronaModel(Model):
    def __init__(self, N, width, height, n_boxes=3):
        super().__init__()
        self.num_agents = N
        self.space = ContinuousSpace(width, height, False)
        self.schedule = RandomActivation(self)
        self.inf_radius = 0.3
        self.isol_boxes = get_isol_boxes(n_boxes, width, height)
        for i in range(self.num_agents):
            a = CoronaAgent(i, self)
            self.schedule.add(a)

            onbox = True
            while onbox: 
                onbox = False
                x = self.random.randrange(self.space.width*1000) / 1000
                y = self.random.randrange(self.space.height*1000) / 1000
                for b in self.isol_boxes:
                    if b[0]-self.inf_radius < x < b[0]+self.inf_radius or \
                        b[1]-self.inf_radius < y < b[1]+self.inf_radius:
                        onbox = True



            self.space.place_agent(a, (x, y))

        self.random.choice(self.schedule.agents).infected = True
        
        self.datacollector = DataCollector(model_reporters={"Infected": compute_infected, "Dead": compute_dead, "Healed": compute_healed})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
