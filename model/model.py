import enum

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector


def compute_infected(model):
    return sum(agent.condition == Condition.Infected for agent in model.schedule.agents)


def compute_dead(model):
    return sum(agent.condition == Condition.Dead for agent in model.schedule.agents)


def compute_healed(model):
    return sum(agent.condition == Condition.Healed for agent in model.schedule.agents)


def compute_not_infected(model):
    return sum(agent.condition == Condition.Not_infected for agent in model.schedule.agents)


def get_isol_boxes(n, width, height):
    return [(i*width/n, i*height/n) for i in range(1, n)]


def get_healthcare_potential(model):
    return model.healthcare_potential * len(model.schedule.agents)
    

class Condition(enum.Enum):
    Not_infected = 0
    Infected = 1
    Dead = 2
    Healed = 3


class EpiAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.condition = Condition.Not_infected
        self.prev_pos = None
        self.days_infected = 0

    def step(self):

        if self.condition == Condition.Dead:
            return

        if self.days_infected > self.model.healing_period:
            self.condition = Condition.Healed

        p = self.random.random()
        if self.condition == Condition.Infected:
            self.days_infected += 1
            if p < self.model.mortality_rate:
                self.condition = Condition.Dead
                self.model.space.move_agent(
                    self, (self.pos[0], self.model.space.height - 0.1))
        self.move()

    def advance(self):
        self.infect()

    def infect(self):
        if not self.condition == Condition.Infected:
            return

        neighbors = self.model.space.get_neighbors(self.pos, include_center=False, radius=self.model.inf_radius)

        if len(neighbors) > 1:
            for n in neighbors:
                r = self.random.random()
                if r < self.model.transmission_prob and n.condition == Condition.Not_infected:
                    n.condition = Condition.Infected

    def move(self):
        if self.prev_pos:
            self.pos = self.prev_pos
            self.prev_pos = None
            return
        x, y = self.pos
        xd = self.random.uniform(-0.1, 0.1) * self.model.travel_dist_factor
        yd = self.random.uniform(-0.1, 0.1) * self.model.travel_dist_factor
        if self.random.random() < self.model.travel_to_point_prob and not self.prev_pos:
            self.prev_pos = self.pos
            x, y = self.model.space.width / 2, self.model.space.height / 2

        nx, ny = (min(self.model.space.width-0.1, max(0, x+xd)),
                  (min(self.model.space.height-0.1, max(0, y+yd))))

        for b in self.model.isol_boxes:
            if b[0]-self.model.inf_radius < nx < b[0]+self.model.inf_radius:
                if self.random.random() < self.model.travel_prob:
                    nx += xd / abs(xd) * 2 * self.model.inf_radius
                else:
                    nx = x

            if b[1]-self.model.inf_radius < ny < b[1]+self.model.inf_radius:
                if self.random.random() < self.model.travel_prob:
                    ny += yd / abs(yd) * 2 * self.model.inf_radius
                else:
                    ny = y

        self.model.space.move_agent(self, (nx, ny))


class EpiModel(Model):
    def __init__(self, population_number, width, height, n_boxes=3, **kwargs):
        super().__init__()
        self.num_agents = population_number
        self.space = ContinuousSpace(width, height, False)
        self.schedule = SimultaneousActivation(self)

        self.travel_dist_factor = kwargs.get('travel_dist_factor', 1)
        self.transmission_prob = kwargs.get('transmission_prob', 0.4)
        self.mortality_rate = kwargs.get('mortality_rate', 0.004)
        self.inf_radius = kwargs.get('inf_radius', 0.5)
        self.healing_period = kwargs.get('healing_period', 7)
        self.travel_prob = kwargs.get('travel_prob', 0)
        self.healthcare_potential = kwargs.get('healthcare_potential', None)
        self.travel_to_point_prob = kwargs.get('travel_to_point_prob', 0)
        self.isol_boxes = get_isol_boxes(n_boxes, width, height)

        for i in range(self.num_agents):
            a = EpiAgent(i, self)
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

        self.random.choice(self.schedule.agents).condition = Condition.Infected

        self.datacollector = DataCollector(model_reporters={
                                           "Infected": compute_infected,
                                           "Not_infected": compute_not_infected,
                                           "Dead": compute_dead,
                                           "Healed": compute_healed,
                                           "Healthcare_potential": get_healthcare_potential})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

        infected = compute_infected(self)
        hp = self.healthcare_potential*len(self.schedule.agents)
        if self.healthcare_potential and infected > hp:
            self.healthcare_potential *= 0.97
            self.mortality_rate *= 1.035
            # self.healthcare_potential *= 1-(infected-hp)/len(self.schedule.agents)
            # self.mortality_rate *= 1+(infected-hp)/len(self.schedule.agents)

