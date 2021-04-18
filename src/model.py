from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace
from mesa.time import SimultaneousActivation

from structures import EpiAgent, District
from utils import get_isol_boxes, Condition, compute_infected, compute_not_infected, compute_dead, compute_healed, \
    get_healthcare_potential


class EpiModel(Model):
    def __init__(self, population_number, width, height, districts, n_boxes=3, **kwargs):
        super().__init__()
        self.districts = []
        for district in districts:
            self.districts.append(District(**district))
        self.num_agents = population_number
        self.space = ContinuousSpace(width, height, False)
        self.schedule = SimultaneousActivation(self)

        self.travel_dist_factor = kwargs.get('travel_dist_factor', 1)
        self.transmission_prob = kwargs.get('transmission_prob', 0.4)
        self.mortality_rate = kwargs.get('mortality_rate', 0.004)
        self.inf_radius = kwargs.get('inf_radius', 0.5)
        self.healing_period = kwargs.get('healing_period', 7)
        self.travel_prob = kwargs.get('travel_prob', 0)
        self.healthcare_potential = kwargs.get('healthcare_potential', 0)
        self.travel_to_point_prob = kwargs.get('travel_to_point_prob', 0)
        self.quaranteen_after = kwargs.get('quaranteen_after', 0)
        self.quaranteen_stricktness = kwargs.get('quaranteen_stricktness', 1)
        self.isol_boxes = get_isol_boxes(n_boxes, width, height)

        for i in range(self.num_agents):
            a = EpiAgent(i, self)
            self.schedule.add(a)

            onbox = True
            while onbox:
                onbox = False
                x = self.random.randrange(self.space.width * 1000) / 1000
                y = self.random.randrange(self.space.height * 1000) / 1000
                for b in self.isol_boxes:
                    if b[0] - self.inf_radius < x < b[0] + self.inf_radius or \
                            b[1] - self.inf_radius < y < b[1] + self.inf_radius:
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
        hp = self.healthcare_potential * len(self.schedule.agents)
        if self.healthcare_potential and infected > hp:
            # self.healthcare_potential *= 0.97
            # self.mortality_rate *= 1.035
            self.healthcare_potential *= 1 - (infected - hp) / len(self.schedule.agents) * 0.15
            self.mortality_rate *= 1 + (infected - hp) / len(self.schedule.agents) * 0.15


