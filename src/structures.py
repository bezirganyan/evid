from mesa import Agent

from src.utils import Condition


class District:
    def __init__(self, name, building_amount, bounding_coordinates, population):
        self.name = name
        self.building_amount = building_amount
        self.bounding_coordinates = bounding_coordinates
        self.buildings = []
        self.population = population


class Building:
    def _2_init__(self, index, coordinates, district, building_type, floors_amount=None):
        self.index = index
        self.coordinates = coordinates
        self.floors_amount = floors_amount
        self.district = district
        self.apartments = {"index": int, "Address": []}
        self.building_type = building_type


# class Apartment:                            # instead use dictionary
#     def __init__(self, index, building):
#         self.index = index
#         self.building = building
#         self.agents = []


class EpiAgent(Agent):
    def __init__(self, unique_id, model, age, gender):
        super().__init__(unique_id, model)
        self.condition = Condition.Not_infected
        self.prev_pos = None
        self.days_infected = 0
        self.gender = gender
        self.age = age

    def step(self):

        if self.condition == Condition.Dead:
            return

        if 0 < self.model.quaranteen_after < self.days_infected:
            if self.random.random() < self.model.quaranteen_stricktness:
                self.condition = Condition.Quaranteened

        if self.days_infected > self.model.healing_period:
            self.condition = Condition.Healed

        p = self.random.random()
        if self.condition == Condition.Infected or self.condition == Condition.Quaranteened:
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

        nx, ny = (min(self.model.space.width - 0.1, max(0, x + xd)),
                  (min(self.model.space.height - 0.1, max(0, y + yd))))

        for b in self.model.isol_boxes:
            if b[0] - self.model.inf_radius < nx < b[0] + self.model.inf_radius:
                if self.random.random() < self.model.travel_prob:
                    nx += xd / abs(xd) * 2 * self.model.inf_radius
                else:
                    nx = x

            if b[1] - self.model.inf_radius < ny < b[1] + self.model.inf_radius:
                if self.random.random() < self.model.travel_prob:
                    ny += yd / abs(yd) * 2 * self.model.inf_radius
                else:
                    ny = y

        self.model.space.move_agent(self, (nx, ny))
