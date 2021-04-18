from mesa import Agent

from src.utils import Condition


class District:
    def __init__(self, uid: int, name: str, n_buildings: int, population: int):
        self.id = uid
        self.name = name
        self.building_amount = n_buildings
        self.buildings = []
        self.population = population


class Building:
    def __init__(self, index, coordinates, district, building_type, n_apartments=None):
        self.index = index
        self.coordinates = coordinates
        self.n_apartments = n_apartments
        self.district = district
        self.apartments = dict()
        self.building_type = building_type
        self.public = False if building_type == 'building' else True
        if self.public:
            self.apartments[0] = []
            self.n_apartments = 1

    def __repr__(self):
        return f"BUILDING\n" \
               f"index: {self.index},\n" \
               f"coordinates: {self.coordinates},\n" \
               f"building_type: {self.building_type},\n" \
               f"district: {self.district},\n" \
               f"n_apartments: {self.n_apartments},\n" \
               f"apartments: {self.apartments}\n" \
               f"public: {self.public}"


class EpiAgent(Agent):
    def __init__(self, unique_id: int, model, age: int, gender: str):
        super().__init__(unique_id, model)
        self.condition = Condition.Not_infected
        self.prev_pos = None
        self.days_infected = 0
        self.gender = gender
        self.age = age
        self.address = None

    def step(self):
        # if 0 < self.model.quaranteen_after < self.days_infected:
        #     if self.random.random() < self.model.quaranteen_stricktness:
        #         self.condition = Condition.Quaranteened

        if self.days_infected > self.model.healing_period:  # TODO - 3 Add illness severity
            self.condition = Condition.Healed

        p = self.random.random()
        if self.condition == Condition.Infected or self.condition == Condition.Quaranteened:
            self.days_infected += 1
            if p < self.model.mortality_rate:
                self.model.dead_count += 1
                self.model.grid.remove(self)
            return
        self.move()

    def advance(self):
        self.infect()

    def infect(self):
        if not self.condition == Condition.Infected:
            return

        contact_candidates = None  # TODO - implement: 2 get contact candidates
        if len(contact_candidates) > 0:
            for c in contact_candidates:
                r = self.random.random()
                if r < self.model.transmission_prob and c.condition == Condition.Not_infected:
                    c.condition = Condition.Infected

    def move(self):
        to_node = None  # TODO - 1 get to which node to move
        self.model.grid.move_agent(self, to_node)
