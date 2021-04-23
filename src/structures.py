import random

from mesa import Agent

from src.utils import Condition
from configs.contact_prob import contact_prob
from src.move_distribution import fillTensor
import numpy as np
from numpy.random import choice
import math


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
        self.current_position = None

    def step(self):
        if 0 < self.model.quaranteen_after < self.days_infected:
            if self.random.random() < self.model.quaranteen_stricktness:
                self.condition = Condition.Quaranteened

        if self.days_infected > self.model.healing_period:  # TODO - 3 Add illness severity
            self.condition = Condition.Healed

        p = self.random.random()
        if self.condition == Condition.Infected or self.condition == Condition.Quaranteened:
            self.days_infected += 1
            if p < self.model.mortality_rate:
                self.model.dead_count += 1
                self.model.grid._remove_agent(self, self.current_position)
                self.condition = Condition.Dead
            return
        self.move()

    def advance(self):
        self.infect()

    def infect(self):
        if self.condition == Condition.Dead:
            self.model.scheduler.remove(self)
        if self.condition == Condition.Infected:
            if self.model.graph.nodes[self.current_position]['building'].public:
                same_place_agents = self.model.graph.nodes[self.current_position]['building'].apartments[0]
                building_type = self.model.graph.nodes[self.current_position]['building'].building_type
                n_contact_people = contact_prob[building_type] * len(same_place_agents)
                n_contact_people = math.ceil(math.ceil(n_contact_people))
                contacted_agents = choice(same_place_agents, size=n_contact_people)
            else:
                same_place_agents = self.model.graph.nodes[self.current_position]['building'].apartments[self.address[1]]
                contacted_agents = same_place_agents
            if not self.model.mask_rule:
                agent_wears_mask = random.choices([0, 1], weights=(1 - self.model.mask_prob[0][0], self.model.mask_prob[0][0]))[0]
                inf_prob = self.model.infection_prob[agent_wears_mask][0] * (1-self.model.mask_prob[0,0]) + self.model.infection_prob[agent_wears_mask][1] * self.model.mask_prob[0,0]
                for agent in contacted_agents:
                    if agent.condition == Condition.Not_infected:
                        agent.condition = random.choices([Condition.Infected, Condition.Not_infected], weights=(inf_prob, (1 - inf_prob)))[0]
                    else:
                        continue
            else:
                agent_wears_mask = random.choices([0, 1], weights=(1 - self.model.mask_prob[0][1], self.model.mask_prob[0][1]))[0]
                inf_prob = self.model.infection_prob[agent_wears_mask][0] * (1-self.model.mask_prob[0,1]) + self.model.infection_prob[agent_wears_mask][1] * self.model.mask_prob[0,1]
                for agent in contacted_agents:
                    if agent.condition == Condition.Not_infected:
                        agent.condition = random.choices([Condition.Infected, Condition.Not_infected], weights=(inf_prob, (1 - inf_prob)))[0]
                    else:
                        continue

        else:
            return

        # contact_candidates = None  #TODO - implement:  get contact candidates
        # if len(contact_candidates) > 0:
        #     for c in contact_candidates:
        #         r = self.random.random()
        #         if r < self.model.transmission_prob and c.condition == Condition.Not_infected:
        #             c.condition = Condition.Infected

    def move(self):
        week_day = self.model.date.weekday()
        time = int(self.model.date.strftime("%H"))
        building_types_dist = self.model.moving_distribution_tensor[week_day, self.age, time]
        to_node_type = random.choices(range(len(self.model.building_types)), weights=building_types_dist)[
            0]  # TODO -hishacnel Grigin
        if to_node_type == len(building_types_dist) - 1:
            to_node = self.address[0]
        else:
            to_node = random.choice(self.model.osmid_by_building_type[to_node_type]['osmids'])
        self.model.grid.move_agent(self, to_node)
        self.current_position = to_node
        # print(to_node)
        # # if to_node is None or to_node == '' or to_node == ' ':
        # #     print(self)
        # #     print(self.address[0])
        # self.infect(self)
        # print('works well')

    def __repr__(self):
        return f"Agent\n" \
               f"unique_id: {self.unique_id},\n" \
               f"prev_pos: {self.prev_pos},\n" \
               f"days_infected: {self.days_infected},\n" \
               f"address: {self.address},\n" \
               f"current_position: {self.current_position}\n" \
               f"gender: {self.gender},\n" \
               f"age: {self.age}\n"
