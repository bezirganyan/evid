import math
import random
import json

from mesa import Agent
import numpy as np
from numpy.random import choice

from configs.contact_prob import contact_prob
from src.utils import Condition, compute_inf_prob, Severity


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

    def place_agent(self, agent):
        if self.building_type == 'hospital':
            if agent.model.hospital_beds <= 0:
                agent.model.grid.place_agent(agent, agent.address[0])
                return
            else:
                agent.model.hospital_beds -= 1
                agent.in_hosptal = True
        if not self.public:
            if not agent.address:
                apartment = random.randint(1, self.n_apartments)
                if apartment not in self.apartments:
                    self.apartments[apartment] = []
                agent.address = (self.index, apartment)
            else:
                apartment = agent.address[1]
            self.apartments[apartment].append(agent)
        else:
            self.apartments[0].append(agent)
        agent.pos = self.index

    def remove_agent(self, agent):
        if self.building_type == 'hospital':
            agent.model.hospital_beds += 1
            agent.in_hospital = False
        if not self.public:
            self.apartments[agent.address[1]].remove(agent)
        else:
            assert agent.pos == self.index
            self.apartments[0].remove(agent)

        agent.pos = None

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
    def __init__(self, unique_id: int, model, age: int, gender: str, work_place: tuple, study_place: tuple):
        super().__init__(unique_id, model)
        self.condition = Condition.Not_infected
        self.prev_pos = None
        self.hours_infected = 0
        self.gender = gender
        self.age = age
        self.address = None
        self.in_hospital = False
        self.severity: Severity = None
        self.countdown_after_infected: int = None
        self.work_place = work_place
        self.study_place = study_place
        self.works = 0;
        self.studies = 0;

    def step(self):
        if self.hours_infected > self.model.healing_period:
            self.condition = Condition.Healed

        p = random.random()
        if self.condition == Condition.Infected:
            self.hours_infected += self.model.step_size
            if p < self.model.mortality_rate[
                self.severity.name]:  # TODO should be changed ( and self.countdown_after_infected <= 0)
                self.model.dead_count += 1
                self.model.grid._remove_agent(self, self.pos)
                self.condition = Condition.Dead
            return
        self.move()
        if self.countdown_after_infected is not None and self.countdown_after_infected > 0:
            self.countdown_after_infected -= self.model.step_size

    def advance(self):
        self.infect()

    def infect(self):
        if self.condition == Condition.Dead:
            self.model.scheduler.remove(self)
        if self.condition == Condition.Infected:
            building_type = self.model.graph.nodes[self.pos]['building'].building_type
            if self.model.graph.nodes[self.pos]['building'].public:
                same_place_agents = self.model.graph.nodes[self.pos]['building'].apartments[0]
                n_contact_people = contact_prob[building_type] * len(same_place_agents)
                n_contact_people = math.ceil(math.ceil(n_contact_people))
                contacted_agents = choice(same_place_agents, size=n_contact_people)
            else:
                same_place_agents = self.model.graph.nodes[self.pos]['building'].apartments[
                    self.address[1]]
                contacted_agents = same_place_agents
            with open('configs/compute-inf-prob.json', 'r') as compute_inf_prob_json:
                arguments = json.load(compute_inf_prob_json)["compute_inf_prob_arguments"][
                    building_type]
                inf_prob = compute_inf_prob(**arguments)
            infected_candidates = random.choices([0, 1], weights=(1 - inf_prob, inf_prob), k=len(contacted_agents))
            for agent, inf in zip(contacted_agents, infected_candidates):
                if inf and agent.condition == Condition.Not_infected:
                    agent.set_infected()

    def move(self):
        if self.condition == Condition.Dead:
            return
        if self.in_hospital and self.condition == Condition.Infected:
            return
        if self.condition == Condition.Infected:
            if self.severity == Severity.mild:
                if self.countdown_after_infected == 0:
                    to_node = self.address[0]
                else:
                    to_node = self.get_target_node_healthy()
            elif self.severity == Severity.severe:
                if self.countdown_after_infected == 0:
                    to_node = random.choice(
                        self.model.get_b_ids_by_types(['hospital']))  # if there is no place in hospital agent goes home
                else:
                    to_node = self.get_target_node_healthy()
            else:
                to_node = self.get_target_node_healthy()
        else:
            to_node = self.get_target_node_healthy()
        self.model.grid.move_agent(self, to_node)

    def get_target_node_healthy(self):
        if self.study_place is not None and self.pos == self.study_place[1]:
            if self.studies < self.study_place[0]:
                self.studies += self.model.step_size
                return self.pos
            else:
                self.studies = 0
        if self.work_place is not None and self.pos == self.work_place[1]:
            if self.works < self.work_place[0]:
                self.works += self.model.step_size
                return self.pos
            else:
                self.works = 0
        week_day = self.model.date.weekday()
        time = int(self.model.date.strftime("%H"))
        building_types_dist = self.model.moving_distribution_tensor[week_day, self.age, time]
        to_node_type = random.choices(range(len(self.model.building_types)), weights=building_types_dist)[
            0]
        if to_node_type == len(building_types_dist) - 1:
            to_node = self.address[0]
        elif to_node_type == 3:
            to_node = self.study_place[1] if self.study_place is not None else self.get_target_node_healthy()
        elif to_node_type == 4:
            to_node = self.study_place[1] if self.study_place is not None else self.get_target_node_healthy()
        elif to_node_type == 7:
            to_node = self.study_place[1] if self.study_place is not None else self.get_target_node_healthy()
        elif to_node_type == 8:
            to_node = self.work_place[1] if self.work_place is not None else self.get_target_node_healthy()
        else:
            to_node = random.choice(self.model.osmid_by_building_type[to_node_type]['osmids'])
        return to_node

    def set_infected(self):
        self.condition = Condition.Infected
        self.severity = \
            random.choices(list(Severity), weights=[self.model.severity_dist[s.name] for s in list(Severity)])[0]
        self.countdown_after_infected = np.random.normal(loc=self.model.infection_countdown_dist['loc'],
                                                         scale=self.model.infection_countdown_dist['scale'],
                                                         size=1)

    def __repr__(self):
        return f"Agent\n" \
               f"unique_id: {self.unique_id},\n" \
               f"prev_pos: {self.prev_pos},\n" \
               f"condition: {self.condition},\n" \
               f"days_infected: {self.hours_infected},\n" \
               f"address: {self.address},\n" \
               f"pos: {self.pos}\n" \
               f"gender: {self.gender},\n" \
               f"age: {self.age}\n" \
               f"work_place: {self.work_place},\n" \
               f"study_place: {self.study_place}\n"
