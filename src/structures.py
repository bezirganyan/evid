import math
import random
import json

from mesa import Agent
import numpy as np
from numpy.random import choice

from src.utils import Condition, compute_inf_prob, Severity


class District:
    def __init__(self, uid: int, name: str, population: int):
        self.id = uid
        self.name = name
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
        self.public = False if building_type == 'residential' else True
        if self.public:
            self.apartments[0] = []
            self.n_apartments = 1

    def place_agent(self, agent):
        if self.building_type == 'hospital' and agent.severity == Severity.severe:
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
        if self.building_type == 'hospital' and agent.condition == Condition.Healed:
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
    def __init__(self, unique_id: int, model, age: int, gender: int, work_place: tuple, study_place: tuple):
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
        self.works = 0
        self.studies = 0
        self.encountered_agents = set()
        self.daily_contacts = []

    def step(self):
        if self.hours_infected > self.model.healing_period:
            self.condition = Condition.Healed

        p = random.random()
        if self.condition == Condition.Infected:
            mortality_rate = self.model.mortality_rate[
                self.severity.name] if not self.in_hospital else self.model.mortality_rate[
                                                                     self.severity.name] * self.model.hospital_efficiency  # TODO check difference of mortality rate in hospital
            self.hours_infected += self.model.step_size
            if p < mortality_rate and self.countdown_after_infected <= 0:
                self.model.dead_count += 1
                self.model.grid._remove_agent(self, self.pos)
                self.condition = Condition.Dead
        self.move()
        if self.countdown_after_infected is not None and self.countdown_after_infected > 0:
            self.countdown_after_infected -= self.model.step_size

    def advance(self):
        self.infect()

    def infect(self):
        if self.condition == Condition.Dead:
            self.model.scheduler.remove(self)
            return
        building = self.model.graph.nodes[self.pos]['building']
        building_type = building.building_type
        btm = self.model.building_type_mapping[building_type]
        ap = self.address[1] if not self.model.graph.nodes[self.pos]['building'].public else 0
        same_place_agents = self.model.graph.nodes[self.pos]['building'].apartments[ap]
        n_contact_people = self.model.facility_conf[building_type]['contact_probability'] * len(
            same_place_agents)  # TODO do we need number of contact people
        n_contact_people = math.ceil(math.ceil(n_contact_people))
        contacted_agents = choice(same_place_agents, size=n_contact_people)
        if self.condition == Condition.Infected:
            fc = self.model.facility_conf[building_type]
            vc = self.model.virus_conf
            inf_prob = compute_inf_prob(rlwr=fc['rlwr'], area=fc['area'], height=fc['height'],
                                        speak_frac=fc['speak_frac'], mask_in=fc['mask_in'], mask_out=fc['mask_out'],
                                        vol=fc['vol'], lifetime=vc['lifetime'], d50=vc['d50'], conc=vc['conc'],
                                        mwd=vc['mwd'], conc_b=vc['conc_b'], conc_s=vc['conc_s'], depo=vc['depo'],
                                        atv=self.model.people_conf['atv'][self.age])
            infected_candidates = np.random.choice([0, 1], p=(1 - inf_prob, inf_prob), size=len(contacted_agents))
            for agent, inf in zip(contacted_agents, infected_candidates):
                if self.unique_id == agent.unique_id: continue
                self.daily_contacts.append(",".join(list(map(str, [self.unique_id, agent.unique_id,
                                                                   self.model.date.strftime('%d-%H'), self.age,
                                                                   agent.age, btm, building.district, building.index,
                                                                   inf]))))
                if inf and agent.condition == Condition.Not_infected:
                    agent.set_infected()
        else:
            for agent in contacted_agents:
                if self.unique_id == agent.unique_id or (agent.unique_id in self.encountered_agents): continue
                agent.encountered_agents.add(self.unique_id)
                self.daily_contacts.append(",".join(list(map(str, [self.unique_id, agent.unique_id,
                                                                   self.model.date.strftime('%d-%H'), self.age,
                                                                   agent.age, btm, building.district, building.index,
                                                                   0]))))
            self.encountered_agents.clear()

    def move(self):
        if self.condition == Condition.Dead:
            return
        if self.in_hospital and self.condition == Condition.Infected:
            return
        if self.condition == Condition.Infected:
            if self.severity == Severity.mild:
                if self.countdown_after_infected <= 0:
                    to_node = self.address[0]
                    assert to_node is not None
                else:
                    to_node = self.get_target_node_healthy()
                    assert to_node is not None
            elif self.severity == Severity.severe:
                if self.countdown_after_infected <= 0:
                    to_node = random.choice(
                        self.model.get_b_ids_by_types(['hospital']))  # if there is no place in hospital agent goes home
                    assert to_node is not None
                else:
                    to_node = self.get_target_node_healthy()
                    assert to_node is not None
            else:
                to_node = self.get_target_node_healthy()
                assert to_node is not None
        else:
            to_node = self.get_target_node_healthy()
            assert to_node is not None
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
        to_node_type = random.choices(list(self.model.facility_conf.keys()), weights=building_types_dist)[
            0]
        if to_node_type == 'residential':
            to_node = self.address[0]
            assert to_node is not None
        elif to_node_type in ['school', 'kindergarten', 'university']:
            to_node = self.study_place[1] if self.study_place is not None else self.get_target_node_healthy()
            assert to_node is not None
        elif to_node_type == 'work':
            to_node = self.work_place[1] if self.work_place is not None else self.get_target_node_healthy()
            assert to_node is not None
        else:
            to_node = random.choice(self.model.osmid_by_building_type[to_node_type])
            assert to_node is not None
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
