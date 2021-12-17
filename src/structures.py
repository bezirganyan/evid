import math
import random
from operator import itemgetter
from typing import Union

from shapely.geometry import Point, Polygon
from mesa import Agent
import numpy as np

from src.utils import Condition, compute_inf_prob, Severity


def rand_bin_array(k: int, n: int) -> np.ndarray:
    """
    Generate a random binary array with the given proportions: Works faster than np.random.choice or random.choices
    :param k: number of 1s
    :type k: int
    :param n: number of elements
    :type n: int
    :return: Binary array
    :rtype: numpy.ndarray
    """
    arr = np.zeros(n).astype(int)
    arr[:k] = 1
    np.random.shuffle(arr)
    return arr


class District:
    """
    Container for buildings in the district
    """
    def __init__(self, uid: int, name: str, population: int):
        """
        :param uid: unique ID for the district
        :type uid: int
        :param name: Name of the district
        :type name: str
        :param population: number of people living in the district
        :type population: int
        """
        self.id = uid
        self.name = name
        self.buildings = []
        self.population = population


class Building:
    def __init__(self, index: int, coordinates: Union[Point, Polygon], district: int,
                 building_type: str, n_apartments: int = None):
        """
        :param index: A Unique ID for the building, same as the OSMID of the building in the OpenStreetMaps
        :type index: int
        :param coordinates: The coordinates of the building from OpenStreetMaps
        :type coordinates: shaply.geometry.Point, shaply.geometry.Polygon
        :param district: The district ID
        :type district: int
        :param building_type: The type of the building
        :type building_type: str
        :param n_apartments: Number of apartments in the building [Optional, required for residential buildings]
        :type n_apartments: int
        """
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

    def place_agent(self, agent: Agent) -> None:
        """
        Place the agent inside the building. If the building is hospital, and the agent is places as a patient,
        then the agent will only be placed if there are enough ICU beds available: ie `hospital_beds > 0`.
        If the building is residential, the agent will be placed in their apartment, otherwise they will be placed
        in the apartment 0. Non-residential buildings have only one apartment: 0

        :param agent: The agent which will be placed
        :type agent: Agent
        :return: None
        :rtype: None
        """
        if self.building_type == 'hospital' and \
                agent.condition == Condition.Infected and \
                agent.severity == Severity.severe and \
                agent.countdown_after_infected <= 0:
            if agent.model.hospital_beds <= 0:
                agent.model.grid.place_agent(agent, agent.address[0])
                return
            else:
                agent.model.hospital_beds -= 1
                agent.in_hospital = True
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

    def remove_agent(self, agent: Agent) -> None:
        """
        Remove the agent from the building.
        :param agent: The agent which will be removed
        :type agent: Agent
        :return: None
        :rtype: None
        """
        if self.building_type == 'hospital' and agent.condition == Condition.Healed and agent.in_hospital:
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
    def __init__(self, unique_id: int, model, age: int, gender: int, work_place: tuple, study_place: tuple,
                 negative_sample_proportion: float = 1.0):
        """
        :param unique_id: a unique ID for the aent
        :type unique_id: int
        :param model: an instance of the EpiModel class
        :type model: EpiModel
        :param age: the index of the age group of the agent [(0-4),(5-19),(20-29),(30-63),(64-120)]
        :type age: int
        :param gender: The gender of the agent
        :type gender: int
        :param work_place: the workplace of the agent, a tuple where the first element is the facility type,
        the second: facility id
        :type work_place: tuple[str, int]
        :param study_place: the study place of the agent, a tuple where the first element is the facility type,
        the second: facility id
        :type study_place: tuple[str, int]
        :param negative_sample_proportion: What percentage of the negative samples to write to logs [Default: 1]
        :type negative_sample_proportion: float
        """
        super().__init__(unique_id, model)
        self.negative_sample_proportion = negative_sample_proportion
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

    def step(self) -> None:
        """
        Performs a simulation step for the agent. No need to call manually, will be called by the scheduler
        :return: None
        :rtype: None
        """
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

    def advance(self) -> None:
        self.infect()

    def infect(self) -> None:
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
        choices = rand_bin_array(n_contact_people, len(same_place_agents))
        # contacted_agents = [same_place_agents[c] for c in np.nonzero(choices)]
        contacted_agents = itemgetter(*np.nonzero(choices)[0])(same_place_agents)
        if contacted_agents is None:
            return
        contacted_agents = contacted_agents if isinstance(contacted_agents, tuple) else (contacted_agents,)
        if self.condition == Condition.Infected and self.countdown_after_infected <= 0:
            fc = self.model.facility_conf[building_type]
            vc = self.model.virus_conf
            inf_prob = compute_inf_prob(rlwr=fc['rlwr'], area=fc['area'], height=fc['height'],
                                        speak_frac=fc['speak_frac'], mask_in=fc['mask_in'], mask_out=fc['mask_out'],
                                        vol=fc['vol'], lifetime=vc['lifetime'], d50=vc['d50'], conc=vc['conc'],
                                        mwd=vc['mwd'], conc_b=vc['conc_b'], conc_s=vc['conc_s'], depo=vc['depo'],
                                        atv=self.model.people_conf['atv'][self.age])
            # infected_candidates = np.random.choice([0, 1], p=(1 - inf_prob, inf_prob), size=len(contacted_agents))
            infected_candidates = rand_bin_array(int(inf_prob * len(contacted_agents)), len(contacted_agents))
            # infected_candidates = np.nonzero((np.random.random(len(contacted_agents)) < inf_prob).astype(int))
            for agent, inf in zip(contacted_agents, infected_candidates):
                if self.unique_id == agent.unique_id: continue
                self.daily_contacts.append(",".join(list(map(str, [self.unique_id, agent.unique_id,
                                                                   self.model.day, self.model.weekday,
                                                                   self.model.hour, building.index,
                                                                   inf]))))
                if inf and agent.condition == Condition.Not_infected:
                    agent.set_infected()
        else:
            for agent in contacted_agents[:int(len(contacted_agents) * self.negative_sample_proportion)]:
                if self.unique_id == agent.unique_id or (agent.unique_id in self.encountered_agents): continue
                agent.encountered_agents.add(self.unique_id)
                self.daily_contacts.append(",".join(list(map(str, [self.unique_id, agent.unique_id,
                                                                   self.model.day, self.model.weekday,
                                                                   self.model.hour, building.index,
                                                                   0]))))
            self.encountered_agents.clear()

    def move(self) -> None:
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
        building_types_dist = self.model.moving_distribution_tensor[self.model.weekday, self.age, self.model.hour]
        to_node_type = random.choices(list(self.model.facility_conf.keys()), weights=building_types_dist)[0]
        if to_node_type == 'residential':
            to_node = self.address[0]
            assert to_node is not None
        elif to_node_type in ['school', 'kindergarten', 'university']:
            to_node = self.study_place[1] if self.study_place is not None and self.study_place[
                1] else self.get_target_node_healthy()
            assert to_node is not None
        elif to_node_type == 'work':
            to_node = self.work_place[1] if self.work_place is not None else self.get_target_node_healthy()
            assert to_node is not None
        else:
            to_node = random.choice(self.model.osmid_by_building_type[to_node_type])
            assert to_node is not None
        return to_node

    def set_infected(self) -> None:
        """
        Set the agent as infected, assign severity and countdown for incubation period
        :return: None
        :rtype: None
        """
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
