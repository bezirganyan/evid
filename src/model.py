import pickle
import random
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import SimultaneousActivation
from shapely import wkt
from tqdm import tqdm

from src.spaces import EpiNetworkGrid
from src.structures import EpiAgent, District, Building
from src.utils import Condition, compute_infected, compute_not_infected, compute_dead, compute_healed, \
    get_healthcare_potential, get_apartments_number, Logger


class EpiModel(Model):
    def __init__(self, population_number,
                 districts,
                 moving_distribution_tensor,
                 facility_conf: dict,
                 virus_conf: dict,
                 people_conf: dict,
                 age_dist: Tuple[float] = None,
                 city_data: pd.DataFrame = None,
                 graph: nx.Graph = None,
                 initial_infected: int = 10,
                 step_size: int = 1,
                 hospital_efficiency: float = 0.3,
                 save_every: int = 24,
                 severity_dist: dict = {"asymptomatic": 0.24, "mild": 0.56, "severe": 0.2},
                 infection_countdown_dist: dict = {"loc": 48, "scale": 7},
                 log_path='output.csv',
                 **kwargs):
        super().__init__()
        self.people_conf = people_conf
        self.virus_conf = virus_conf
        self.save_every = save_every
        self.day = 0
        self.hour = 0
        self.weekday = 0
        self.facility_conf = facility_conf
        self.districts = dict()
        self.dead_count = 0
        for d_id in districts:
            self.districts[d_id] = District(uid=d_id, **districts[d_id])
        self.graph = graph if graph else self.create_graph(city_data, facility_conf['residential']['floor_probs'])
        self.num_agents = population_number
        self.grid = EpiNetworkGrid(self.graph)
        self.scheduler = SimultaneousActivation(self)
        self.mortality_rate = kwargs.get('mortality_rate',
                                         {"asymptomatic": 0.0000001, "mild": 0.0002,
                                          "severe": 0.0004})  # TODO numbers should be changed
        self.healing_period = kwargs.get('healing_period', 8) * 24
        self.hospital_beds = kwargs.get('hospital_beds', 5000)
        self.checkpoint_directory = kwargs.get('checkpoint_directory', 'checkpoints')
        self.osmid_by_building_type = dict()
        self.moving_distribution_tensor = moving_distribution_tensor
        self.severity_dist = severity_dist
        self.infection_countdown_dist = infection_countdown_dist
        self.step_size = step_size
        self.distribute_osmid_by_building_type(city_data, list(facility_conf.keys()))
        self.hospital_efficiency = hospital_efficiency
        self.step_counts = 0
        self.logger = Logger(log_path, self)

        self.building_type_mapping = {t: i for i, t in enumerate(list(facility_conf.keys()))}
        for d in self.districts.values():
            self.distribute_people(d, age_dist)

        for a in self.random.choices(self.scheduler.agents, k=initial_infected):
            a.set_infected()

        self.datacollector = DataCollector(model_reporters={
            "Infected": compute_infected,
            "Not_infected": compute_not_infected,
            "Dead": compute_dead,
            "Healed": compute_healed,
            "Healthcare_potential": get_healthcare_potential})

    def do_time_step(self, step: int = 1) -> None:
        self.hour += step
        if self.hour >= 24:
            self.hour %= 24
            self.day += 1
            self.weekday = (self.weekday + 1) % 7

    def step(self):
        self.datacollector.collect(self)
        self.scheduler.step()
        self.do_time_step(self.step_size)
        self.step_counts += 1
        if self.step_counts % self.save_every == 0:
            self.save_model(self.checkpoint_directory)
        self.logger.write_log()

    def distribute_osmid_by_building_type(self, data_frame, building_types):
        groups_df = data_frame[data_frame['type'] != 'building']
        groups_df = groups_df.groupby('type')['osmid'].apply(list).reset_index(name='osmid')
        for ind, t in enumerate(building_types):
            if t == 'residential':
                continue
            self.osmid_by_building_type[t] = groups_df[groups_df['type'] == t]['osmid'].values[0]
        self.osmid_by_building_type['other_work_types'] = []
        for ind, t in enumerate(building_types):
            if t in ['residential', 'work']:
                continue
            self.osmid_by_building_type['other_work_types'].extend(groups_df[groups_df['type'] == t]['osmid'].values[0])

    def create_graph(self, data_frame: pd.DataFrame, building_params):
        if data_frame is None:
            raise ValueError('A valid data frame needs to be provided')
        graph = nx.Graph()
        for building in data_frame.iterrows():
            building = building[1]
            b = Building(building[0], wkt.loads(building[1]), building[4], building[2])
            self.districts[building[4]].buildings.append(building[0])
            floors_amount = random.choices(list(building_params.keys()), weights=list(building_params.values()))[0]
            b.n_apartments = get_apartments_number(floors_amount)
            graph.add_node(building[0], building=b)
        return graph

    def get_closest_osmid(self, osmid, btype):
        distances = [self.graph.nodes[osmid]["building"].coordinates.distance(self.graph.nodes[id]["building"].coordinates) for id in self.osmid_by_building_type[btype]]
        osmid = self.osmid_by_building_type[btype][np.argmin(distances)]
        assert osmid is not None
        return osmid

    def distribute_people(self, district: District,
                          age_dist: Tuple[float],
                          gender_dist: Tuple[float] = (0.5, 0.5)):
        for ind in tqdm(range(district.population)):
            while True:
                building_osmid = random.choice(district.buildings)
                b = self.graph.nodes[building_osmid]["building"]
                if not b.public:
                    break
            age = random.choices(range(len(age_dist)), weights=age_dist)[0]
            gender = random.choices([0, 1], weights=gender_dist)[0]
            work_place = None
            study_place = None
            if age == 0:
                kindergarten_or_none_types = ['kindergarten', None]
                kindergarten_or_none = random.choices(kindergarten_or_none_types, [0.3, 0.7])[0]
                study_place = (7, self.get_closest_osmid(building_osmid, 'kindergarten') if
                        kindergarten_or_none is not None else None)
            elif age == 1:
                school_or_none_types = ['school', None]
                school_or_none = random.choices(school_or_none_types, [0.3, 0.7])[0]
                study_place = (
                    7, self.get_closest_osmid(building_osmid, 'kindergarten')) if school_or_none is not None else None
            elif age == 2:
                work_or_study_types = ['work', 'study', 'both', None]
                work_or_study = random.choices(work_or_study_types,
                                               (0.3, 0.2, 0.55, 0.1))[0]  # TODO distribution should be changed
                if work_or_study == 'work':
                    work_place = (8, random.choices((random.choice(self.osmid_by_building_type['work']),
                                                     random.choice(self.osmid_by_building_type['other_work_types'])),
                                                    weights=(0.9, 0.1))[0])
                elif work_or_study == 'study':
                    study_place = (7, self.get_closest_osmid(building_osmid, 'university'))
                elif work_or_study == 'both':
                    study_place = (5, self.get_closest_osmid(building_osmid, 'university'))
                    work_place = (4, random.choices((random.choice(self.osmid_by_building_type['work']),
                                                     random.choice(self.osmid_by_building_type['other_work_types'])),
                                                    weights=(0.9, 0.1))[0])
            elif age == 3:
                work_none_types = ['work', None]
                work_none = random.choices(work_none_types, [0.82, 0.18])[0]
                work_places = [random.choice(self.osmid_by_building_type['work']),
                               random.choice(self.osmid_by_building_type['other_work_types'])]
                work_place = (8, (random.choices(work_places,
                                                 weights=(0.6, 0.4))[0])) if work_none is not None else None
            agent = EpiAgent(int(f'{str(district.id*100 + 10)[:3]}{ind}'), self, age, gender, work_place, study_place)
            self.grid.place_agent(agent, building_osmid)
            self.scheduler.add(agent)

    def get_b_ids_by_types(self, types: List[str]):
        ids = []
        for (p, d) in self.graph.nodes(data=True):
            for t in types:
                if d["building"].building_type == t:
                    ids.append(p)
        return ids

    def save_model(self, checkpoint_folder='checkpoints'):
        Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
        with open(f'{checkpoint_folder}/checkpoint_{self.day}_{self.hour}.pickle', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
