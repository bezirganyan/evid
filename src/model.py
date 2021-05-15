import pickle
import random
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import SimultaneousActivation
from shapely import wkt
from tqdm import tqdm
from datetime import datetime, timedelta

from src.spaces import EpiNetworkGrid
from src.structures import EpiAgent, District, Building
from src.utils import Condition, compute_infected, compute_not_infected, compute_dead, compute_healed, \
    get_healthcare_potential, get_apartments_number


class EpiModel(Model):
    def __init__(self, population_number,
                 districts,
                 start_date: datetime,
                 moving_distribution_tensor,
                 building_types: List,
                 contact_prob,
                 building_params: Tuple[Tuple[int], Tuple[float]] = None,
                 age_dist: Tuple[float] = None,
                 inf_prob_args=None,
                 data_frame: pd.DataFrame = None,
                 graph: nx.Graph = None,
                 initial_infected=10,
                 step_size=1,
                 hospital_efficiency=0.3,
                 save_every=24,
                 severity_dist: object = {"asymptomatic": 0.24, "mild": 0.56, "severe": 0.2},
                 infection_countdown_dist: object = {"loc": 48, "scale": 7},
                 **kwargs):
        super().__init__()
        self.contact_prob = contact_prob
        self.save_every = save_every
        self.date = start_date
        self.districts = dict()
        self.dead_count = 0
        for district in districts:
            self.districts[district['name']] = District(**district)
        self.graph = graph if graph else self.create_graph(data_frame, building_params)
        self.num_agents = population_number
        self.grid = EpiNetworkGrid(self.graph)
        self.scheduler = SimultaneousActivation(self)
        self.inf_prob_args = inf_prob_args
        self.mortality_rate = kwargs.get('mortality_rate',
                                         {"asymptomatic": 0.0000001, "mild": 0.0002,
                                          "severe": 0.0004})  # TODO numbers should be changed
        self.healing_period = kwargs.get('healing_period', 14 * 24)
        self.hospital_beds = kwargs.get('hospital_beds', 5000)
        self.checkpoint_directory = kwargs.get('checkpoint_directory', 'checkpoints')
        # self.osmid_by_building_type = {'cafe':{'index':0,'osmid':[]} , 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [],
        #                                8: []}
        self.building_types = building_types
        self.osmid_by_building_type = dict()
        self.moving_distribution_tensor = moving_distribution_tensor
        self.severity_dist = severity_dist
        self.infection_countdown_dist = infection_countdown_dist
        self.step_size = step_size
        self.distribute_osmid_by_building_type(data_frame, building_types)
        self.hospital_efficiency = hospital_efficiency
        self.step_counts = 0

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

    def step(self):
        self.datacollector.collect(self)
        self.scheduler.step()
        self.date = self.date + timedelta(hours=self.step_size)
        self.step_counts += 1
        if self.step_counts % self.save_every == 0:
            self.save_model(self.checkpoint_directory)

    def distribute_osmid_by_building_type(self, data_frame, building_types):
        groups_df = data_frame[data_frame['type'] != 'building']
        groups_df = groups_df.groupby('type')['osmid'].apply(list).reset_index(name='osmid')
        for ind, t in enumerate(building_types[:-1]):
            self.osmid_by_building_type[ind] = {'type': t,
                                                'osmids': groups_df[groups_df['type'] == t]['osmid'].values[0]}
        other_work_places = []
        for key in self.osmid_by_building_type:
            if key != 8 and key != 2:
                other_work_places += self.osmid_by_building_type[key]['osmids']
        self.osmid_by_building_type[9] = {'type': 9,
                                          'osmids': other_work_places}

    def create_graph(self, data_frame: pd.DataFrame, building_params):
        if data_frame is None:
            raise ValueError('A valid data frame needs to be provided')
        graph = nx.Graph()
        for building in data_frame.iterrows():
            building = building[1]
            b = Building(building[0], wkt.loads(building[1]), building[3], building[2])
            self.districts[building[3]].buildings.append(building[0])
            floors_amount = random.choices(building_params[0], weights=building_params[1])[0]
            b.n_apartments = get_apartments_number(floors_amount)
            graph.add_node(building[0], building=b)
        return graph

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
                study_place = (7, random.choice(
                    self.osmid_by_building_type[3]['osmids'])) if kindergarten_or_none is not None else None
            elif age == 1:
                school_or_none_types = ['school', None]
                school_or_none = random.choices(school_or_none_types, [0.3, 0.7])[0]
                study_place = (
                    7, random.choice(self.osmid_by_building_type[4]['osmids'])) if school_or_none is not None else None
            elif age == 2:
                work_or_study_types = ['work', 'study', 'both', None]
                work_or_study = random.choices(work_or_study_types,
                                               (0.3, 0.1, 0.55, 0.2))[0]  # TODO distribution should be changed
                if work_or_study == 'work':
                    work_place = (8, random.choices((random.choice(self.osmid_by_building_type[9]['osmids']),
                                                     random.choice(self.osmid_by_building_type[8]['osmids'])),
                                                    weights=(0.9, 0.1))[0])
                elif work_or_study == 'study':
                    study_place = (7, random.choice(self.osmid_by_building_type[7]['osmids']))
                elif work_or_study == 'both':
                    study_place = (5, random.choice(self.osmid_by_building_type[7]['osmids']))
                    work_place = (4, random.choices((random.choice(self.osmid_by_building_type[9]['osmids']),
                                                     random.choice(self.osmid_by_building_type[8]['osmids'])),
                                                    weights=(0.9, 0.1))[0])
            elif age == 3:
                work_none_types = ['work', None]
                work_none = random.choices(work_none_types, [0.82, 0.18])[0]
                work_places = [random.choice(self.osmid_by_building_type[9]['osmids']),
                               random.choice(self.osmid_by_building_type[8]['osmids'])]
                work_place = (8, (random.choices(work_places,
                                                 weights=(0.6, 0.4))[0])) if work_none is not None else None
            agent = EpiAgent(int(f'{building_osmid}{ind}'), self, age, gender, work_place, study_place)
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
        with open(f'{checkpoint_folder}/checkpoint_{self.date}.pickle', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
