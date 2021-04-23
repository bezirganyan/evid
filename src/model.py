import random
from typing import List, Tuple

import networkx as nx
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import SimultaneousActivation
from shapely import wkt
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np

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
                 mask_prob,
                 infection_prob,
                 mask_rule: int = 0,
                 building_params: Tuple[Tuple[int], Tuple[float]] = None,
                 age_dist: Tuple[float] = None,
                 data_frame: pd.DataFrame = None,
                 graph: nx.Graph = None,
                 **kwargs):
        super().__init__()
        self.date = start_date
        self.districts = dict()
        self.dead_count = 0
        for district in districts:
            self.districts[district['name']] = District(**district)
        self.graph = graph if graph else self.create_graph(data_frame, building_params)
        self.num_agents = population_number
        self.grid = EpiNetworkGrid(self.graph)
        self.scheduler = SimultaneousActivation(self)
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
        # self.osmid_by_building_type = {'cafe':{'index':0,'osmid':[]} , 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [],
        #                                8: []}  # TODO - 8 should be changed as it is workplace
        self.building_types = building_types
        self.osmid_by_building_type = dict()
        self.moving_distribution_tensor = moving_distribution_tensor
        self.mask_rule = mask_rule
        self.mask_prob = mask_prob
        self.infection_prob = infection_prob

        for d in self.districts.values():
            self.distribute_people(d, age_dist)

        self.random.choice(self.scheduler.agents).condition = Condition.Infected

        self.datacollector = DataCollector(model_reporters={
            "Infected": compute_infected,
            "Not_infected": compute_not_infected,
            "Dead": compute_dead,
            "Healed": compute_healed,
            "Healthcare_potential": get_healthcare_potential})
        self.distribute_osmid_by_building_type(data_frame, building_types)

    def step(self):
        self.datacollector.collect(self)
        self.scheduler.step()

        infected = compute_infected(self)
        hp = self.healthcare_potential * len(self.scheduler.agents)
        if self.healthcare_potential and infected > hp:
            self.healthcare_potential *= 1 - (infected - hp) / len(self.scheduler.agents) * 0.15
            self.mortality_rate *= 1 + (infected - hp) / len(self.scheduler.agents) * 0.15
        self.date = self.date + timedelta(hours=1)

    def create_graph(self, data_frame: pd.DataFrame, building_params):
        if data_frame is None:
            raise ValueError('A valid data frame needs to be provided')
        graph = nx.Graph()
        for building in data_frame.iterrows():
            building = building[1]
            b = Building(building[0], wkt.loads(building[1]), building[3], building[2])  # district problem
            self.districts[building[3]].buildings.append(building[0])
            floors_amount = random.choices(building_params[0], weights=building_params[1])[0]
            b.n_apartments = get_apartments_number(floors_amount)
            graph.add_node(building[0], building=b)
        return graph

    def distribute_people(self, district: District,
                          age_dist: Tuple[float],
                          gender_dist: Tuple[float] = (0.5, 0.5)):
        for ind in tqdm(range(district.population)):  # TODO remove tqdm
            while True:
                building_osmid = random.choice(district.buildings)
                if not self.graph.nodes[building_osmid]["building"].public:
                    break
            age = random.choices(range(len(age_dist)), weights=age_dist)[0]
            gender = random.choices([0, 1], weights=gender_dist)[0]
            agent = EpiAgent(int(f'{building_osmid}{ind}'), self, age, gender)
            self.grid.place_agent(agent, building_osmid)
            self.scheduler.add(agent)

    def get_b_ids_by_types(self, types: List[str]):
        ids = []
        for (p, d) in self.graph.nodes(data=True):
            for t in types:
                if d["building"].building_type == t:
                    ids.append(p)
        return ids

    def distribute_osmid_by_building_type(self, data_frame, building_types):
        groups_df = data_frame[data_frame['type'] != 'building']
        groups_df = groups_df.groupby('type')['osmid'].apply(list).reset_index(name='osmid')
        for ind, t in enumerate(building_types[:-1]):
            self.osmid_by_building_type[ind] = {'type': t,
                                                'osmids': groups_df[groups_df['type'] == t]['osmid'].values[0]}
