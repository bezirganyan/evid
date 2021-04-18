import argparse

from src.model import EpiModel
from src.model_viz import Visualizer

model = EpiModel(1000, 10, 10, n_boxes=7)
vis = Visualizer(model)
vis.create_animation()

parser = argparse.ArgumentParser()
parser.add_argument('--population_number', help='The number of population', type=int, required=True)
parser.add_argument('--box_size', help='The size of box enviroment', type=int, required=True)
parser.add_argument('--transmission_prob',
                    help='The probability of transmitting the disease from infected person to a healthy one',
                    type=float, required=True)
parser.add_argument('--mortality_rate', help='The mortality rate of the disease', type=float, required=True)
parser.add_argument('--inf_radius', help='The radius within which the disease can be transmitted', type=float,
                    required=True)
parser.add_argument('--healing_period', help='The amount of days after which the person is healed', type=int,
                    required=True)
parser.add_argument('--travel_prob', help='The probability of a person travelling to adel blocks', type=float,
                    required=True)
parser.add_argument('--healthcare_potential',
                    help='The portion of people from population, which the healthcare system can handle [0-1]',
                    type=float, required=True)
parser.add_argument('--file_name', help='Name of output animation file', type=str, required=True)

if __name__ == '__main__':
    args = vars(parser.parse_args())
    agent_args = {
        'population_number': args['population_number'],
        'width': args['box_size'],
        'height': args['box_size'],
        'transmission_prob': args['transmission_prob'],
        'mortality_rate': args['mortality_rate'],
        'inf_radius': args['inf_radius'],
        'healing_period': args['healing_period'],
        'travel_prob': args['travel_prob'],
        'healthcare_potential': args['healthcare_potential']}
    model = EpiModel(**agent_args)
    vis = Visualizer(model)
    vis.create_animation(save_path=args['file_name'])
