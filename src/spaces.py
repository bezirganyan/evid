import random
from typing import Any, List

from mesa import Agent
from mesa.space import NetworkGrid, GridContent


class EpiNetworkGrid(NetworkGrid):
    def __init__(self, G: Any) -> None:
        self.G = G

    def move_agent(self, agent: Agent, node_id: int) -> None:
        """ Move an agent from its current node to a new node. """
        self._remove_agent(agent, agent.pos)
        self._place_agent(agent, node_id)

    def place_agent(self, agent: Agent, node_id: int) -> None:
        """ Place a agent in a node. """
        self._place_agent(agent, node_id)

    def _place_agent(self, agent: Agent, node_id: int) -> None:
        """ Place the agent at the correct node. """
        building = self.G.nodes[node_id]["building"]
        building.place_agent(agent)

    def _remove_agent(self, agent: Agent, node_id: int) -> None:
        """ Remove an agent from a node. """
        try:
            building = self.G.nodes[node_id]["building"]
            building.remove_agent(agent)
        except KeyError as e:
            print(e)
            raise e


def is_cell_empty(self, node_id: int) -> bool:
    """ Returns a bool of the contents of a cell. """
    building = self.G.nodes[node_id]["building"]
    return not sum(list(map(len, list(building.apartments.values()))))


def iter_cell_list_contents(self, cell_list: List[int]) -> List[GridContent]:
    list_of_lists = [
        list(self.G.nodes[node_id]["building"].apartments.values())
        for node_id in cell_list
        if not self.is_cell_empty(node_id)
    ]
    return [item for sublist in list_of_lists for ap in sublist for item in ap]

