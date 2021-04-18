import enum


def compute_infected(model):
    return sum(agent.condition == Condition.Infected or agent.condition == Condition.Quaranteened for agent in
               model.schedule.agents)


def compute_dead(model):
    return sum(agent.condition == Condition.Dead for agent in model.schedule.agents)


def compute_healed(model):
    return sum(agent.condition == Condition.Healed for agent in model.schedule.agents)


def compute_not_infected(model):
    return sum(agent.condition == Condition.Not_infected for agent in model.schedule.agents)


def get_isol_boxes(n, width, height):
    return [(i * width / n, i * height / n) for i in range(1, n)]


def get_healthcare_potential(model):
    return model.healthcare_potential * len(model.schedule.agents)


class Condition(enum.Enum):
    Not_infected = 0
    Infected = 1
    Dead = 2
    Healed = 3
    Quaranteened = 4
