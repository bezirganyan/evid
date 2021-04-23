import enum

def compute_infected(model):
    return sum(agent.condition == Condition.Infected or agent.condition == Condition.Quaranteened for agent in
               model.scheduler.agents)


def compute_dead(model):
    return model.dead_count


def compute_healed(model):
    return sum(agent.condition == Condition.Healed for agent in model.scheduler.agents)


def compute_not_infected(model):
    return sum(agent.condition == Condition.Not_infected for agent in model.scheduler.agents)


def get_healthcare_potential(model):
    return model.healthcare_potential * len(model.scheduler.agents)


def get_apartments_number(n_floors):
    if n_floors == 1 or n_floors == 2:
        apartments_number = 1
    elif 3 <= n_floors <= 6:
        apartments_number = n_floors * 3 * 4
    elif 7 <= n_floors <= 9:
        apartments_number = n_floors * 3 * 3
    else:
        apartments_number = n_floors * 5
    return apartments_number


class Condition(enum.Enum):
    Not_infected = 0
    Infected = 1
    Healed = 2
    Quaranteened = 3
    Dead = 4

