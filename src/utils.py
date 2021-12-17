import enum
import math
import os

from tqdm import tqdm


class Logger:
    """
    Logger manager
    """
    def __init__(self, contact_log_path, status_log_path, statistics_log_path, model, write_every=None):
        self.statistics_log_path = statistics_log_path
        self.contact_log_path = contact_log_path
        self.status_log_path = status_log_path
        self.model = model
        self.write_every = write_every
        if self.statistics_log_path:
            directory = os.path.dirname(self.statistics_log_path)
            os.makedirs(directory, mode=0o777, exist_ok=True)
            with open(f'{self.statistics_log_path}/statistics.csv', 'w+') as f:
                print('day,weekday,hour,infected,not_infected,dead,healed,icu_beds', file=f)

    def write_log(self):
        if self.contact_log_path:
            self.write_contact_log()
        if self.status_log_path:
            self.write_status_log()
        if self.statistics_log_path:
            self.write_statistics_log()

    def write_contact_log(self):
        directory = os.path.dirname(self.contact_log_path)
        os.makedirs(directory, mode=0o777, exist_ok=True)
        with open(f'{self.contact_log_path}/cont_{self.model.day}.csv', 'w+') as f:
            print("Writing Contact Logs...")
            print('agent_id,contact_id,day,dow,hour,building_osmid,infection',
                  file=f)
            for agent in tqdm(self.model.scheduler.agents):
                f.writelines('\n'.join(agent.daily_contacts))
                if agent.daily_contacts:
                    f.write('\n')
                agent.daily_contacts = []

    def write_status_log(self):
        directory = os.path.dirname(self.status_log_path)
        os.makedirs(directory, mode=0o777, exist_ok=True)
        with open(f'{self.status_log_path}/stat_{self.model.day}.csv', 'w+') as f:
            print("Writing Status Logs...")
            print('agent_id,day,in_hospital,condition,severity,building_osmid,age',
                  file=f)
            for agent in tqdm(self.model.scheduler.agents):
                sev = agent.severity.name if agent.severity is not None else "healthy"
                l = f'{agent.unique_id},{self.model.day},{agent.in_hospital},{agent.condition.name},{sev},{agent.address},{agent.age}'
                f.write(l)
                f.write('\n')

    def write_statistics_log(self):
        r = self.model.datacollector.get_model_vars_dataframe().iloc[-1].values
        with open(f'{self.statistics_log_path}/statistics.csv', 'a+') as f:
            print(f'{self.model.day},{self.model.weekday},{self.model.hour},{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}', file=f)


def compute_infected(model):
    return sum(agent.condition == Condition.Infected for agent in
               model.scheduler.agents)


def compute_dead(model):
    return model.dead_count


def compute_healed(model):
    return sum(agent.condition == Condition.Healed for agent in model.scheduler.agents)


def compute_not_infected(model):
    return sum(agent.condition == Condition.Not_infected for agent in model.scheduler.agents)


def get_healthcare_potential(model):
    return model.hospital_beds


def compute_inf_prob(duration: int = 1, rlwr=0.35, lifetime=1.7, area=100, height=4, d50=316, speak_frac=3, conc=5e+8,
                     mwd=5, conc_b=0.06, conc_s=0.6, vol=5.32, mask_out=0, mask_in=0, atv=10, depo=0.5):
    """
    :param duration: Duration of exposure (in hours)
    :param rlwr: air exchange rate [/h] [0.35=no ventilation, 2=burst ventilation once per h, 6=public places/supermarket]
    :param lifetime: lifetime of the virus in air [h]
    :param area: Area of the room
    :param height: height [m]
    :param d50: dose of virus at 50% infection probability
    :param speak_frac: fraction of speaking [0-100%]
    :param conc: "The reported viral RNA concentration of approximately 5 × 10^8 / mL represents the category of
    highly infectious patients and represents approximately 20% of individuals tested positive for SARS-CoV-2"
    [10^7 - 10^11]
    :param mwd: effective Mean diameter [µm] [1-50]
    :param conc_b: number concentration while breathing [#/cm³]  [0.001 - 1]
    :param conc_s: number concentration while speaking [#/cm³]  [0.01 - 9]
    :param vol: Speaking volume [1=quietly, 3=loud, 4..9= singing/screaming] [1 - 9]
    :param mask_out: mask efficiency (exhale) [0-1; surgical mask ~0.7, everyday mask (2 fabric layers) ~0.5] [0 - 1]
    :param mask_in: mask efficiency (inhale) [0-1; surgical mask ~0.5 b   , community mask (2 fabric layers) ~0.2] [0 - 1]
    :param atv: respiratory rate [l/min] [7.5-15; adult=10]
    :param depo: deposition efficiency lung
    :return: individual infection risk if one person is infectious
    """

    i_prop = 1 - math.exp(math.log(0.5) / d50)
    rna_in_aerosol = conc * math.pi / 6 * math.pow(mwd / 10000, 3)
    a_emis = (conc_b * (1 - speak_frac / 100) + conc_s * speak_frac / 100 * math.pow(2, vol - 2)) * 1000 * atv * 60 * (
            1 - mask_out)
    room_vol = (area * height)
    ssc_aero = a_emis / ((rlwr + 1 / lifetime) * room_vol * 1000)
    ssc_rna = ssc_aero * rna_in_aerosol
    rna_uptake_h = atv * 60 * depo * ssc_rna
    rna_dos = rna_uptake_h * duration * (1 - mask_in)
    sin_inf = (1 - math.pow(1 - i_prop, rna_dos))

    return sin_inf


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
    Dead = 3


class Severity(enum.Enum):
    asymptomatic = 0
    mild = 1
    severe = 2
