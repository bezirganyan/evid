from src.model import EpiModel
import pandas as pd
import time
import yaml
from src.move_distribution import get_agent_movement_distribution
import sys



if __name__ == "__main__":
    with open(sys.argv[1], "r") as stream:
        cfg = yaml.safe_load(stream)

    # move_distribution_tensor = np.zeros(shape=(7, 5, 24, 10))
    city_df = pd.read_csv(cfg['files']['city_dataset_path'])

    model = EpiModel(**cfg['model'],
                     districts=cfg['districts'],
                     facility_conf=cfg['facilities'],
                     virus_conf=cfg['virus'],
                     people_conf=cfg['people'],
                     log_path=cfg['files']['log_path'],
                     city_data=city_df,
                     age_dist=cfg['people']['age_dist'],
                     severity_dist=cfg['virus']['infection_severity_dist'],
                     moving_distribution_tensor=get_agent_movement_distribution(cfg))

    for i in range(int(sys.argv[2])):
        t = time.time()
        model.step()
        d = time.time() - t
        r = model.datacollector.get_model_vars_dataframe().iloc[-1].values
        print(f'{i}: {model.day}:{model.weekday}:{model.hour} - Duration: {d},  I: {r[0]}, N: {r[1]}, D: {r[2]}, H: {r[3]}, P: {r[4]}')
