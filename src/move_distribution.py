import numpy as np

def get_probability_for_bt(building_type, possible_facilities, probs, n_types=10):
    if building_type in possible_facilities:
        prob_value = probs[possible_facilities.index(building_type)]
    else:
        prob_value = (1 - sum(probs)) / (n_types - len(possible_facilities))
    return prob_value

def get_agent_movement_distribution(cfg):
    # ageRange = [(0-4),(5-19),(20-29),(30-63),(64-120)]
    # building_type =[cafe, church,hospital, kindergarten, school,shop,sport,university,work,residential]
    building_type = list(cfg['facilities'].keys())
    print(building_type)
    tensor = np.zeros(shape=(7, 5, 24, len(building_type)))
    for wd in range(7):
        for age in range(5):
            for time in range(24):
                for i, bt in enumerate(building_type):
                    if 0 <= wd <= 4:
                        if age == 0:
                            if 10 <= time < 17:
                                pos_bts = ['cafe', 'kindergarten', 'residential', 'church', 'university', 'work', 'sport']
                                probs = [0.01, 0.4, 0.5, 0.01, 0, 0, 0]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            elif 17 <= time < 21:
                                pos_bts = ['cafe', 'kindergarten', 'residential', 'church', 'university', 'work', 'sport']
                                probs = [0.02, 0, 0.9, 0.01, 0, 0, 0]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)

                            else:
                                if bt == 'residential':
                                    prob_value = 1
                                else:
                                    prob_value = 0
                        elif age == 1:
                            if 8 <= time < 15:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten']
                                probs = [0.002, 0.5, 0.2, 0.1, 0]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)

                            elif 15 <= time < 22:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten', 'residential']
                                probs = [0.01, 0.001, 0.05, 0.2, 0, 0.6]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            else:
                                if bt == 'residential':
                                    prob_value = 1
                                else:
                                    prob_value = 0
                        elif age == 2:
                            if 9 <= time < 16:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten']
                                probs = [0.005, 0, 0.05, 0.7, 0]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            elif 16 <= time < 20:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten']
                                probs = [0.01, 0, 0.01, 0.4, 0]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            elif time >= 20 or time < 1:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten']
                                probs = [0.09, 0, 0., 0.09, 0]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            else:
                                if bt == 'residential':
                                    prob_value = 1
                                else:
                                    prob_value = 0
                        elif age == 3:
                            if 9 <= time <= 18:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten']
                                probs = [0.005, 0, 0.001, 0.8, 0]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)

                            elif time > 18 or time <= 1:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten', 'residential']
                                probs = [0.02, 0, 0.001, 0.1, 0, 0.7]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            else:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten', 'residential']
                                probs = [0.001, 0, 0.001, 0.001, 0, 0.9]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                        else:
                            if 8 <= time <= 16:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten', 'residential']
                                probs = [0.02, 0, 0.001, 0.1, 0, 0.6]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            elif 16 <= time <= 21:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten', 'residential']
                                probs = [0.1, 0, 0, 0.0001, 0, 0.6]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            else:
                                if bt == 'residential':
                                    prob_value = 1
                                else:
                                    prob_value = 0
                    else:
                        if age == 0:
                            if 10 <= time < 17:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten', 'residential']
                                probs = [0.1, 0, 0, 0.000, 0, 0.8]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            elif 17 <= time < 21:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten', 'residential']
                                probs = [0.1, 0, 0, 0.000, 0, 0.85]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            else:
                                if bt == 'residential':
                                    prob_value = 1
                                else:
                                    prob_value = 0
                        elif age == 1 or age == 2 or age == 3:
                            if 12 <= time < 18:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten', 'residential']
                                probs = [0.1, 0, 0, 0.000, 0, 0.85]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            elif 18 <= time < 22:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten', 'residential']
                                probs = [0.2, 0, 0, 0.000, 0, 0.5]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            else:
                                if bt == 'work':
                                    prob_value = 0.002
                                elif bt == 'residential':
                                    prob_value = 0.998
                                else:
                                    prob_value = 0

                        else:
                            if 8 <= time <= 16:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten', 'residential']
                                probs = [0.2, 0, 0, 0.000, 0, 0.6]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            elif 16 <= time <= 21:
                                pos_bts = ['cafe', 'school', 'university', 'work', 'kindergarten', 'residential']
                                probs = [0.2, 0, 0, 0.000, 0, 0.7]
                                prob_value = get_probability_for_bt(bt, pos_bts, probs)
                            else:
                                if bt == 'residential':
                                    prob_value = 1
                                else:
                                    prob_value = 0

                    tensor[wd, age, time, i] = prob_value
    return tensor
