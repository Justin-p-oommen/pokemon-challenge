import pandas as pd

def load_type_chart(csv_path):
    df_type_chart = pd.read_csv(csv_path)
    return df_type_chart.set_index(['attacking_type', 'defending_type'])['multiplier'].to_dict()

def get_multiplier(type_chart, atk_type, def_type):
    if atk_type == 'notype' or def_type == 'notype':
        return 1.0
    return type_chart.get((atk_type, def_type), 1.0)

def calculate_multiplier(type_chart, attacking, defending):
    multiplier = []
    for atk_type in attacking:
        if atk_type == 'notype':
            continue
        total_multi = 1.0
        for def_type in defending:
            total_multi *= get_multiplier(type_chart, atk_type, def_type)
        multiplier.append(total_multi)

    if not multiplier:
        return 1.0
    return max(multiplier)
