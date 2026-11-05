"""
type_chart.py

What is this file?

This file contains the helper functions needed to handle the Generation 2 
type effectiveness chart and the stat boost multipliers. 

Instead of hardcoding the type matchups, I load them directly from the provided 
`type_chart_gen2.csv` file.

It has two main functions:
1. Type Multipliers: Figuring out if an attack is super effective (2.0x), 
   not very effective (0.5x), or immune (0.0x) against a defending Pokemon.
2. Boost Multipliers: Converting the integer stat boosts (like +2 Attack) into 
   the actual mathematical multipliers.

Input: The path to the CSV file, or specific types/stats to compare.
Output: A floating-point multiplier (e.g., 2.0, 0.5, 1.5).
"""

import pandas as pd

def load_type_chart(csv_path):
    # I load the CSV and turn it into a dictionary
    # I use a tuple as the key: (attacking_type, defending_type)
    df_type_chart = pd.read_csv(csv_path)
    return df_type_chart.set_index(['attacking_type', 'defending_type'])['multiplier'].to_dict()

def get_type_multiplier(type_chart, atk_type, def_type):
    # Looks up the multiplier for a single attacking type vs a single defending type
    # If a Pokemon has no second type, the data says 'notype' 
    # I just treat 'notype' as neutral (1.0x) 
    if atk_type == 'notype' or def_type == 'notype':
        return 1.0
    # If the matchup isn't in the CSV, I assume it's neutral (1.0x)
    return type_chart.get((atk_type, def_type), 1.0)

def calculate_type_multiplier(type_chart, attacking, defending):
    # This function handles Pokemon with two types 
    multiplier = []
    for atk_type in attacking:
        if atk_type == 'notype':
            continue
        total_multi = 1.0
        # Multiply the effectiveness against every type the defender has
        for def_type in defending:
            total_multi *= get_type_multiplier(type_chart, atk_type, def_type)
        multiplier.append(total_multi)

    # If the attacker has no valid types, return neutral
    if not multiplier:
        return 1.0
    
    # Since I don't know the exact moveset at the start of the battle, 
    # I return the maximum possible multiplier. This assumes the attacker 
    # will use their best available type matchup.
    return max(multiplier)

def get_boost_multiplier(boost):
    # In Pokemon, stat boosts don't just add numbers, they multiply the stat
    # Pokemon stat stages (from -6 to +6) use a fraction-based multiplier in the game engine
    # I am implementing the standard engine formula
    if boost >= 0:
        return (2+boost)/2
    else:
        return 2/(2-boost)

