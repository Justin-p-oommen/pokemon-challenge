import numpy as np
import pandas as pd
from type_chart import calculate_type_multiplier, get_boost_multiplier


def calculate_stat_averages(train_data):

    # Calculate the average of stats to later fill if player 2 pokemon is unknown
    avrg_atk, avrg_def, avrg_spa, avrg_spd = [], [], [], []

    for battle in train_data:
        for pokemon in battle.get('p1_team_details',[]):
            avrg_atk.append(pokemon.get(base_atk,0))
            avrg_def.append(pokemon.get(base_def,0))
            avrg_spa.append(pokemon.get(base_spa,0))
            avrg_spd.append(pokemon.get(base_spd,0))
            
    return np.mean(avrg_atk), np.mean(avrg_def), np.mean(avrg_spa), np.mean(avrg_spd) 

            
def build_features(battles, type_chart, avrg_stats):


    avrg_atk, avrg_def, avrg_spa, avrg_spd = avrg_stats

    
    pokemon_battle = []
    
    for battle in battles:
        battle_id = battle.get('battle_id')
        player_won = battle.get('player_won', None)
        
        # --- STATIC FEATURES (INITIAL MATCH UP)  ---
        p1_team = battle.get('p1_team_details', [])
        
        p1_lead = p1_team[0] if len(p1_team) > 0 else {}
        p2_lead = battle.get('p2_lead_details', {}) or {}

        p1_lead_hp = p1_lead.get('base_hp', 0)
        p1_lead_atk = p1_lead.get('base_atk', 0)
        p1_lead_def = p1_lead.get('base_def', 0)
        p1_lead_spa = p1_lead.get('base_spa', 0)
        p1_lead_spd = p1_lead.get('base_spd', 0)
        p1_lead_spe = p1_lead.get('base_spe', 0)

        p1_stats = {

            'hp': p1_lead_hp
            'atk': p1_lead_atk
            'def': p1_lead_def
            'spa': p1_lead_spa
            'spd': p1_lead_spd
            'spe': p1_lead_spe
            
        }


        p2_lead_hp = p2_lead.get('base_hp', 0)
        p2_lead_atk = p2_lead.get('base_atk', 0)
        p2_lead_def = p2_lead.get('base_def', 0)
        p2_lead_spa = p2_lead.get('base_spa', 0)
        p2_lead_spd = p2_lead.get('base_spd', 0)
        p2_lead_spe = p2_lead.get('base_spe', 0)


        
        p2_stats = {

            'hp': p2_lead_hp
            'atk': p2_lead_atk
            'def': p2_lead_def
            'spa': p2_lead_spa
            'spd': p2_lead_spd
            'spe': p2_lead_spe
            
        }


        p1_lead_types = p1_lead.get('types', [])
        p2_lead_types = p2_lead.get('types', [])

        # Initial Speed matchup

        is_p1_faster = 1 if p1_lead_spe > p2_lead_spe else 0
        
        damage_multi_p1p2 = calculate_type_multiplier(type_chart, p1_lead_types, p2_lead_types)
        damage_multi_p2p1 = calculate_type_multiplier(type_chart, p2_lead_types, p1_lead_types)


        # We can introduce a damage_capacity or we can call fatality factor
        # fatality_factor = [(Attack/Defense)*type_multiplier]/ hp
        # If we do fatality_factor_p1p2 --> we get a number which reflect on how much hp
        # of p2 lost. We can do the same for p2 on p1

        #attack-defense-ratio = max(Value1, Value2)
        # Value1 = Atk/max(Def)
        # Value2 = Spa/max(Spd)

        # fatality_factor = (attack-defense-ratio*type_multiplier)/hp of defending
        
        p1_ad_ratio = max(p1_stats['atk']/max(p2_stats['def'],1), p1_stats['spa']/max(p2_stats['spd'],1))

        fatality_factor_p1p2 = (p1_ad_ratio*damage_multi_p1p2)/max(p2_stats['hp'],1)


        p2_ad_ratio = max(p2_stats['atk']/max(p1_stats['def'],1), p2_stats['spa']/max(p1_stats['spd'],1))

        fatality_factor_p2p1 = (p2_ad_ratio*damage_multi_p2p1)/max(p1_stats['hp'],1)

        

        # do I need to include also item advantage?


        # --- TIMELINE FEATURES ---
        timeline = battle.get('battle_timeline', [])
        #total_turns = len(timeline)
        
        p1_switch , p2_switch = 0, 0 # counter for how many times p1 and p2 switch
        p1_total_boost, p2_total_boost = 0, 0 # counter for the sum of stat  boost held on each turn
        p1_status, p2_status = 0, 0 # count total turn with  status conditions like tox, par, etc. for p1/p2
        p1_super_hit, p2_super_hit = 0, 0 # counter for super-effective move type multiplier (ie, > 1.0)  
        p1_threat, p2_threat = [], [] # threat score using fatality factor for each turn
        p1_hp_diff = [] # difference between p1 hp and p2 hp  for every turn
        p1_accuracy, p1_power = [], [] # accuracy and power of each p1 move 
        p1_faint_name, p2_faint_name = set(), set() # unique names of pokemon fainted 
        p1_plus2_atk, p1_plus2_spe = 0, 0 # boolean, if atk and speed reached +2 (a winning condition)
        t10_hp_diff, t20_hp_diff = None, None # a snapshot of hp differences at turn 10 and 20
        
        
        
        
        
        
        for turn_data in timeline:
            # Count P1 switches
            p1_move = turn_data.get('p1_move_details')
            p1_state = turn_data.get('p1_pokemon_state', {})
            if p1_move is None and p1_state.get('hp_pct', 0) > 0:
                p1_switches += 1
                

        #  all features into one dictionary
        battle_features = {
            'battle_id': battle_id,
            'is_p1_faster': is_p1_faster,
            'diff_p1atk_p2def': p1_lead_atk - p2_lead_def,
            'diff_p1spa_p2spd': p1_lead_spa - p2_lead_spd,
            'diff_p2atk_p1def': p2_lead_atk - p1_lead_def,
            'diff_p2spa_p1spd': p2_lead_spa - p1_lead_spd,
            'p1_mean_hp': p1_mean_hp,
            'p1_mean_spe': p1_mean_spe,
            'p1_mean_atk': p1_mean_atk,
            'p1_mean_def': p1_mean_def,
            'damage_multi_p1p2': damage_multi_p1p2,
            'damage_multi_p2p1': damage_multi_p2p1,
            #'p1_switch_rate': p1_switches / total_turns if total_turns > 0 else 0,
           # 'total_turns': total_turns,
           # 'player_won': int(player_won) if player_won is not None else 0 # Handle test set
            'player_won': player_won
            
        }
        
        initial_matchup.append(battle_features)
        
    return pd.DataFrame(initial_matchup)
