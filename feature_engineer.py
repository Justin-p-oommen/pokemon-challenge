import numpy as np
import pandas as pd
from type_chart import calculate_multiplier


def build_features(battles, type_chart):
    initial_matchup = []
    
    for battle in battles:
        battle_id = battle.get('battle_id')
        player_won = battle.get('player_won', None)
        
        # --- STATIC FEATURES ---
        p1_team = battle.get('p1_team_details', [])
        p1_lead = p1_team[0] if len(p1_team) > 0 else {}
        p2_lead = battle.get('p2_lead_details', {}) or {}

        p1_lead_hp = p1_lead.get('base_hp', 0)
        p1_lead_atk = p1_lead.get('base_atk', 0)
        p1_lead_def = p1_lead.get('base_def', 0)
        p1_lead_spa = p1_lead.get('base_spa', 0)
        p1_lead_spd = p1_lead.get('base_spd', 0)
        p1_lead_spe = p1_lead.get('base_spe', 0)
        
        p1_mean_hp = np.mean([p.get('base_hp', 0) for p in p1_team])
        p1_mean_spe = np.mean([p.get('base_spe', 0) for p in p1_team])
        p1_mean_atk = np.mean([p.get('base_atk', 0) for p in p1_team])
        p1_mean_def = np.mean([p.get('base_def', 0) for p in p1_team])

        p1_lead_types = p1_lead.get('types', [])

        p2_lead_hp = p2_lead.get('base_hp', 0)
        p2_lead_atk = p2_lead.get('base_atk', 0)
        p2_lead_def = p2_lead.get('base_def', 0)
        p2_lead_spa = p2_lead.get('base_spa', 0)
        p2_lead_spd = p2_lead.get('base_spd', 0)
        p2_lead_spe = p2_lead.get('base_spe', 0)

        p2_lead_types = p2_lead.get('types', [])

        is_p1_faster = 1 if p1_lead_spe > p2_lead_spe else 0
        
        damage_multi_p1p2 = calculate_multiplier(type_chart, p1_lead_types, p2_lead_types)
        damage_multi_p2p1 = calculate_multiplier(type_chart, p2_lead_types, p1_lead_types)

        # --- TIMELINE FEATURES ---
        timeline = battle.get('battle_timeline', [])
        total_turns = len(timeline)
        p1_switches = 0
        
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
            'p1_switch_rate': p1_switches / total_turns if total_turns > 0 else 0,
            'total_turns': total_turns,
            'player_won': int(player_won) if player_won is not None else 0 # Handle test set
        }
        
        initial_matchup.append(battle_features)
        
    return pd.DataFrame(initial_matchup)
