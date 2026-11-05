"""
feature_engineer.py

What is this file?

This file takes the raw JSON battle data and turns it into a dataframe with rows = 10000 battle and columns = number of features
that my machine learning models can understand. 

The code breaks the battle down into three main parts:
1. Static features: The starting stats and types of the lead Pokemon.
2. Timeline features: What happened turn-by-turn (HP drops, switches, faints, status conditions etc.).
3. Custom features: I created a custom "threat formula" to calculate how dangerous each attack was, 
   and I track the overall health of the team as the battle progresses.

Input: A list of battle dictionaries (from train.jsonl or test.jsonl) and the type chart.

Output: A Pandas DataFrame where each row is one battle, and each column is a feature.

"""


import numpy as np
import pandas as pd
from type_chart import calculate_type_multiplier, get_boost_multiplier


def calculate_stat_averages(train_data):

    # Calculate the average of stats
    # I need these averages because sometimes P2 switches to a Pokemon I don't 
    # have stats for. So I use the average as a fallback.
    avrg_atk, avrg_def, avrg_spa, avrg_spd = [], [], [], []

    for battle in train_data:
        for pokemon in battle.get('p1_team_details',[]):
            avrg_atk.append(pokemon.get('base_atk',0))
            avrg_def.append(pokemon.get('base_def',0))
            avrg_spa.append(pokemon.get('base_spa',0))
            avrg_spd.append(pokemon.get('base_spd',0))
            
    return np.mean(avrg_atk), np.mean(avrg_def), np.mean(avrg_spa), np.mean(avrg_spd) 

            
def build_features(battles, type_chart, avrg_stats):
    avrg_atk, avrg_def, avrg_spa, avrg_spd = avrg_stats
    pokemon_battle = []
    
    for battle in battles:
        battle_id = battle.get('battle_id')
        player_won = battle.get('player_won', None)
        
        # --- STATIC FEATURES (INITIAL MATCH UP)  ---
        p1_team = battle.get('p1_team_details', [])

        # get all the unique attacking types from the type chart
        # so that I can test p1's defense against everything
        atk_types = set(chart[0] for chart in type_chart.keys() )
        p1_unique_type = set()
        p1_weak_count = 0
        p1_resist_count = 0
        p1_immune_count = 0
        
        for pok in p1_team:
            p_types = pok.get('types',['notype'])

            # add unique type of p1
            for type in p_types:
                if type != 'notype':
                    p1_unique_type.add(type)

            # check the defensive matchup of p1 type against every  atk types
            for type in atk_types:
                multi = calculate_type_multiplier(type_chart, [type], p_types)
                if multi > 1.0: # p1 is weak
                    p1_weak_count += 1
                elif multi == 0.0: 
                    p1_immune_count += 1
                elif multi < 1.0:
                    p1_resist_count += 1

        p1_unique_type_count = len(p1_unique_type)

        # I observed four distinct multiplier classes:
        # 2.0 (weak), 1.0 (neutral), 0.5 (resist), and 0.0 (immune).
        # so i put a heuristic sum of these as: I give,
        # +1 for a resist,
        # +2 for immune (because taking 0 damage is better than half)
        # -1 for weak
        # defense_score = (resist + (immune * 2)) - weak
        p1_defense_score = (p1_resist_count + (p1_immune_count * 2)) - p1_weak_count

        

        # full team stat to calculate the averages
        p1_team_hp = [pok.get('base_hp', 0) for pok in p1_team]
        p1_team_atk = [pok.get('base_atk', 0) for pok in p1_team]
        p1_team_def = [pok.get('base_def', 0) for pok in p1_team]
        p1_team_spa = [pok.get('base_spa', 0) for pok in p1_team]
        p1_team_spd = [pok.get('base_spd', 0) for pok in p1_team]
        p1_team_spe = [pok.get('base_spe', 0) for pok in p1_team]

        
        p1_team_avg_hp = np.mean(p1_team_hp) if p1_team_hp else 0
        p1_team_avg_atk = np.mean(p1_team_atk) if p1_team_atk else 0
        p1_team_avg_def = np.mean(p1_team_def) if p1_team_def else 0
        p1_team_avg_spa = np.mean(p1_team_spa) if p1_team_spa else 0
        p1_team_avg_spd = np.mean(p1_team_spd) if p1_team_spd else 0
        p1_team_avg_spe = np.mean(p1_team_spe) if p1_team_spe else 0
        # maximum speed among the p1 team
        p1_team_max_spe = max(p1_team_spe) if p1_team_spe else 0 
        
        # A dictionary used to call base stat of p1 pokemons during battle
        p1_team_pokemon = {pok['name']: pok for pok in p1_team}

        # lead pokemon details
        p1_lead = p1_team[0] if len(p1_team) > 0 else {}
        p2_lead = battle.get('p2_lead_details', {}) or {}

        # p1 lead stats
        p1_lead_hp = p1_lead.get('base_hp', 0)
        p1_lead_atk = p1_lead.get('base_atk', 0)
        p1_lead_def = p1_lead.get('base_def', 0)
        p1_lead_spa = p1_lead.get('base_spa', 0)
        p1_lead_spd = p1_lead.get('base_spd', 0)
        p1_lead_spe = p1_lead.get('base_spe', 0)

        # p2 lead stats
        p2_lead_hp = p2_lead.get('base_hp', 0)
        p2_lead_atk = p2_lead.get('base_atk', 0)
        p2_lead_def = p2_lead.get('base_def', 0)
        p2_lead_spa = p2_lead.get('base_spa', 0)
        p2_lead_spd = p2_lead.get('base_spd', 0)
        p2_lead_spe = p2_lead.get('base_spe', 0)

        # lead types of p1 and p2
        p1_lead_types = p1_lead.get('types', [])
        p2_lead_types = p2_lead.get('types', [])

        # Initial Speed matchup - to check who is faster and calculate the ratio
        is_p1_faster = 1 if p1_lead_spe > p2_lead_spe else 0
        lead_speed_ratio = p1_lead_spe / max(p2_lead_spe, 1)
        log_lead_speed_ratio = np.log(max(p1_lead_spe, 1)/max(p2_lead_spe, 1))

        # calculate type multiplier at the start of the battle between the lead types of p1 and p2
        damage_multi_lead_p1p2 = calculate_type_multiplier(type_chart, p1_lead_types, p2_lead_types)
        damage_multi_lead_p2p1 = calculate_type_multiplier(type_chart, p2_lead_types, p1_lead_types)


        # We can introduce a fatality factor
        # If we do fatality_factor_p1p2 --> we get a number which reflect on how much hp
        # of p2 lost. We can do the same for p2 on p1

        #attack-defense-ratio = max(Value1, Value2)
        # Value1 = Atk/max(Def)
        # Value2 = Spa/max(Spd)

        # fatality_factor = (attack-defense-ratio*type_multiplier)/hp of defending
        
        p1_ad_ratio = max(p1_lead_atk/max(p2_lead_def, 1), p1_lead_spa/max(p2_lead_spd, 1))

        fatality_factor_p1p2 = (p1_ad_ratio*damage_multi_lead_p1p2)/max(p2_lead_hp, 1)


        p2_ad_ratio = max(p2_lead_atk/max(p1_lead_def, 1), p2_lead_spa/max(p1_lead_spd, 1))

        fatality_factor_p2p1 = (p2_ad_ratio*damage_multi_lead_p2p1)/max(p1_lead_hp, 1)
        
        delta_static_threat = fatality_factor_p1p2 - fatality_factor_p2p1
        
        # --- TIMELINE FEATURES ---
        timeline = battle.get('battle_timeline', [])
        total_turns = len(timeline) if len(timeline) > 0 else 1

        # initialize the counters for the battle
        p1_switch , p2_switch = 0, 0 # counter for how many times p1 and p2 switch
        p1_net_status, p2_net_status = 0, 0 # count total turn with  status conditions like tox, par, etc. for p1/p2
        p1_super_hit, p2_super_hit = 0, 0 # counter for super-effective move type multiplier (ie, > 1.0)  

        # move counter for status ratio
        p1_status_moves = 0
        p1_total_moves = 0
        p2_status_moves = 0
        p2_total_moves = 0

        # list for turn by turn
        p1_threats, p2_threats = [], [] # threat score using fatality factor for each turn
        p1_hp_diff = [] # difference between p1 hp and p2 hp  for every turn
        p1_accuracy, p1_power = [], [] # accuracy and power of each p1 move 
        p2_accuracy, p2_power = [], [] # accuracy and power of each p2 move

        # sets
        p1_faint_name, p2_faint_name = set(), set() # unique names of pokemon fainted

        # store the exact turn of the first faint
        p1_first_faint_turn = 30
        p2_first_faint_turn = 30

        t10_hp_diff, t20_hp_diff = None, None # a snapshot of hp differences at turn 10 and 20
        # for average boosts
        p1_sum_off_boost = 0 # record p1 atk/spa boost every turn 
        p2_sum_off_boost = 0 # record p2 atk/spa boost every turn
        p1_sum_spe_boost = 0 # record p1 speed boost every turn
        p2_sum_spe_boost = 0 # record p2 speed boost every turn
        
        # this is for finding max atk/spa and speed boost for p1 and p2 in each turns
        p1_max_off_boost = 0 
        p1_max_spe_boost = 0
        p2_max_off_boost = 0 
        p2_max_spe_boost = 0

        # this is to count the number of times the boost is positive in each turns
        p1_boosted_turns = 0
        p2_boosted_turns = 0
        
        
        # dict to store the hp of pokemon of p1 and p2 turn by turn
        p1_current_hp = {} # store the hp of latest pokemon appeared for p1
        p2_current_hp = {} # store the hp of latest pokemon appeared for p2

        #  specific status counters
        p1_par_turns, p2_par_turns = 0, 0 # paralyzed turns
        p1_slp_turns, p2_slp_turns = 0, 0 # asleep turns
        p1_brn_turns, p2_brn_turns = 0, 0 # burned turns
        p1_tox_turns, p2_tox_turns = 0, 0 # poisoned turns
        
        # enter to the turn by turn in battle timeline
        for turn_data in timeline:
            turn_num = turn_data.get('turn', 0)
            p1_move = turn_data.get('p1_move_details')
            p1_state = turn_data.get('p1_pokemon_state', {})
            p2_move = turn_data.get('p2_move_details')
            p2_state = turn_data.get('p2_pokemon_state', {})

            # counter for total moves and status moves for ratio calculation
            if p1_move is not None:
                p1_total_moves += 1
                if p1_move.get('category') == 'status':
                    p1_status_moves += 1

            if p2_move is not None:
                p2_total_moves += 1
                if p2_move.get('category') == 'status':
                    p2_status_moves += 1

            # track faints        
            # A pokemon fainted or dead when hp becomes 0
            if p1_state.get('hp_pct', 1) == 0:
                p1_faint_name.add(p1_state.get('name'))
                if p1_first_faint_turn == 30: p1_first_faint_turn = turn_num
            if p2_state.get('hp_pct', 1) == 0:
                p2_faint_name.add(p2_state.get('name'))
                if p2_first_faint_turn == 30: p2_first_faint_turn = turn_num

            # updating the hp dictionary for any pokemon on the field    
            p1_name = p1_state.get('name')
            if p1_name:
                p1_current_hp[p1_name] = p1_state.get('hp_pct', 1)

            p2_name = p2_state.get('name')
            if p2_name:
                p2_current_hp[p2_name] = p2_state.get('hp_pct', 1)
                

            # Calculate the hp difference between p1 and p2 pokemon at each turn
            current_hp_diff = p1_state.get('hp_pct', 1) - p2_state.get('hp_pct', 1)
            p1_hp_diff.append(current_hp_diff) # updating the list with hp difference at each turn

            # taking a snapshot of the hp difference at turn 10 and turn 20
            if turn_num == 10:
                t10_hp_diff = current_hp_diff
            if turn_num == 20:
                t20_hp_diff = current_hp_diff


            # boosts tracking for player 1
            p1_boosts = p1_state.get('boosts', {})
            p1_boost_atk = p1_boosts.get('atk', 0) 
            p1_boost_spa = p1_boosts.get('spa', 0) 
            p1_boost_spe = p1_boosts.get('spe', 0) 

            p1_sum_off_boost += (p1_boost_atk + p1_boost_spa)
            p1_sum_spe_boost += p1_boost_spe

            p1_max_off_boost = max(p1_max_off_boost, p1_boost_atk, p1_boost_spa)
            p1_max_spe_boost = max(p1_max_spe_boost, p1_boost_spe)

            # This is to calculate the number which a positive boost stat is maintained for palyer 1
            is_p1_boosted = False
            for b in p1_boosts.values():
                if b > 0:
                    is_p1_boosted = True
                    break
            if is_p1_boosted:
                p1_boosted_turns += 1

            # boosts tacking for player 2
            p2_boosts = p2_state.get('boosts', {})
            p2_boost_atk = p2_boosts.get('atk', 0) 
            p2_boost_spa = p2_boosts.get('spa', 0) 
            p2_boost_spe = p2_boosts.get('spe', 0) 

            p2_sum_off_boost += (p2_boost_atk + p2_boost_spa)
            p2_sum_spe_boost += p2_boost_spe

            
            p2_max_off_boost = max(p2_max_off_boost, p2_boost_atk, p2_boost_spa)
            p2_max_spe_boost = max(p2_max_spe_boost, p2_boost_spe)

            # This is to calculate the number which a positive boost stat is maintained for palyer 2
            is_p2_boosted = False
            for b in p2_boosts.values():
                if b > 0:
                    is_p2_boosted = True
                    break
            if is_p2_boosted:
                p2_boosted_turns += 1
                
                

            # status conditions           
            p1_status = p1_state.get('status', 'nostatus')
            p2_status = p2_state.get('status', 'nostatus')

            # counter for specific status turns
            if p1_status == 'par': p1_par_turns += 1
            elif p1_status == 'slp': p1_slp_turns += 1
            elif p1_status == 'brn': p1_brn_turns += 1
            elif p1_status == 'tox': p1_tox_turns += 1

            if p2_status == 'par': p2_par_turns += 1
            elif p2_status == 'slp': p2_slp_turns += 1
            elif p2_status == 'brn': p2_brn_turns += 1
            elif p2_status == 'tox': p2_tox_turns += 1
            
            # total turns with any status
            if p1_status != 'nostatus': p1_net_status += 1
            if p2_status != 'nostatus': p2_net_status += 1


            # A feature on switch - we have to check at each turn if there is no move
            # but it is alive (this is to distinguish switch from faint)
            if p1_move is None and p1_state.get('hp_pct', 0) > 0:
                p1_switch += 1
            if p2_move is None and p2_state.get('hp_pct', 0) > 0:
                p2_switch += 1
            
            # ---calculation of threat for p1---
            
            # A pokemon with power and alive of player 1 against player 2
            if p1_move and p1_move.get('base_power', 0) > 0 and p1_state.get('hp_pct', 0) > 0:
                p1_accuracy.append(p1_move.get('accuracy', 1.0))
                p1_power.append(p1_move.get('base_power', 1))
                # calling the current active pokemon of player 1
                p1_active_pokemon = p1_team_pokemon.get(p1_state.get('name'), {})
                # calling the current active pokemon of player 2
                # if p2 pokemon is the lead one, take it's stats
                # if p2 pokemon is other than lead then we use the average stats as guess
                if p2_state.get('name') == p2_lead.get('name'):
                    p2_def, p2_spd, p2_types = p2_lead_def, p2_lead_spd, p2_lead_types
                else:
                    p2_def, p2_spd, p2_types = avrg_def, avrg_spd, ['notype']

                # player 1 attack and special-attack boost on player 2 pokemon
                p1_atk_boost = get_boost_multiplier(p1_boosts.get('atk', 0))
                p1_spa_boost = get_boost_multiplier(p1_boosts.get('spa', 0))

                # player 2 defense and special-defense boost against player 1 pokemon
                p2_def_boost = get_boost_multiplier(p2_boosts.get('def', 0))
                p2_spd_boost = get_boost_multiplier(p2_boosts.get('spd', 0))
                
                # attack-defense ratio based on move category
                if p1_move.get('category') == 'physical':
                    
                    atk_def_ratio =((p1_active_pokemon.get('base_atk', avrg_atk)*p1_atk_boost)/max(p2_def*p2_def_boost, 1))
                else:
                    atk_def_ratio = ((p1_active_pokemon.get('base_spa', avrg_spa)*p1_spa_boost)/max(p2_spd*p2_spd_boost, 1))

                type_mult = calculate_type_multiplier(type_chart, [p1_move.get('type')], p2_types)

                if type_mult > 1.0:
                    p1_super_hit += 1
                                
                p2_hp = max(p2_state.get('hp_pct', 1.0), 0.10) # 0.10 is an arbitary value i choose to cap hp(can check other values)
                fatality_factor = (atk_def_ratio * type_mult * p1_move.get('base_power',1) * p1_move.get('accuracy', 1.0))/p2_hp

                p1_threats.append(fatality_factor)


            # ---calculation of threat for p2---
            
            # A pokemon with power and alive of player 2 against player 1
            if p2_move and p2_move.get('base_power', 0) > 0 and p2_state.get('hp_pct', 0) > 0:
                p2_accuracy.append(p2_move.get('accuracy', 1.0))
                p2_power.append(p2_move.get('base_power', 1))
                # calling the active pokemon of player 1
                p1_active_pokemon = p1_team_pokemon.get(p1_state.get('name'), {})
                p1_def, p1_spd = p1_active_pokemon.get('base_def', avrg_def), p1_active_pokemon.get('base_spd', avrg_spd)
                p1_types = p1_active_pokemon.get('types', ['notype'])
                
                # calling the active pokemon of player 2
                if p2_state.get('name') == p2_lead.get('name'):
                    p2_atk, p2_spa = p2_lead_atk, p2_lead_spa
                    p2_types = p2_lead_types
                else:
                    p2_atk, p2_spa = avrg_atk, avrg_spa
                    p2_types = ['notype']

                # player 2 attack and special-attack boost on player 1 pokemon
                p2_atk_boost = get_boost_multiplier(p2_boosts.get('atk', 0))
                p2_spa_boost = get_boost_multiplier(p2_boosts.get('spa', 0))

                # player 1 defense and special-defense boost against player 2 pokemon
                p1_def_boost = get_boost_multiplier(p1_boosts.get('def', 0))
                p1_spd_boost = get_boost_multiplier(p1_boosts.get('spd', 0))

                # attack-defense ratio based on move category                         
                if p2_move.get('category') == 'physical':
                    
                    atk_def_ratio =((p2_atk *p2_atk_boost)/max(p1_def*p1_def_boost, 1))
                else:
                    atk_def_ratio = ((p2_spa *p2_spa_boost)/max(p1_spd*p1_spd_boost, 1))

                type_mult = calculate_type_multiplier(type_chart, [p2_move.get('type')], p1_types)

                if type_mult > 1.0:
                    p2_super_hit += 1

                p1_hp = max(p1_state.get('hp_pct', 1.0), 0.10) # 0.10 is an arbitary value i choose to cap hp(can check other values)
                fatality_factor = (atk_def_ratio * type_mult * p2_move.get('base_power',1) * p2_move.get('accuracy', 1.0))/p1_hp

                p2_threats.append(fatality_factor)

                

                


        # ---  END OF THE BATTLE ------

        # The hp difference at the last turn, can be > 30 or < 30, so we take the last element of the list        
        last_hp_diff = p1_hp_diff[-1]  if p1_hp_diff else 0
        mean_hp_diff = np.mean(p1_hp_diff) if p1_hp_diff else 0
        hp_trajectory = (p1_hp_diff[-1] - p1_hp_diff[0]) if len(p1_hp_diff) > 1 else 0
        
        # fill in turn 10/20 snapshots if battle ended early
        if t10_hp_diff is None:
            t10_hp_diff = last_hp_diff
        if t20_hp_diff is None:
            t20_hp_diff = last_hp_diff

        # sum of hp of all pokemon appeared in the battle
        p1_team_sum_hp = sum(p1_current_hp.values())
        p2_team_sum_hp = sum(p2_current_hp.values())
        delta_sum_hp = p1_team_sum_hp - p2_team_sum_hp

        # number of unique pokemon appeared each battle
        p1_team_pok_count = len(p1_current_hp)
        p2_team_pok_count = len(p2_current_hp)

        # count healthy pokemon (hp > 50%) and critical pokemon (hp < 25%)
        p1_healthy_count = sum(1 for hp in p1_current_hp.values() if hp > 0.50)
        p2_healthy_count = sum(1 for hp in p2_current_hp.values() if hp > 0.50)
        delta_healthy_count = p1_healthy_count - p2_healthy_count

        p1_critical_count = sum(1 for hp in p1_current_hp.values() if 0 < hp < 0.25)
        p2_critical_count = sum(1 for hp in p2_current_hp.values() if 0 < hp < 0.25)
        delta_critical_count = p1_critical_count - p2_critical_count        

        # positive means p2 fainted first. negative means p1 fainted first.
        first_faint_advantage = p2_first_faint_turn - p1_first_faint_turn 
        
        # threat averages, I use log to compress big numbers
        p1_avrg_threat = np.log1p(np.mean(p1_threats)) if p1_threats else 0
        p2_avrg_threat = np.log1p(np.mean(p2_threats)) if p2_threats else 0
        delta_avrg_threat = p1_avrg_threat - p2_avrg_threat

        # max threats
        p1_max_threat = np.log1p(max(p1_threats)) if p1_threats else 0
        p2_max_threat = np.log1p(max(p2_threats)) if p2_threats else 0
        delta_max_threat = p1_max_threat - p2_max_threat

        # faints and squad advantage
        p1_faints = len(p1_faint_name)
        p2_faints = len(p2_faint_name)
        squad_advantage = (6 - p1_faints) - (6 - p2_faints)

        delta_super_hit = p1_super_hit - p2_super_hit

        # Calculate averages with  total_turns for each battle
        p1_avg_off_boost = p1_sum_off_boost / total_turns
        p2_avg_off_boost = p2_sum_off_boost / total_turns
        net_avg_off_boost = p1_avg_off_boost - p2_avg_off_boost

        p1_avg_spe_boost = p1_sum_spe_boost / total_turns
        p2_avg_spe_boost = p2_sum_spe_boost / total_turns
        net_avg_spe_boost = p1_avg_spe_boost - p2_avg_spe_boost

        net_max_off_boost =  p1_max_off_boost - p2_max_off_boost
        net_max_spe_boost = p1_max_spe_boost - p2_max_spe_boost
        net_boosted_turns = p1_boosted_turns - p2_boosted_turns 

        # status deltas
        net_par_turns = p1_par_turns - p2_par_turns
        net_slp_turns =  p1_slp_turns - p2_slp_turns
        net_brn_turns =  p1_brn_turns - p2_brn_turns
        net_tox_turns =  p1_tox_turns - p2_tox_turns
        net_status = p1_net_status - p2_net_status
        
        # switch rates
        p1_switch_rate = p1_switch/total_turns
        p2_switch_rate = p2_switch/total_turns
        
        # move stats
        p1_avrg_accuracy = np.mean(p1_accuracy) if p1_accuracy else 1.0
        p1_power_normalized = (np.mean(p1_power)/100) if p1_power else 0
        
        p2_avrg_accuracy = np.mean(p2_accuracy) if p2_accuracy else 1.0
        p2_power_normalized = (np.mean(p2_power)/100) if p2_power else 0

        delta_accuracy = p1_avrg_accuracy - p2_avrg_accuracy
        delta_power = p1_power_normalized - p2_power_normalized
        
        
        p1_status_move_ratio = p1_status_moves / p1_total_moves if p1_total_moves > 0 else 0
        p2_status_move_ratio = p2_status_moves / p2_total_moves if p2_total_moves > 0 else 0
        delta_status_move_ratio = p1_status_move_ratio - p2_status_move_ratio        
        
             
        #  all features into one dictionary
        battle_features = {
            'battle_id': battle_id,

            # static features
            'lead_speed_ratio': lead_speed_ratio,
            'log_lead_speed_ratio': log_lead_speed_ratio,
            'p1_team_avg_hp': p1_team_avg_hp,
            'p1_team_avg_atk': p1_team_avg_atk,
            'p1_team_avg_def': p1_team_avg_def,
            'p1_team_avg_spa': p1_team_avg_spa,
            'p1_team_avg_spd': p1_team_avg_spd,
            'p1_team_avg_spe': p1_team_avg_spe,
            'p1_team_max_spe': p1_team_max_spe,
            'p1_unique_type_count': p1_unique_type_count,
            'p1_weak_count': p1_weak_count,
            'p1_resist_count': p1_resist_count,
            'p1_immune_count': p1_immune_count,
            'p1_defense_score': p1_defense_score,
            'is_p1_faster': is_p1_faster,
            'damage_multi_lead_p1p2': damage_multi_lead_p1p2,
            'damage_multi_lead_p2p1': damage_multi_lead_p2p1,
            'delta_static_threat': delta_static_threat,
            # threat features
            'delta_avrg_threat': delta_avrg_threat,
            'p1_max_threat': p1_max_threat,
            'p2_max_threat': p2_max_threat,
            'delta_max_threat': delta_max_threat,
            # hp and battle flow
            'last_hp_diff': last_hp_diff,
            't10_hp_diff': t10_hp_diff,
            't20_hp_diff': t20_hp_diff,
            'hp_shifts': t20_hp_diff - t10_hp_diff,
            'mean_hp_diff': mean_hp_diff,
            'hp_trajectory': hp_trajectory,
            'p1_team_pok_count': p1_team_pok_count,
            'p2_team_pok_count': p2_team_pok_count,
            'delta_sum_hp': delta_sum_hp,
            'delta_healthy_count': delta_healthy_count,
            'delta_critical_count': delta_critical_count,

            # faints
            'p1_faints': p1_faints,
            'p2_faints': p2_faints,
            'squad_advantage': squad_advantage,
            'first_faint_advantage': first_faint_advantage,
            'delta_super_hit': delta_super_hit,

            # move stats
            'p1_avrg_accuracy': p1_avrg_accuracy,
            'p1_power_normalized': p1_power_normalized,
            'p2_avrg_accuracy': p2_avrg_accuracy,
            'p2_power_normalized': p2_power_normalized,
            'delta_accuracy': delta_accuracy,
            'delta_power': delta_power,
            'relative_damage_byp1': p1_avrg_accuracy * p1_power_normalized,
            'relative_damage_byp2': p2_avrg_accuracy * p2_power_normalized,
            
            # status and boosts
            'net_par_turns':net_par_turns,
            'net_slp_turns':net_slp_turns,
            'net_brn_turns':net_brn_turns,
            'net_tox_turns':net_tox_turns,
            'net_status': net_status,
            'net_max_off_boost': net_max_off_boost,
            'net_max_spe_boost': net_max_spe_boost,
            'net_boosted_turns': net_boosted_turns,
            'net_avg_off_boost': net_avg_off_boost,
            'net_avg_spe_boost': net_avg_spe_boost,
            'p1_status_move_ratio': p1_status_move_ratio,
            'p2_status_move_ratio': p2_status_move_ratio,
            'delta_status_move_ratio': delta_status_move_ratio,
            
            # swtich 
            'p1_switch_rate': p1_switch_rate,
            'p2_switch_rate': p2_switch_rate,
            'delta_switch' : p2_switch_rate - p1_switch_rate,
            
            'player_won': player_won
            
        }
        
        pokemon_battle.append(battle_features)
        
    return pd.DataFrame(pokemon_battle)
