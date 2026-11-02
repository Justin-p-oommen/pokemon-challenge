import numpy as np
import pandas as pd
from type_chart import calculate_type_multiplier, get_boost_multiplier


def calculate_stat_averages(train_data):

    # Calculate the average of stats to later fill if player 2 pokemon is unknown
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

        # A dictionary used to call base stat of p1 pokemons at each turn during battle
        p1_team_pokemon = {pok['name']: pok for pok in p1_team}
        
        p1_lead = p1_team[0] if len(p1_team) > 0 else {}
        p2_lead = battle.get('p2_lead_details', {}) or {}

        p1_lead_hp = p1_lead.get('base_hp', 0)
        p1_lead_atk = p1_lead.get('base_atk', 0)
        p1_lead_def = p1_lead.get('base_def', 0)
        p1_lead_spa = p1_lead.get('base_spa', 0)
        p1_lead_spd = p1_lead.get('base_spd', 0)
        p1_lead_spe = p1_lead.get('base_spe', 0)

        p1_stats = {

            'hp': p1_lead_hp,
            'atk': p1_lead_atk,
            'def': p1_lead_def,
            'spa': p1_lead_spa,
            'spd': p1_lead_spd,
            'spe': p1_lead_spe
            
        }


        p2_lead_hp = p2_lead.get('base_hp', 0)
        p2_lead_atk = p2_lead.get('base_atk', 0)
        p2_lead_def = p2_lead.get('base_def', 0)
        p2_lead_spa = p2_lead.get('base_spa', 0)
        p2_lead_spd = p2_lead.get('base_spd', 0)
        p2_lead_spe = p2_lead.get('base_spe', 0)


        
        p2_stats = {

            'hp': p2_lead_hp,
            'atk': p2_lead_atk,
            'def': p2_lead_def,
            'spa': p2_lead_spa,
            'spd': p2_lead_spd,
            'spe': p2_lead_spe
            
        }


        p1_lead_types = p1_lead.get('types', [])
        p2_lead_types = p2_lead.get('types', [])

        # Initial Speed matchup

        is_p1_faster = 1 if p1_lead_spe > p2_lead_spe else 0
        
        damage_multi_lead_p1p2 = calculate_type_multiplier(type_chart, p1_lead_types, p2_lead_types)
        damage_multi_lead_p2p1 = calculate_type_multiplier(type_chart, p2_lead_types, p1_lead_types)


        # We can introduce a damage_capacity or we can call fatality factor
        # fatality_factor = [(Attack/Defense)*type_multiplier]/ hp
        # If we do fatality_factor_p1p2 --> we get a number which reflect on how much hp
        # of p2 lost. We can do the same for p2 on p1

        #attack-defense-ratio = max(Value1, Value2)
        # Value1 = Atk/max(Def)
        # Value2 = Spa/max(Spd)

        # fatality_factor = (attack-defense-ratio*type_multiplier)/hp of defending
        
        p1_ad_ratio = max(p1_stats['atk']/max(p2_stats['def'],1), p1_stats['spa']/max(p2_stats['spd'],1))

        fatality_factor_p1p2 = (p1_ad_ratio*damage_multi_lead_p1p2)/max(p2_stats['hp'],1)


        p2_ad_ratio = max(p2_stats['atk']/max(p1_stats['def'],1), p2_stats['spa']/max(p1_stats['spd'],1))

        fatality_factor_p2p1 = (p2_ad_ratio*damage_multi_lead_p2p1)/max(p1_stats['hp'],1)
        
        delta_static_threat = fatality_factor_p1p2 - fatality_factor_p2p1
        
        # do I need to include also item advantage?


        # --- TIMELINE FEATURES ---
        timeline = battle.get('battle_timeline', [])

        total_turns = len(timeline) if len(timeline) > 0 else 1
        
        p1_switch , p2_switch = 0, 0 # counter for how many times p1 and p2 switch
        p1_total_boost, p2_total_boost = 0, 0 # counter for the sum of stat  boost held on each turn
        p1_status, p2_status = 0, 0 # count total turn with  status conditions like tox, par, etc. for p1/p2
        p1_super_hit, p2_super_hit = 0, 0 # counter for super-effective move type multiplier (ie, > 1.0)  
        p1_threats, p2_threats = [], [] # threat score using fatality factor for each turn
        p1_hp_diff = [] # difference between p1 hp and p2 hp  for every turn
        p1_accuracy, p1_power = [], [] # accuracy and power of each p1 move 
        p1_faint_name, p2_faint_name = set(), set() # unique names of pokemon fainted 
        p1_plus2_atk, p1_plus2_spe = 0, 0 # boolean, if atk and speed reached +2 (a winning condition)
        t10_hp_diff, t20_hp_diff = None, None # a snapshot of hp differences at turn 10 and 20
        
        
        
        
        
        
        for turn_data in timeline:
            turn_num = turn_data.get('turn', 0)

            p1_move = turn_data.get('p1_move_details')
            p1_state = turn_data.get('p1_pokemon_state', {})

            p2_move = turn_data.get('p2_move_details')
            p2_state = turn_data.get('p2_pokemon_state', {})

            # A pokemon fainted or dead when hp becomes 0
            if p1_state.get('hp_pct', 1) == 0:
                p1_faint_name.add(p1_state.get('name'))
            if p2_state.get('hp_pct', 1) == 0:
                p2_faint_name.add(p2_state.get('name'))

            # Calculate the hp difference between p1 and p2 pokemon at each turn
            current_hp_diff = p1_state.get('hp_pct', 1) - p2_state.get('hp_pct', 1)
            p1_hp_diff.append(current_hp_diff) # updating the list with hp difference at each turn

            # taking a snapshot of the hp difference at turn 10 and turn 20
            if turn_num == 10:
                t10_hp_diff = current_hp_diff
            if turn_num == 20:
                t20_hp_diff = current_hp_diff

            


                
            #if p1_move is None and p1_state.get('hp_pct', 0) > 0:
              #p1_switches += 1

            p1_boosts = p1_state.get('boosts', {})
            p2_boosts = p2_state.get('boosts', {})
            p1_total_boost += sum(p1_boosts.values())
            p2_total_boost += sum(p2_boosts.values())


            # A special coditon for win
            if p1_boosts.get('atk', 0) >= 2 or p1_boosts.get('spa', 0) >= 2:
                p1_plus2_atk = 1
            if p1_boosts.get('spe', 0) >= 2:
                p1_plus2_spe = 1


            # Making a feature out of status
            if p1_state.get('status', 'nostatus') != 'nostatus':
                p1_status += 1
            if p2_state.get('status', 'nostatus') != 'nostatus':
                p2_status += 1

            # A feature on switch - we have to check at each turn if there is a move
            # and is it alive(this is to distinguish switch from faint)
            if p1_move is None and p1_state.get('hp_pct', 0) > 0:
                p1_switch += 1
            if p2_move is None and p2_state.get('hp_pct', 0) > 0:
                p2_switch += 1

            # A pokemon with power and alive of player 1 against player 2
            if p1_move and p1_move.get('base_power', 0) > 0 and p1_state.get('hp_pct', 0) > 0:
                p1_accuracy.append(p1_move.get('accuracy', 1.0))
                p1_power.append(p1_move.get('base_power', 1))
                # calling the active pokemon of player 1
                p1_active_pokemon = p1_team_pokemon.get(p1_state.get('name'), {})
                # calling the active pokemon of player 2
                if p2_state.get('name') == p2_lead.get('name'):
                    p2_def, p2_spd, p2_types = p2_stats['def'], p2_stats['spd'], p2_lead_types
                else:
                    p2_def, p2_spd, p2_types = avrg_def, avrg_spd, ['notype']

                # player 1 attack and special-attack boost on player 2 pokemon
                p1_atk_boost = get_boost_multiplier(p1_boosts.get('atk', 0))
                p1_spa_boost = get_boost_multiplier(p1_boosts.get('spa', 0))

                # player 2 defense and special-defense boost against player 1 pokemon
                p2_def_boost = get_boost_multiplier(p2_boosts.get('def', 0))
                p2_spd_boost = get_boost_multiplier(p2_boosts.get('spd', 0))



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



            # A pokemon with power and alive of player 2 against player 1
            if p2_move and p2_move.get('base_power', 0) > 0 and p2_state.get('hp_pct', 0) > 0:
                #p1_accuracy.append(p1_move.get('accuracy', 1.0))
                #p1_power.append(p1_move.get('base_power', 1))
                # calling the active pokemon of player 1
                p1_active_pokemon = p1_team_pokemon.get(p1_state.get('name'), {})
                p1_def, p1_spd = p1_active_pokemon.get('base_def', avrg_def), p1_active_pokemon.get('base_spd', avrg_spd)
                p1_types = p1_active_pokemon.get('types', ['notype'])
                
                # calling the active pokemon of player 2
                if p2_state.get('name') == p2_lead.get('name'):
                    p2_atk, p2_spa = p2_stats['atk'], p2_stats['spa']
                else:
                    p2_atk, p2_spa = avrg_atk, avrg_spa

                # player 1 attack and special-attack boost on player 2 pokemon
                p2_atk_boost = get_boost_multiplier(p2_boosts.get('atk', 0))
                p2_spa_boost = get_boost_multiplier(p2_boosts.get('spa', 0))

                # player 2 defense and special-defense boost against player 1 pokemon
                p1_def_boost = get_boost_multiplier(p1_boosts.get('def', 0))
                p1_spd_boost = get_boost_multiplier(p1_boosts.get('spd', 0))



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

                

                

                    

        # The hp difference at the last turn, can be > 30 or < 30, so we take the last element of the list 
       
        last_hp_diff = p1_hp_diff[-1]  if p1_hp_diff else 0
        


        if t10_hp_diff is None:
            t10_hp_diff = last_hp_diff
        if t20_hp_diff is None:
            t20_hp_diff = last_hp_diff


        
        p1_avrg_threat = np.mean(p1_threats) if p1_threats else 0
        

        
        p2_avrg_threat = np.mean(p2_threats) if p2_threats else 0
        

        delta_avrg_threat = p1_avrg_threat - p2_avrg_threat

        p1_faints = len(p1_faint_name)
        p2_faints = len(p2_faint_name)

        squad_advantage = (6 - p1_faints) - (6 - p2_faints)

        delta_super_hit = p1_super_hit - p2_super_hit

        net_boosts = p1_total_boost - p2_total_boost

        net_status = p1_status - p2_status

        p1_switch_rate = p1_switch/total_turns

        p2_switch_rate = p2_switch/total_turns

        
        p1_avrg_accuracy = np.mean(p1_accuracy) if p1_accuracy else 1.0
        

        
        p1_power_normalized = (np.mean(p1_power)/100) if p1_power else 0
        
        
        
             
        #  all features into one dictionary
        battle_features = {
            'battle_id': battle_id,
            #'lead_hp_diff': p1_stats['hp'] - p2_stats['hp'],
            #'lead_atk_diff': p1_stats['atk'] - p2_stats['atk'],
            #'lead_def_diff': p1_stats['def'] - p2_stats['def'],
            #'lead_spa_diff': p1_stats['spa'] - p2_stats['spa'],
            #'lead_spd_diff': p1_stats['spd'] - p2_stats['spd'],
            #'lead_spe_diff': p1_stats['spe'] - p2_stats['spe'],
            'is_p1_faster': is_p1_faster,
            'damage_multi_lead_p1p2': damage_multi_lead_p1p2,
            'damage_multi_lead_p2p1': damage_multi_lead_p2p1,
            'delta_static_threat': delta_static_threat,
            'delta_avrg_threat': delta_avrg_threat,
            'last_hp_diff': last_hp_diff,
            't10_hp_diff': t10_hp_diff,
            't20_hp_diff': t20_hp_diff,
            'hp_shifts': t20_hp_diff - t10_hp_diff
            'p1_faints': p1_faints,
            'p2_faints': p2_faints,
            'squad_advantage': squad_advantage,
            'delta_super_hit': delta_super_hit,
            'p1_plus2_atk': p1_plus2_atk,
            'p1_plus2_spe': p1_plus2_spe,
            'p1_avrg_accuracy': p1_avrg_accuracy,
            'p1_power_normalized': p1_power_normalized,
            'relative_damage_byp1': p1_avrg_accuracy * p1_power_normalized,
            'net_boosts': net_boosts,
            'net_status': net_status,
            'p1_switch_rate': p1_switch_rate,
            'p2_switch_rate': p2_switch_rate,
            'delta_switch' : p2_switch_rate - p1_switch_rate,
            
            

            #'diff_p1atk_p2def': p1_lead_atk - p2_lead_def,
            #'diff_p1spa_p2spd': p1_lead_spa - p2_lead_spd,
            #'diff_p2atk_p1def': p2_lead_atk - p1_lead_def,
            #'diff_p2spa_p1spd': p2_lead_spa - p1_lead_spd,
            #'p1_mean_hp': p1_mean_hp,
            #'p1_mean_spe': p1_mean_spe,
            #'p1_mean_atk': p1_mean_atk,
            #'p1_mean_def': p1_mean_def,
            #'damage_multi_p1p2': damage_multi_p1p2,
            #'damage_multi_p2p1': damage_multi_p2p1,
            #'p1_switch_rate': p1_switches / total_turns if total_turns > 0 else 0,
           # 'total_turns': total_turns,
           # 'player_won': int(player_won) if player_won is not None else 0 # Handle test set
            'player_won': player_won
            
        }
        
        pokemon_battle.append(battle_features)
        
    return pd.DataFrame(pokemon_battle)
