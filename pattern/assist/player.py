import numpy as np
from gymnasium.spaces import Box
from poke_env.data import GenData
from myenvplayer import Gen9SimplePlayer
from poke_env.player import RandomPlayer

GEN_9_DATA = GenData.from_gen(9)
move_json = GEN_9_DATA.moves

class SimpleRLPlayer(Gen9SimplePlayer):
    def __init__(self, *args, fainted_value=2, hp_value=1, victory_value=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.fainted_value = fainted_value
        self.hp_value = hp_value
        self.victory_value = victory_value

    def embed_battle(self, battle):
        def normalize_stat(value):
            return min(max(value / 255, 0.0), 1.0)
    
        def boost_to_multiplier(boost: int) -> float:
            if boost >= 0:
                return (2 + boost) / 2
            else:
                return 2 / (2 - boost)

        def normalize_multiplier(m: float) -> float:
            return min(m / 4.0, 1.0)

        attacker = battle.active_pokemon
        defender = battle.opponent_active_pokemon

        atk = normalize_stat(attacker.stats["atk"])
        spa = normalize_stat(attacker.stats["spa"])
        atk_boost_multiplier = normalize_multiplier(boost_to_multiplier(attacker._boosts["atk"]))
        spa_boost_multiplier = normalize_multiplier(boost_to_multiplier(attacker._boosts["spa"]))
        def_boost_multiplier = normalize_multiplier(boost_to_multiplier(defender._boosts["def"]))
        spd_boost_multiplier = normalize_multiplier(boost_to_multiplier(defender._boosts["spd"]))

        move_vectors = []

        for i in range(4):
            if i < len(battle.available_moves):
                move = battle.available_moves[i]
                power = move.base_power
                accuracy = move.accuracy if move.accuracy is not None else 100
                expected_power = power * (accuracy / 100)

                #power 1(-1~3)
                power_norm = min(expected_power / 100.0, 3.0) if power > 0 else -1.0

                #mult 1(0~1)        
                multiplier = move.type.damage_multiplier(
                    defender.type_1, defender.type_2, type_chart=GEN_9_DATA.type_chart
                )
                stab = 1.5 if move.type in [attacker.type_1, attacker.type_2] else 1.0
                multiplier_norm = min(stab*multiplier / 6.0, 1.0)

                #category 8(0~1)
                if move.category == "Physical":
                    move_type_vector = [1.0, 0.0]
                elif move.category == "Special":
                    move_type_vector = [0.0, 1.0]
                else:
                    move_type_vector = [0.0, 0.0]

                attack_stats = [atk, spa]
                attack_boosts = [atk_boost_multiplier, spa_boost_multiplier]
                defense_boosts = [def_boost_multiplier, spd_boost_multiplier]

                move_vector = np.concatenate([
                    [power_norm],
                    [multiplier_norm],
                    move_type_vector,
                    attack_stats,
                    attack_boosts,
                    defense_boosts
                ]).astype(np.float32)
            else:
                move_vector = np.array([-1] + [0]*9, dtype=np.float32)

            move_vectors.append(move_vector)

        embedding = np.concatenate(move_vectors).astype(np.float32)
        return embedding

    def calc_reward(self, last_state, current_state) -> float:
    #    last_score = self.reward_computing_helper(last_state, fainted_value=self.fainted_value,
    #                                              hp_value=self.hp_value, victory_value=self.victory_value)
        current_score = self.reward_computing_helper(current_state, fainted_value=self.fainted_value,
                                                     hp_value=self.hp_value, victory_value=self.victory_value)
        return current_score

    def describe_embedding(self):
        low = np.concatenate([
            [-1], np.zeros(9),
            [-1], np.zeros(9),
            [-1], np.zeros(9),
            [-1], np.zeros(9),
        ])
        high = np.concatenate([
            [3], np.ones(9),
            [3], np.ones(9),
            [3], np.ones(9),
            [3], np.ones(9),
        ])
        return Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)

class MaxDamagePlayer(RandomPlayer):
    def __init__(self, *args, max_power_only=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_power_only = max_power_only  # 以后可以扩展更多策略参数

    def choose_move(self, battle):
        if battle.available_moves:
            if self.max_power_only:
                # 选择最大 base_power 的招式
                best = max(battle.available_moves, key=lambda m: m.base_power)
                return self.create_order(best)
            else:
                # 这里可以加入其他策略，比如随机选，或者加权选择
                return self.choose_random_move(battle)
        return self.choose_random_move(battle)

class TypedMaxPlayer(RandomPlayer):
    def choose_move(self, battle):
        def_type_1 = battle.opponent_active_pokemon.type_1
        def_type_2 = battle.opponent_active_pokemon.type_2

        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power * move.type.damage_multiplier(
                def_type_1,
                def_type_2,
                type_chart=GEN_9_DATA.type_chart
            ))
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

