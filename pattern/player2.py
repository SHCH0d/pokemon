import numpy as np
from gymnasium.spaces import Box
from poke_env.data import GenData
from poke_env.player import RandomPlayer
from poke_env.environment.pokemon_type import PokemonType
from myenvplayer import Gen9SimplePlayer
from log import translate_embedding

boo = ["accuracy", "atk", "def", "evasion", "spa", "spd", "spe"]
GEN_9_DATA = GenData.from_gen(9)
move_json = GEN_9_DATA.moves

self_targets = {"self", "adjacentAllyOrSelf", "allAdjacent"}
opponent_targets = {"normal", "any", "randomNormal", "foeSide"}

class SimpleRLPlayer(Gen9SimplePlayer):
    def __init__(self, *args, fainted_value=2, hp_value=1, victory_value=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.fainted_value = fainted_value
        self.hp_value = hp_value
        self.victory_value = victory_value

    def embed_battle(self, battle):
        
        def encode_boosts(boosts: dict, weight: float = 1.0) -> np.ndarray:
            return np.array([boosts.get(stat, 0) * weight / 2 for stat in boo], dtype=np.float32)

        def move_multiplier_against(me, move):
            if move is None or move.category == "Status":
                return 0.0
            return move.type.damage_multiplier(me.type_1, me.type_2, type_chart=GEN_9_DATA.type_chart)

        def encode_move_effects_from_move(move) -> np.ndarray:
            move_data = move_json.get(move.id)
            if move_data is None:
                return np.zeros(14, dtype=np.float32)

            opp_boosts = np.zeros(7, dtype=np.float32)
            self_boosts = np.zeros(7, dtype=np.float32)
            boosts = move_data.get("boosts", {})
            target = move_data.get("target", "normal")
            if boosts:
                if target in self_targets:
                    self_boosts += encode_boosts(boosts, 1.0)
                elif target in opponent_targets:
                    opp_boosts += encode_boosts(boosts, 1.0)
            effects = []
            if isinstance(move_data.get("secondaries"), list):
                effects.extend(move_data["secondaries"])
            if isinstance(move_data.get("secondary"), dict):
                effects.append(move_data["secondary"])
            for eff in effects:
                ch = eff.get("chance", 100) / 100.0
                eb = eff.get("boosts", {})
                if eb:
                    if target in self_targets:
                        self_boosts += encode_boosts(eb, ch)
                    elif target in opponent_targets:
                        opp_boosts += encode_boosts(eb, ch)
            return np.concatenate([opp_boosts/6, self_boosts/6])

        def calc_move_damage_potential(move, attacker, defender):
            if move is None or move.base_power <= 0:
                return 0.0
            A = attacker.stats["atk"] if move.category == "Physical" else attacker.stats["spa"]
            P = move.base_power
            k = move.type.damage_multiplier(defender.type_1, defender.type_2, type_chart=GEN_9_DATA.type_chart)
            stab = 1.5 if move.type in [attacker.type_1, attacker.type_2] else 1.0
            total_multiplier = stab * k
            raw = A * P * total_multiplier
            return min(raw / 90000, 3.0)

        move_vectors = np.tile(np.array([-1] + [1]*7 + [-1]*7, dtype=np.float32), (4, 1))
        for i, move in enumerate(battle.available_moves):
            dmg = calc_move_damage_potential(move, battle.active_pokemon, battle.opponent_active_pokemon)
            vec14 = encode_move_effects_from_move(move)
            move_vectors[i] = np.concatenate(([dmg], vec14))

        player_boosts = np.array([battle.active_pokemon._boosts[s] / 6 for s in boo], dtype=np.float32)
        opponent_boosts = np.array([battle.opponent_active_pokemon._boosts[s] / 6 for s in boo], dtype=np.float32)
        hp_ratio_opp = battle.opponent_active_pokemon.current_hp_fraction or 0.0
        hp_ratio_self = battle.active_pokemon.current_hp_fraction or 0.0

        # 敌方已知技能（最多4）对我方当前宝可梦的伤害倍率
        known_moves = list(battle.opponent_active_pokemon._moves.values())[:4]
        opp_move_multipliers = []
        for move in known_moves:
            if move and move.type and battle.active_pokemon.type_1:
                mult = move_multiplier_against(battle.active_pokemon, move)
            else:
                mult = 0.0
            opp_move_multipliers.append(mult / 4)
        while len(opp_move_multipliers) < 4:
            opp_move_multipliers.append(0.0)

        # 我方其余 5 只可交换宝可梦，每个 [hp_ratio, 4×敌方技能倍率]
        switches = battle.available_switches
        switch_features = []
        for i in range(5):
            if i < len(switches):
                mon = switches[i]
                hp_ratio = mon.current_hp_fraction or 0.0
                multipliers = []
                for move in known_moves:
                    if move and move.type and mon.type_1:
                        mult = move_multiplier_against(mon, move)
                    else:
                        mult = 0.0
                    multipliers.append(mult / 4)
                while len(multipliers) < 4:
                    multipliers.append(0.0)
                switch_features.extend([hp_ratio] + multipliers)
            else:
                switch_features.extend([0.0, 1.0, 1.0, 1.0, 1.0])  # 填充

        embedding = np.concatenate([
            move_vectors.flatten(),  # 60
            player_boosts,          # 7
            opponent_boosts,        # 7
            [hp_ratio_opp],         # 1
            [hp_ratio_self],        # 1
            opp_move_multipliers,   # 4
            np.array(switch_features, dtype=np.float32)  # 25
        ]).astype(np.float32)
#        print("#####################################")
 #       print(translate_embedding(embedding))
        return embedding

    def calc_reward(self, last_state, current_state) -> float:
        base_reward = self.reward_computing_helper(
            current_state,
            fainted_value=self.fainted_value,
            hp_value=self.hp_value,
            victory_value=self.victory_value
        )
        penalty = -5.0 if self.miss_flag else 0.0
        return base_reward + penalty

    def describe_embedding(self):
        low = np.concatenate([
            #moves
            [-1], -np.ones(14),
            [-1], -np.ones(14),
            [-1], -np.ones(14),
            [-1], -np.ones(14),
            #boosts
            -np.ones(7),
            -np.ones(7),
            #hp
            np.zeros(2),
            #activate_resist
            np.zeros(4),
            #switch
            np.zeros(25)
            #60+14+2+4+25=105
        ])
        high = np.concatenate([
            #moves
            [3], np.ones(14),
            [3], np.ones(14),
            [3], np.ones(14),
            [3], np.ones(14),
            #boosts
            np.ones(7),
            np.ones(7),
            #hp
            np.ones(2),
            #activate_resist
            np.ones(4),
            #switch
            np.ones(25)
        ])
        return Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)

class MaxDamagePlayer(RandomPlayer):
    def __init__(self, *args, max_power_only=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_power_only = max_power_only

    def choose_move(self, battle):
        if battle.available_moves:
            if self.max_power_only:
                best = max(battle.available_moves, key=lambda m: m.base_power)
                return self.create_order(best)
            return self.choose_random_move(battle)
        return self.choose_random_move(battle)

class TypedMaxPlayer(RandomPlayer):
    def choose_move(self, battle):
        def_type_1 = battle.opponent_active_pokemon.type_1
        def_type_2 = battle.opponent_active_pokemon.type_2
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move:
                move.base_power * move.type.damage_multiplier(
                    def_type_1, def_type_2, type_chart=GEN_9_DATA.type_chart))
            return self.create_order(best_move)
        return self.choose_random_move(battle)

class DUMPlayer(RandomPlayer):
    def choose_move(self, battle):
        return self.choose_random_move(battle)
