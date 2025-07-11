import numpy as np
from datetime import datetime
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from gymnasium.spaces import Box
from poke_env.data import GenData

from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer
from poke_env.player.player import Player
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen9EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                    move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_9_DATA.type_chart

                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
                len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
                len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def calc_reward(self, last_state, current_state) -> float:
        return self.reward_computing_helper(
            current_state, fainted_value=0.02, hp_value=0.01, victory_value=1
        )

    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


class MyAgent(Player):
    def __init__(self, _model):
        super().__init__()
        self.model = _model

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                    move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_9_DATA.type_chart

                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
                len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
                len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:

        if action == -1:
            return ForfeitBattleOrder()
        elif (
                action < 4
                and action < len(battle.available_moves)
                and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif (
                not battle.force_switch
                and battle.can_z_move
                and battle.active_pokemon
                and 0 <= action - 4 < len(battle.active_pokemon.available_z_moves)
        ):
            return self.create_order(
                battle.active_pokemon.available_z_moves[action - 4], z_move=True
            )
        elif (
                battle.can_mega_evolve
                and 0 <= action - 8 < len(battle.available_moves)
                and not battle.force_switch
        ):
            return self.create_order(
                battle.available_moves[action - 8], mega=True
            )
        elif (
                battle.can_dynamax
                and 0 <= action - 12 < len(battle.available_moves)
                and not battle.force_switch
        ):
            return self.create_order(
                battle.available_moves[action - 12], dynamax=True
            )
        elif (
                battle.can_tera
                and 0 <= action - 16 < len(battle.available_moves)
                and not battle.force_switch
        ):
            return self.create_order(
                battle.available_moves[action - 16], terastallize=True
            )
        elif 0 <= action - 20 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 20])
        else:
            return self.choose_random_move(battle)

    def choose_move(self, battle):
        obs = self.embed_battle(battle)
        action, _ = self.model.predict(obs, deterministic=True)
        return self.action_to_move(action=action, battle=battle)


TRAINING_STEPS = 100_000
NB_EVALUATION_EPISODES = 100
GEN_9_DATA = GenData.from_gen(9)

num = datetime.now().strftime("%m%d%H%M")

np.random.seed(0)


def a2c_evaluation(env_player, model, nb_episodes):
    obs, reward, done, _, info = env_player.step(0)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)

    finished_episodes = 0
    env_player.reset_battles()
    obs, _ = env_player.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncate, info = env_player.step(action)

        if done:
            finished_episodes += 1
            obs, _ = env_player.reset()
            if finished_episodes >= nb_episodes:
                break
        if truncate:
            print("!")
            obs, _ = env_player.reset()

    return env_player.n_won_battles


opponent_path = "\XXZL\B3A\projectA\A2C\model\Maxdamegemodel.zip"
player_path = "\XXZL\B3A\projectA\A2C\model\model06210906.zip"

if __name__ == "__main__":
    opponent_model = A2C.load(opponent_path)
    opponent = MyAgent(opponent_model)

    env_player = SimpleRLPlayer(opponent=opponent)

    model = A2C.load(player_path)
    model.set_env(env_player)
    model.set_logger(configure("/XXZL/B3A/projectA/A2C/a2c_test", ["stdout", "tensorboard"]))

    model.learn(total_timesteps=TRAINING_STEPS)

    won_battles = a2c_evaluation(env_player=env_player, model=model, nb_episodes=NB_EVALUATION_EPISODES)

    print("Won", won_battles, "battles against", opponent_path)
