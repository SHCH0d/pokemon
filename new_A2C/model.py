import numpy as np
import torch as th
from datetime import datetime
from stable_baselines3 import A2C
from gymnasium.spaces import Box
from poke_env.data import GenData

from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer

from CustomMLP import CustomActorCriticPolicy

boo = [
            "accuracy",
            "atk",
            "def",
            "evasion",
            "spa",
            "spd",
            "spe"
        ]

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

        #Player boost

        player_boosts=np.zeros(7);
        for i in range(7):
            player_boosts[i]=battle.active_pokemon._boosts[boo[i]]/6

        #Opponent boost

        opponent_boosts = np.zeros(7);
        for i in range(7):
            opponent_boosts[i] = battle.opponent_active_pokemon._boosts[boo[i]]/6

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                player_boosts,
                opponent_boosts,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def calc_reward(self, last_state, current_state) -> float:
        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=30
        )

    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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



NB_TRAINING_STEPS = 300000
NB_EVALUATION_EPISODES = 100
TEST_EPISODES = 100
GEN_9_DATA = GenData.from_gen(9)

num=datetime.now().strftime("%m%d%H%M")

np.random.seed(0)

if __name__ == "__main__":
    opponent = MaxDamagePlayer()
    env_player = SimpleRLPlayer(opponent=opponent)

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[64, 64], vf=[64, 64]))

    model = A2C("MlpPolicy", env_player, policy_kwargs=policy_kwargs, verbose=1,tensorboard_log="/XXZL/B3A/projectA/A2C/new_A2C/a2c_tensorboard"+num+"/")
    model.learn(total_timesteps=NB_TRAINING_STEPS)
    model.save(f"/XXZL/B3A/projectA/A2C/new_A2C/model{num}/")

    obs, reward, done, _, info = env_player.step(0)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, done, truncate, info = env_player.step(action)

    finished_episodes = 0

    env_player.reset_battles()
    obs, _ = env_player.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncate, info = env_player.step(action)

        if done:
            finished_episodes += 1
            obs, _ = env_player.reset()
            if finished_episodes >= TEST_EPISODES:
                break
        if truncate:
            print("!")
            obs, _ = env_player.reset()

    print("Won", env_player.n_won_battles, "battles against", env_player._opponent)
