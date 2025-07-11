import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from gymnasium.spaces import Box
from poke_env.data import GenData

import torch
from player import SimpleRLPlayer, MaxDamagePlayer, TypedMaxPlayer

# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods


NB_EVALUATION_EPISODES = 100
TEST_EPISODES = 2000


num=datetime.now().strftime("%m%d%H%M")

np.random.seed(0)

model_store = {}

if __name__ == "__main__":
    opponent = TypedMaxPlayer()
    env_player = SimpleRLPlayer(opponent=opponent)
   #print(env_player.action_space_size())
    model = PPO.load(r"D:\XXZL\B3A\project\A\A2C\pattern\assist\model_07082340\ppo_model_900000_steps.zip")
    obs, reward, done, _, info = env_player.step(0)

    import torch
    while not done:
        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncate, info = env_player.step(action)

    # 批量测试
    finished_episodes = 0
    env_player.reset_battles()
    obs, _ = env_player.reset()
    while True:
        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        #print(action)
        obs, reward, done, truncate, info = env_player.step(action)

        if done:
            finished_episodes += 1
            obs, _ = env_player.reset()
            if finished_episodes >= TEST_EPISODES:
                break
        if truncate:
            print("!")
            obs, _ = env_player.reset()


    print("Won", env_player.n_won_battles/20, "battles against", env_player._opponent)
