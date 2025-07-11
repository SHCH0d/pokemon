import numpy as np
import torch
from datetime import datetime
from stable_baselines3 import A2C, PPO
from player2 import SimpleRLPlayer, MaxDamagePlayer  # 你自定义的 player 文件
from log import translate_embedding

if __name__ == "__main__":
    # 初始化对战对手和环境
    opponent = MaxDamagePlayer()
    env_player = SimpleRLPlayer(opponent=opponent)

    # 加载模型（支持 PPO/A2C）
    model_path = r"D:\XXZL\B3A\project\A\A2C\pattern\model_07071940\ppo_model_100000_steps.zip"
    model = PPO.load(model_path, env=env_player)
    obs, reward, done, _, info = env_player.step(0)
    import torch
    while not done:
        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncate, info = env_player.step(action)

    # 重置环境，开始一个 episode
    obs, _ = env_player.reset()
    done = False
    step_id = 0
    print(translate_embedding(obs))
    print("\n按 Enter 执行一步，输入 q 退出调试。\n")
    
    while not done:
        inp = input(f"\n--- Step {step_id} ---\n> 按 Enter 继续，输入 q 退出: ")
        if inp.lower() == 'q':
            break

        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        print("##########################")
        
        print(f"动作 (action): {action}")

        obs, reward, done, truncated, info = env_player.step(action)
        
        print(translate_embedding(obs))

        # 输出 obs 信息（可定制）
        step_id += 1

    print("\nEpisode 完成。")
