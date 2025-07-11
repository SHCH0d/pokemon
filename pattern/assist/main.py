import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from player import SimpleRLPlayer, MaxDamagePlayer

if __name__ == "__main__":
    NB_TRAINING_STEPS = 1000_000
    SAVE_INTERVAL = 100_000  # 每 10 万步保存一次模型
    TEST_EPISODES = 100
    num = datetime.now().strftime("%m%d%H%M")
    np.random.seed(0)

    opponent = MaxDamagePlayer()
#    opponent = DUMPlayer()
    env_player = SimpleRLPlayer(
        opponent=opponent,
        fainted_value=2,
        hp_value=1,
        victory_value=10
    )
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 128, 64, 32],
            vf=[256, 128, 64, 32]
        )
    )

    #policy_kwargs = dict(net_arch=[256, 128, 64, 32])

    model = PPO(
        "MlpPolicy", env_player,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.92,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cpu",
        tensorboard_log=f"D:/XXZL/B3A/project/A/A2C/pattern/assist/ppo_tensorboard_{num}/"
    )

    # ✅ 设置定期保存模型的 callback
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_INTERVAL,
        save_path=f"D:/XXZL/B3A/project/A/A2C/pattern/assist/model_{num}/",
        name_prefix="ppo_model"
    )

    # ✅ 开始训练并自动保存
    model.learn(
        total_timesteps=NB_TRAINING_STEPS,
        callback=checkpoint_callback
    )

    # ✅ 最终保存一次
    model.save(f"D:/XXZL/B3A/project/A/A2C/pattern/assist/model_{num}/final_model")

    # ✅ 简单测试一次
    obs, reward, done, _, info = env_player.step(0)
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)

    # ✅ 连续测试
    finished = 0
    env_player.reset_battles()
    obs, _ = env_player.reset()
    while finished < TEST_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env_player.step(action)
        if done or truncated:
            finished += 1
            obs, _ = env_player.reset()
