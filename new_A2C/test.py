import numpy as np
from datetime import datetime
from stable_baselines3 import A2C
from gymnasium.spaces import Box
from poke_env.data import GenData

from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer


GEN_9_DATA = GenData.from_gen(9)

import pandas as pd

# 示例字典
pokedex = GEN_9_DATA.pokedex

# 将字典的键提取为列表
keys = list(pokedex.keys())

# 将键转换为DataFrame
df = pd.DataFrame(keys, columns=['species'])

# 将DataFrame导出到Excel文件
df.to_excel('pokedex.xlsx', index=False)

skilldex = GEN_9_DATA.moves
keys = list(skilldex.keys())
df = pd.DataFrame(keys, columns=['skills'])

df.to_excel('movedex.xlsx', index=False)

print("字典的所有键已成功导出到keys.xlsx文件中")
