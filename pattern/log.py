import numpy as np

boo = ["accuracy", "atk", "def", "evasion", "spa", "spd", "spe"]

def translate_embedding(vec):
    assert vec.shape[0] == 105, f"Expected embedding length 105, got {len(vec)}"

    result = []

    # 1. 招式向量（4×15）
    result.append("== Move Features ==")
    for i in range(4):
        start = i * 15
        move = vec[start:start + 15]
        dmg = move[0]
        opp_boosts = move[1:8]
        self_boosts = move[8:15]
        result.append(f"Move {i + 1}: Damage Potential = {dmg:.5f}")
        if np.any(opp_boosts):
            result.append(f"  → Affects Opponent Boosts: " + ", ".join(
                f"{stat} {v*6:+.2f}" for stat, v in zip(boo, opp_boosts) if v != 0))
        if np.any(self_boosts):
            result.append(f"  → Affects Self Boosts: " + ", ".join(
                f"{stat} {v*6:+.2f}" for stat, v in zip(boo, self_boosts) if v != 0))

    # 2. 玩家能力提升（7）
    player_boosts = vec[60:67]
    result.append("\n== Player Boosts ==")
    for stat, val in zip(boo, player_boosts):
        if val != 0:
            result.append(f"  {stat}: {val*6:+.2f}")

    # 3. 敌人能力提升（7）
    opp_boosts = vec[67:74]
    result.append("\n== Opponent Boosts ==")
    for stat, val in zip(boo, opp_boosts):
        if val != 0:
            result.append(f"  {stat}: {val*6:+.2f}")

    # 4. 敌方当前HP（1）
    hp_opp = vec[74]
    result.append(f"\n== Opponent HP Ratio ==\n  {hp_opp:.2f}")

    # 5. 己方当前HP（1）
    hp_self = vec[75]
    result.append(f"\n== Your Active HP Ratio ==\n  {hp_self:.2f}")

    # 6. 敌方当前宝可梦已知技能的倍率（4）
    opp_moves_mults = vec[76:80]
    result.append("\n== Opponent Active Pokémon Move Multipliers ==")
    result.append("  " + ", ".join(f"x{v*4:.2f}" for v in opp_moves_mults))

    # 7. 可切换宝可梦（5只，每只5维：hp + 4技能倍率）
    team_vec = vec[80:]
    result.append("\n== Your Team Switch Candidates ==")
    for i in range(5):
        offset = i * 5
        hp_ratio = team_vec[offset]
        mults = team_vec[offset + 1:offset + 5]
        result.append(f"  Pokémon {i + 1}: HP = {hp_ratio:.2f}, Move Multipliers = " + ", ".join(f"x{m*4:.2f}" for m in mults))

    return "\n".join(result)
