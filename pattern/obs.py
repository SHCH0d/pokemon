import numpy as np

def decode_obs(obs: np.ndarray):
    """
    输入obs（76维），翻译成可读结构。

    结构来源于SimpleRLPlayerFlexible.embed_battle：
    - 4个招式，每个15维，共60维
      [damage(1), opponent boosts(7), self boosts(7)] ×4
    - player boosts(7)
    - opponent boosts(7)
    - 双方当前HP比例(2)
    """
    result = {}

    # 4招式 × 15维
    moves = []
    for i in range(4):
        base = i * 15
        dmg = obs[base]
        opp_boosts = obs[base+1:base+8]
        self_boosts = obs[base+8:base+15]
        moves.append({
            "damage": float(dmg),
            "opponent_boosts": opp_boosts.tolist(),
            "self_boosts": self_boosts.tolist(),
        })
    result["moves"] = moves

    # 玩家自身boosts
    player_boosts = obs[60:67]
    result["player_boosts"] = player_boosts.tolist()

    # 对手boosts
    opponent_boosts = obs[67:74]
    result["opponent_boosts"] = opponent_boosts.tolist()

    # 双方血量比例
    hp_self = obs[74]
    hp_opp = obs[75]
    result["hp_self"] = float(hp_self)
    result["hp_opp"] = float(hp_opp)

    return result

def decode_action(battle, action_idx: int):
    """
    根据动作索引和battle当前状态，返回动作描述（招式名）。
    """
    moves = battle.available_moves
    if not moves:
        return "No moves available"
    if action_idx >= len(moves):
        action_idx = 0
    move = moves[action_idx]
    return move.name  # 或 move.id，看你想要哪个字段
