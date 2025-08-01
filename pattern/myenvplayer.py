"""This module defines a player class exposing the Open AI Gym API with utility functions.
"""

from abc import ABC
from threading import Lock
from typing import Dict, List, Optional, Union

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
from poke_env.player.openai_api import ActType, ObsType, OpenAIGymEnv
from poke_env.player.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder


class EnvPlayer(OpenAIGymEnv[ObsType, ActType], ABC):
    """Player exposing the Open AI Gym Env API."""

    _ACTION_SPACE: List[int] = []
    _DEFAULT_BATTLE_FORMAT = "gen8randombattle"

    def __init__(
        self,
        opponent: Optional[Union[Player, str]],
        account_configuration: Optional[AccountConfiguration] = None,
        *,
        avatar: Optional[int] = None,
        battle_format: Optional[str] = None,
        log_level: Optional[int] = None,
        save_replays: Union[bool, str] = False,
        server_configuration: Optional[ServerConfiguration] = None,
        start_listening: bool = True,
        start_timer_on_battle_start: bool = False,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        team: Optional[Union[str, Teambuilder]] = None,
        start_challenging: bool = True,
    ):
        """
        :param opponent: Opponent to challenge.
        :type opponent: Player or str, optional
        :param account_configuration: Player configuration. If empty, defaults to an
            automatically generated username with no password. This option must be set
            if the server configuration requires authentication.
        :type account_configuration: AccountConfiguration, optional
        :param avatar: Player avatar id. Optional.
        :type avatar: int, optional
        :param battle_format: Name of the battle format this player plays. Defaults to
            gen8randombattle.
        :type battle_format: Optional, str. Default to randombattles, with specifics
            varying per class.
        :param log_level: The player's logger level.
        :type log_level: int. Defaults to logging's default level.
        :param save_replays: Whether to save battle replays. Can be a boolean, where
            True will lead to replays being saved in a potentially new /replay folder,
            or a string representing a folder where replays will be saved.
        :type save_replays: bool or str
        :param server_configuration: Server configuration. Defaults to Localhost Server
            Configuration.
        :type server_configuration: ServerConfiguration, optional
        :param start_listening: Whether to start listening to the server. Defaults to
            True.
        :type start_listening: bool
        :param start_timer_on_battle_start: Whether to automatically start the battle
            timer on battle start. Defaults to False.
        :type start_timer_on_battle_start: bool
        :param ping_interval: How long between keepalive pings (Important for backend
            websockets). If None, disables keepalive entirely.
        :type ping_interval: float, optional
        :param ping_timeout: How long to wait for a timeout of a specific ping
            (important for backend websockets.
            Increase only if timeouts occur during runtime).
            If None pings will never time out.
        :type ping_timeout: float, optional
        :param team: The team to use for formats requiring a team. Can be a showdown
            team string, a showdown packed team string, of a ShowdownTeam object.
            Defaults to None.
        :type team: str or Teambuilder, optional
        :param start_challenging: Whether to automatically start the challenge loop
            or leave it inactive.
        :type start_challenging: bool
        """
        self._reward_buffer: Dict[AbstractBattle, float] = {}
        self._opponent_lock = Lock()
        self._opponent = opponent
        b_format = self._DEFAULT_BATTLE_FORMAT
        if battle_format:
            b_format = battle_format
        if opponent is None:
            start_challenging = False
        super().__init__(
            account_configuration=account_configuration,
            avatar=avatar,
            battle_format=b_format,
            log_level=log_level,
            save_replays=save_replays,
            server_configuration=server_configuration,
            start_listening=start_listening,
            start_timer_on_battle_start=start_timer_on_battle_start,
            team=team,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            start_challenging=start_challenging,
        )

    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
    ) -> float:
        """A helper function to compute rewards.

        The reward is computed by computing the value of a game state, and by comparing
        it to the last state.

        State values are computed by weighting different factor. Fainted pokemons,
        their remaining HP, inflicted statuses and winning are taken into account.

        For instance, if the last time this function was called for battle A it had
        a state value of 8 and this call leads to a value of 9, the returned reward will
        be 9 - 8 = 1.

        Consider a single battle where each player has 6 pokemons. No opponent pokemon
        has fainted, but our team has one fainted pokemon. Three opposing pokemons are
        burned. We have one pokemon missing half of its HP, and our fainted pokemon has
        no HP left.

        The value of this state will be:

        - With fainted value: 1, status value: 0.5, hp value: 1:
            = - 1 (fainted) + 3 * 0.5 (status) - 1.5 (our hp) = -1
        - With fainted value: 3, status value: 0, hp value: 1:
            = - 3 + 3 * 0 - 1.5 = -4.5

        :param battle: The battle for which to compute rewards.
        :type battle: AbstractBattle
        :param fainted_value: The reward weight for fainted pokemons. Defaults to 0.
        :type fainted_value: float
        :param hp_value: The reward weight for hp per pokemon. Defaults to 0.
        :type hp_value: float
        :param number_of_pokemons: The number of pokemons per team. Defaults to 6.
        :type number_of_pokemons: int
        :param starting_value: The default reference value evaluation. Defaults to 0.
        :type starting_value: float
        :param status_value: The reward value per non-fainted status. Defaults to 0.
        :type status_value: float
        :param victory_value: The reward value for winning. Defaults to 1.
        :type victory_value: float
        :return: The reward.
        :rtype: float
        """
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        to_return = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value

        return to_return

    def action_space_size(self) -> int:
        return len(self._ACTION_SPACE)

    def get_opponent(self) -> Union[Player, str, List[Player], List[str]]:
        with self._opponent_lock:
            if self._opponent is None:
                raise RuntimeError(
                    "Unspecified opponent. "
                    "Specify it in the constructor or use set_opponent"
                )
            return self._opponent

    def set_opponent(self, opponent: Union[Player, str]):
        """
        Sets the next opponent to the specified opponent.

        :param opponent: The next opponent to challenge
        :type opponent: Player or str
        """
        with self._opponent_lock:
            self._opponent = opponent

    def reset_env(
        self, opponent: Optional[Union[Player, str]] = None, restart: bool = True
    ):
        """
        Resets the environment to an inactive state: it will forfeit all unfinished
        battles, reset the internal battle tracker and optionally change the next
        opponent and restart the challenge loop.

        :param opponent: The opponent to use for the next battles. If empty it
            will not change opponent.
        :type opponent: Player or str, optional
        :param restart: If True the challenge loop will be restarted before returning,
            otherwise the challenge loop will be left inactive and can be
            started manually.
        :type restart: bool
        """
        self.close(purge=False)
        self.reset_battles()
        if opponent:
            self.set_opponent(opponent)
        if restart:
            self.start_challenging()


class Gen9SimplePlayer(EnvPlayer[ObsType, ActType], ABC):
    _ACTION_SPACE = list(range(9))  # 0~8
    _DEFAULT_BATTLE_FORMAT = "gen9randombattle"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.miss_flag = False  # 新增，记录本步动作是否错选

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        self.miss_flag = False  # 每次动作转换前先重置

        if action == -1:
            return ForfeitBattleOrder()

        # 0~3: 使用第 0~3 个普通招式（仅当不是强制换人）
        if 0 <= action <= 3 and not battle.force_switch:
            if action < len(battle.available_moves):
                return self.agent.create_order(battle.available_moves[action])
            else:
                self.miss_flag = True  # 动作非法，超出可用招式范围

        # 4~8: 尝试切换到第 (action - 4) 个可用宝可梦
        if 4 <= action <= 8:
            switch_index = action - 4
            if switch_index < len(battle.available_switches):
                return self.agent.create_order(battle.available_switches[switch_index])
            else:
                self.miss_flag = True  # 动作非法，超出可换宝可梦范围

        # 动作不合法或没匹配上的情况，修正成随机合法动作
        
        if battle.force_switch and battle.available_switches:
            return self.agent.create_order(battle.available_switches[-1])
        if battle.available_moves:
            return self.agent.create_order(battle.available_moves[-1])
        if battle.available_switches:
            return self.agent.create_order(battle.available_switches[-1])
        return self.agent.choose_random_move(battle) 
            
