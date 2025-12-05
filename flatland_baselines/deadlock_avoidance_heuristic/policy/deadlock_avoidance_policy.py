from collections import defaultdict
from functools import lru_cache
from typing import List, Any, Dict, Set, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from flatland.core.env_observation_builder import AgentHandle
from flatland.envs.fast_methods import fast_count_nonzero
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState
from flatland_baselines.deadlock_avoidance_heuristic.policy.shortest_path_policy_dup import DupShortestPathPolicy
from flatland_baselines.deadlock_avoidance_heuristic.utils.flatland.shortest_distance_walker import ExtendedShortestDistanceWalker

# activate LRU caching
flatland_deadlock_avoidance_policy_lru_cache_functions = []


def _enable_flatland_deadlock_avoidance_policy_lru_cache(*args, **kwargs):
    def decorator(func):
        func = lru_cache(*args, **kwargs)(func)
        flatland_deadlock_avoidance_policy_lru_cache_functions.append(func)
        return func

    return decorator


def _send_flatland_deadlock_avoidance_policy_data_change_signal_to_reset_lru_cache():
    for func in flatland_deadlock_avoidance_policy_lru_cache_functions:
        func.cache_clear()


class DeadLockAvoidancePolicy(DupShortestPathPolicy):
    def __init__(self,
                 action_size: int = 5,
                 min_free_cell: int = 1,
                 enable_eps: bool = False,
                 show_debug_plot: bool = False,
                 ):
        super().__init__()
        
        self.loss = 0
        self.action_size = action_size
        self.show_debug_plot = show_debug_plot
        self.enable_eps = enable_eps
        self.min_free_cell = min_free_cell

        # will be injected from observation (`FullEnvObservation`)
        self.raiL_env: Optional[RailEnv] = None

        # -1 if no agent, agent handle otherwise
        self.agent_positions = None

        # next (r,c,d) and action to get there; or no entry if train must not move
        self.agent_can_move: Dict[AgentHandle, Tuple[int, int, int, RailEnvActions]] = {}
        # set of oncoming agents
        self.opp_agent_map: Dict[AgentHandle, Set[AgentHandle]] = defaultdict(set)

    def _init_env(self, env: RailEnv):
        distance_walker = ExtendedShortestDistanceWalker(env)

        def get_k_shortest_paths(handle, env,
                                 source_position: Tuple[int, int],
                                 source_direction: int,
                                 target_position=Tuple[int, int],
                                 k: int = 1, debug=False) -> List[Tuple[Waypoint]]:
            return [distance_walker.walk_to_target2(handle, source_position, source_direction, target_position, max_step=5000)]

        super(DeadLockAvoidancePolicy, self).__init__(_get_k_shortest_paths=get_k_shortest_paths)

        # 1 if position is a switch, 0 otherwise
        self.switches = np.zeros((self.raiL_env.height, self.raiL_env.width), dtype=int)
        for r in range(self.raiL_env.height):
            for c in range(self.raiL_env.width):
                if self._is_switch_cell((r, c)):
                    self.switches[(r, c)] = 1

        # 1 if current shortest path (without current cell!), 0 otherwise
        self.shortest_distance_agent_map = np.zeros((self.raiL_env.get_num_agents(),
                                                     self.raiL_env.height,
                                                     self.raiL_env.width),
                                                    dtype=int)
        # all positions on current shortest path (without current cell!)
        self.shortest_distance_positions_agent_map: Dict[AgentHandle, Set[Tuple[int, int]]] = defaultdict(set)
        # directions for all positions on current shortest path (without current cell!)
        self.shortest_distance_positions_directions_agent_map: Dict[Tuple[AgentHandle, Tuple[int, int]], Set[int]] = defaultdict(set)
        # number of cells till first oncoming agent (without current cell!)
        self.shortest_distance_agent_len: Dict[AgentHandle, int] = defaultdict(lambda: 0)
        # 1 if current shortest path (without current cell!) before first oncoming train, 0 else.
        self.full_shortest_distance_agent_map = np.zeros((self.raiL_env.get_num_agents(),
                                                          self.raiL_env.height,
                                                          self.raiL_env.width),
                                                         dtype=int)

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        assert isinstance(observations[0], RailEnv)
        if self.raiL_env is None:
            self.raiL_env = observations[0]
            self._init_env(self.raiL_env)
        self._start_step()
        return {handle: self._act(handle, observations[handle]) for handle in handles}

    def _act(self, handle: int, state, eps=0.) -> RailEnvActions:
        # Epsilon-greedy action selection
        if self.enable_eps:
            if np.random.random() < eps:
                return np.random.choice(np.arange(self.action_size))

        check = self.agent_can_move.get(handle, None)
        act = RailEnvActions.STOP_MOVING
        if check is not None:
            act = check[3]
        # TODO port to client.py:  File "msgpack/_packer.pyx", line 257, in msgpack._cmsgpack.Packer._pack_inner
        # submission-1      | TypeError: can not serialize 'RailEnvActions' object
        # if isinstance(act, RailEnvActions):
        #    act = act.value
        return act

    def _start_step(self):
        # (1)
        self._build_agent_position_map()

        # (2)
        self._update_shortest_distance_maps_and_opp_agent_map()

        # (3)
        self._extract_agent_can_move()

    def _build_agent_position_map(self):
        """
        start_step (1): update agent positions at start of step.
        More precisely, updates:
        - `self.agent_positions`
        """
        # build map with agent positions (only active agents)
        self.agent_positions = np.zeros((self.raiL_env.height, self.raiL_env.width), dtype=int) - 1
        for handle in range(self.raiL_env.get_num_agents()):
            agent = self.raiL_env.agents[handle]
            if agent.state in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
                if agent.position is not None:
                    self.agent_positions[agent.position] = handle

    def _update_shortest_distance_maps_and_opp_agent_map(self):
        """
       start_step (2): update the shortest paths to current position and update opposing agent as well as bitmap representation at start of step.
        More precisely, updates:
        - `super()._shortest_paths`
        - `super()._remaining_targets`.
        - `self.opp_agent_map`
        - `self.shortest_distance_agent_map`
        - `self.full_shortest_distance_agent_map`
        """
        all_agent_positions: Set[Tuple[int, int]] = self._collect_all_agent_positions()

        for agent in self.raiL_env.agents:
            handle = agent.handle

            replanned = super()._update_agent(agent, self.raiL_env)

            self._build_full_shortest_distance_agent_map(agent, handle, replan = replanned)
            if agent.state == TrainState.DONE or agent.state == TrainState.WAITING:
                continue

            self._build_shortest_distance_agent_map(agent, handle, all_agent_positions)

    def _collect_all_agent_positions(self):
        all_agent_positions = set()
        for agent in self.raiL_env.agents:
            all_agent_positions.add(agent.position)
        return all_agent_positions
    
    def _agent_remove_planned_positions(self, handle):
        self.full_shortest_distance_agent_map[handle,:,:] = 0
        self.shortest_distance_positions_agent_map[handle] = set()
        for (handle_i, position_i) in list(self.shortest_distance_positions_directions_agent_map.keys()):
            if handle_i == handle:
                del self.shortest_distance_positions_directions_agent_map[(handle_i, position_i)]

    def _build_full_shortest_distance_agent_map(self, agent, handle, replan = False):
        if self.raiL_env._elapsed_steps == 1 or replan:
            if replan:
                self._agent_remove_planned_positions(handle)
            for wp in self._shortest_paths[agent.handle][1:]:
                position, direction = wp.position, wp.direction
                self.full_shortest_distance_agent_map[(handle, position[0], position[1])] = 1
                self.shortest_distance_positions_agent_map[handle].add(position)
                self.shortest_distance_positions_directions_agent_map[(handle, position)].add(direction)
        if agent.position is not None and agent.position != agent.old_position and not replan:
            assert agent.position == self._shortest_paths[agent.handle][0].position
            if agent.position not in {wp.position for wp in self._shortest_paths[agent.handle][1:]}:
                self.full_shortest_distance_agent_map[(handle, agent.position[0], agent.position[1])] = 0
                # the initial position is never added to shortest_distance_positions_agent_map
                if agent.old_position is not None:
                    self.shortest_distance_positions_agent_map[handle].remove(agent.position)
            # the initial position is never added to shortest_distance_positions_agent_map
            # N.B. We must remove each direction separately, as there can be "loopy" paths going through same cell twice but with different direction!
            if agent.old_position is not None:
                self.shortest_distance_positions_directions_agent_map[(handle, agent.position)].remove(int(agent.direction))

    def _build_shortest_distance_agent_map(self, agent, handle, all_agent_positions):
        prev_opp_agents = self.opp_agent_map[handle]
        overlap = self.shortest_distance_positions_agent_map[handle].intersection(all_agent_positions)
        if overlap == prev_opp_agents:
            return

        self._rebuild_opp_agent_map(handle, overlap)

        self._rebuild_shortest_distance_agent_map(agent, handle)

    def _rebuild_shortest_distance_agent_map(self, agent, handle):
        # TODO performance improvement idea: we could update over the previous opposing trains (beware of close-following, map must be non-binary) and update their position and we only have to look at non-facing switches where new opposing trains can occur.
        self.shortest_distance_agent_map[handle].fill(0)
        self.shortest_distance_agent_len[handle] = 0
        num_opp_agents = 0
        for wp in self._shortest_paths[agent.handle][1:]:
            position, direction = wp.position, wp.direction
            opp_a = self.agent_positions[position]
            if opp_a != -1 and opp_a != handle:
                if self.raiL_env.agents[opp_a].direction != direction:
                    num_opp_agents += 1
                    break
            if num_opp_agents == 0:
                self.shortest_distance_agent_len[handle] += 1
                self.shortest_distance_agent_map[(handle, position[0], position[1])] = 1

    def _rebuild_opp_agent_map(self, handle, overlap):
        self.opp_agent_map[handle] = set()
        for position in overlap:
            opp_a = self.agent_positions[position]
            if opp_a != -1 and opp_a != handle:
                directions = self.shortest_distance_positions_directions_agent_map[(handle, position)]
                assert len(directions) > 0
                for direction in directions:
                    if self.raiL_env.agents[opp_a].direction != direction:
                        self.opp_agent_map[handle].add(opp_a)

    def _get_action(self, configuration: Tuple[Tuple[int, int], int], next_configuration: Tuple[Tuple[int, int], int]):
        for action in [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]:
            new_cell_valid, new_configuration, transition_valid, preprocessed_action = self.raiL_env.rail.check_action_on_agent(action, configuration)
            if new_configuration == next_configuration:
                return preprocessed_action
        raise

    def _extract_agent_can_move(self):
        """
        start_step (3): update whether agent can move. More precisely, updates:
        - `self.agent_can_move`
        """
        self.agent_can_move = {}

        for handle in range(self.raiL_env.get_num_agents()):
            agent = self.raiL_env.agents[handle]
            if TrainState.DONE > agent.state >= TrainState.WAITING and len(self._shortest_paths[agent.handle]) > 1:
                if self._check_agent_can_move(
                        self.shortest_distance_agent_map[handle],
                        self.shortest_distance_agent_len[handle],
                        self.opp_agent_map.get(handle, set()),
                        self.full_shortest_distance_agent_map,
                        self.switches,
                        True
                ):
                    if agent.position is not None:
                        position = agent.position
                        direction = agent.direction
                    else:
                        position = agent.initial_position
                        direction = agent.initial_direction

                    next_position = self._shortest_paths[agent.handle][1].position
                    next_direction = self._shortest_paths[agent.handle][1].direction
                    action = self._get_action((position, direction), (next_position, next_direction))

                    self.agent_can_move.update({handle: [next_position[0], next_position[1], next_direction, action]})

        if self.show_debug_plot:
            a = np.floor(np.sqrt(self.raiL_env.get_num_agents()))
            b = np.ceil(self.raiL_env.get_num_agents() / a)
            for handle in range(self.raiL_env.get_num_agents()):
                plt.subplot(a, b, handle + 1)
                plt.imshow(self.full_shortest_distance_agent_map[handle] + self.shortest_distance_agent_map[handle])
            plt.show(block=False)
            plt.pause(0.01)

    def _check_agent_can_move(
            self,
            my_shortest_walking_path: np.ndarray,
            my_shortest_walking_path_len: int,
            opp_agents: Set,
            full_shortest_distance_agent_map: np.ndarray,
            switches: Optional[np.ndarray] = None,
            count_num_opp_agents_towards_min_free_cell: bool = False
    ):
        """
        The algorithm collects for each train along its route all trains that are currently on a resource in the route.
        For each collected train (`opp_agents`), the method has to decide at which position along the route the train
        must let pass the collected opposing train:  by searching the train's path required resources backward along the path
        starting at the collected train position; stop the search when the resource along the collected train's path is not equal.
        This yields `free_cells` ahead of the agent without overlap with any opposing agent's travelling path.
        If `free_cells >= min_free_cells >= 1` for all opposing agents, then the agent can move.
        A deadlock can only occur if a jam "fills in" the free space and is not detected by the algorithm.

        To determine `free_cells`, the implementation compares takes the difference of
        - the bitmap of the agent's shortest path (up to first opposing agent) and
        - the bitmap opposing agent's path
        and counts the positive elements.

        The forward and backward traveling along the train and the collected train path must be done step-by-step synchronous.
        If the first non-equal resource position along the train's path is more than one resource from train's current location away,
        then the train can move and no deadlock will occur for the next time step.

        2 heuristics to avoid "fill-in":
        - switches: if switches is given, then switches do not count towards free cells
        - count_num_opp_agents_towards_min_free_cell: the number of opposing agents is added to `min_free_cell`

        Parameters
        ----------
        my_shortest_walking_path : np.ndarray
            the shortest path up to first opposing agent
        my_shortest_walking_path_len: int
            length of shortest path up to first opposing agent
        opp_agents : np.ndarray
            positions of opposing agents
        full_shortest_distance_agent_map : np.ndarray
            full shortest path
        switches: Optional[np.ndarray] = None
            positions of switches. If None, disable the switches heuristics.
        count_num_opp_agents_towards_min_free_cell: bool = False
            the number of opposing agents is added to `min_free_cell`.
        """
        len_opp_agents = len(opp_agents)
        if len_opp_agents == 0:
            return True

        if my_shortest_walking_path_len < self.min_free_cell - len_opp_agents:
            return False
        min_free_cell = self.min_free_cell
        if count_num_opp_agents_towards_min_free_cell:
            min_free_cell += len_opp_agents

        for opp_a in opp_agents:
            opp = full_shortest_distance_agent_map[opp_a]
            if switches is None:
                free_cells = np.count_nonzero((my_shortest_walking_path - opp) > 0)
            else:
                free_cells = np.count_nonzero((my_shortest_walking_path - switches - opp) > 0)

            if free_cells < min_free_cell:
                return False
        return True

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    @_enable_flatland_deadlock_avoidance_policy_lru_cache(maxsize=100000)
    def _is_switch_cell(self, position) -> bool:
        for new_dir in range(4):
            possible_transitions = self.raiL_env.rail.get_transitions((position, new_dir))
            num_transitions = fast_count_nonzero(possible_transitions)
            if num_transitions > 1:
                return True
        return False
