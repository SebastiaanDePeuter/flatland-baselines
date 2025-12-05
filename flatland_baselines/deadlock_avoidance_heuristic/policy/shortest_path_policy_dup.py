from typing import List, Dict, Tuple

from flatland.envs.agent_chains import AgentHandle
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState


# TODO https://github.com/flatland-association/flatland-baselines/issues/24 backport to flatland-rl after refactorings. We need to re-generate the regression trajectories with `get_k_shortest_paths` instead of custom `ShortestDistanceWalker`. For now use `ShortestDistanceWalker` as regression tests are based on the shortest paths produced by this method.
class DupShortestPathPolicy(RailEnvPolicy[RailEnv, RailEnv, RailEnvActions]):
    """
    Works with `FullEnvObservation`
    """

    def __init__(self, _get_k_shortest_paths=None):
        super().__init__()
        self._shortest_paths: Dict[AgentHandle, Tuple[Waypoint]] = {}
        self._remaining_targets: Dict[AgentHandle, List[List[Waypoint]]] = {}
        if _get_k_shortest_paths is None:
            self.get_k_shortest_paths = lambda *args: get_k_shortest_paths(*args[1:])
        else:
            self.get_k_shortest_paths = _get_k_shortest_paths

    def _act(self, env: RailEnv, agent: EnvAgent):
        if agent.position is None:
            return RailEnvActions.MOVE_FORWARD

        if len(self._remaining_targets[agent.handle]) == 0:
            return RailEnvActions.DO_NOTHING

        for a in {RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT}:
            new_cell_valid, (new_position, new_direction), transition_valid, preprocessed_action = env.rail.check_action_on_agent(
                RailEnvActions.from_value(a), (agent.position, agent.direction)
            )
            if new_cell_valid and transition_valid and (
                    new_position == self._remaining_targets[agent.handle][0] or (
                    new_position == self._shortest_paths[agent.handle][1].position and new_direction == self._shortest_paths[agent.handle][1].direction)):
                return a
        raise Exception("Invalid state")

    def act_many(self, handles: List[int], observations: List[RailEnv], **kwargs):
        actions = {}
        for handle, env in zip(handles, observations):
            agent = env.agents[handle]
            self._update_agent(agent, env)
            actions[handle] = self._act(env, agent)
        return actions

    def _update_agent(self, agent: EnvAgent, env: RailEnv):
        """
        Update `_shortest_paths` and `_remaining_targets`.
        Returns True if a new path was planned for this agent
        """
        replanned = False
        if agent.state == TrainState.DONE:
            self._shortest_paths.pop(agent.handle, None)
            self._remaining_targets.pop(agent.handle, None)
            return False
        if agent.handle not in self._remaining_targets:
            self._remaining_targets[agent.handle] = agent.waypoints
            assert len(agent.waypoints) == 2, "implementation does not support intermediate waypoints"
        if agent.handle not in self._shortest_paths:
            # TODO https://github.com/flatland-association/flatland-baselines/issues/7 inconsistent: shortest path is shortest path to target, whereas above we update when intermediate target reached....?
            self._shortest_paths[agent.handle] = self.get_k_shortest_paths(agent.handle, env, agent.initial_position, agent.initial_direction, agent.target)[0]

        if agent.position is None:
            return False

        # This handles two cases where re-planning is necessary:
        # (1) The agent has gone off-path. This happens is it has moved in the previous step (first two conditions) but has not reached the next point along it shortst path.
        # (2) The agent't current plan has fewer than 2 steps left. This generally happens if the pathfinding previously did not find a complete path to the target.
        # In both cases we need to re-plan
        if  len(self._shortest_paths[agent.handle]) < 2:
            self._shortest_paths[agent.handle] = self.get_k_shortest_paths(agent.handle, env, agent.position, agent.direction, agent.target)[0]
            replanned = True
        elif (agent.old_position is not None) and (agent.position != agent.old_position) and (agent.position != self._shortest_paths[agent.handle][1].position):
            self._shortest_paths[agent.handle] = self.get_k_shortest_paths(agent.handle, env, agent.position, agent.direction, agent.target)[0]
            replanned = True

        while self._shortest_paths[agent.handle][0].position != agent.position:
            self._shortest_paths[agent.handle] = self._shortest_paths[agent.handle][1:]
        assert self._shortest_paths[agent.handle][0].position == agent.position

        return replanned

