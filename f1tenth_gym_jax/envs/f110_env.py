"""JAX-compatible f1tenth_gym_jax environment."""

# other
from collections.abc import Callable
from functools import partial
from numbers import Integral, Real

# typing
from typing import Dict, NamedTuple, Optional, Tuple

import chex

# jax
import jax
import jax.numpy as jnp

# numpy scipy
import numpy as np

# scanning
from jax_pf.ray_marching import get_scan
from scipy.ndimage import distance_transform_edt as edt

# collisions
from .collision_models import collision, collision_map, get_vertices

# dynamics
from .dynamic_models import (
    vehicle_dynamics_ks,
    vehicle_dynamics_st_smooth,
    vehicle_dynamics_st_switching,
)

# integrators
from .integrator import integrate_euler, integrate_rk4
from .multi_agent_env import MultiAgentEnv
from .spaces import Box

# track
from .track import Track

# dataclasses
from .utils import VALID_REWARDS, Param, State


ScanHook = Callable[[chex.PRNGKey, State, chex.Array], chex.Array]


class ArrayStepResult(NamedTuple):
    """Loop-free all-agent transition returned by :meth:`step_env_array`.

    Array shapes are ``observations=[agents, observation_dim]``,
    ``rewards=[agents]``, and ``terminated/truncated=[agents]``.  ``state`` is
    the terminal next state for ``step_env_array`` and the collector state for
    ``step_array`` after its optional auto-reset.
    """

    observations: chex.Array
    state: State
    rewards: chex.Array
    terminated: chex.Array
    truncated: chex.Array
    infos: dict


def combine_scan_ranges(
    static_ranges: chex.Array,
    external_ranges: chex.Array,
    max_range: float,
) -> chex.Array:
    """Combine equally shaped static ranges and external hits.

    Valid returns are finite and non-negative. An external value equal to
    ``max_range`` is a ray-caster miss sentinel, while a static value at that
    bound remains a valid map return. A valid external hit can therefore
    replace an invalid static value. If neither source is valid, the original
    static value is retained so its invalid-value semantics are not erased.
    """

    static_scan = jnp.asarray(static_ranges)
    external_scan = jnp.asarray(external_ranges, dtype=static_scan.dtype)
    if static_scan.shape != external_scan.shape:
        raise ValueError("external scan ranges must match static scan shape")
    maximum = jnp.asarray(max_range, dtype=static_scan.dtype)
    static_is_valid = (
        jnp.isfinite(static_scan)
        & (static_scan >= 0.0)
        & (static_scan <= maximum)
    )
    external_is_hit = (
        jnp.isfinite(external_scan)
        & (external_scan >= 0.0)
        & (external_scan < maximum)
    )
    static_candidate = jnp.where(static_is_valid, static_scan, maximum)
    external_candidate = jnp.where(external_is_hit, external_scan, maximum)
    nearest = jnp.minimum(static_candidate, external_candidate)
    return jnp.where(static_is_valid | external_is_hit, nearest, static_scan)


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_positive_number(name: str, value: float) -> None:
    if not isinstance(value, Real) or value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_ordered_bounds(name: str, lower: float, upper: float) -> None:
    if not isinstance(lower, Real) or not isinstance(upper, Real) or lower >= upper:
        raise ValueError(f"{name} lower bound must be less than upper bound.")


class F110Env(MultiAgentEnv):
    """
    JAX-compatible multi-agent environment for F1TENTH.

    Parameters
    ----------
    num_agents : int, default=1
        Number of agents in the environment.
    params : Param, default=Param()
        Vehicle, map, reward, control, and simulation parameters.
    external_scan_hook : callable, optional
        Static JAX callable ``(key, state, current_ranges) -> external_ranges``.
        Its result is combined with the static map scan by elementwise minimum.
    scan_corruption_hook : callable, optional
        Static JAX callable ``(key, state, ranges) -> corrupted_ranges`` run
        after external ranges have been combined.
    scan_only_observation : bool, default=False
        Expose only ``[beams]`` scan observations while retaining the full
        simulator state internally.

    """

    def __init__(
        self,
        num_agents: int = 1,
        params: Param = Param(),
        external_scan_hook: Optional[ScanHook] = None,
        scan_corruption_hook: Optional[ScanHook] = None,
        scan_only_observation: bool = False,
        **kwargs,
    ):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unsupported F110Env constructor keyword argument(s): {unknown}. "
                "Use f1tenth_gym_jax.make(..., **overrides) for parameter overrides."
            )
        _validate_positive_int("number of agents", num_agents)
        _validate_positive_int("timestep ratio", params.timestep_ratio)
        _validate_positive_int("max steps", params.max_steps)
        _validate_positive_int("max number of laps", params.max_num_laps)
        _validate_positive_int("theta discretization", params.theta_dis)
        _validate_positive_int("number of scan beams", params.num_beams)
        if params.num_beams < 2:
            raise ValueError("number of scan beams must be at least 2.")
        _validate_positive_number("surface friction coefficient", params.mu)
        _validate_positive_number("front cornering stiffness", params.C_Sf)
        _validate_positive_number("rear cornering stiffness", params.C_Sr)
        _validate_positive_number("front axle distance", params.lf)
        _validate_positive_number("rear axle distance", params.lr)
        _validate_positive_number("center of gravity height", params.h)
        _validate_positive_number("vehicle mass", params.m)
        _validate_positive_number("vehicle inertia", params.I)
        _validate_positive_number("switching velocity", params.v_switch)
        _validate_positive_number("maximum acceleration", params.a_max)
        _validate_positive_number("vehicle width", params.width)
        _validate_positive_number("vehicle length", params.length)
        _validate_positive_number("timestep", params.timestep)
        _validate_positive_number("field of view", params.fov)
        _validate_positive_number("scan epsilon", params.eps)
        _validate_positive_number("max scan range", params.max_range)
        _validate_ordered_bounds("steering angle", params.s_min, params.s_max)
        _validate_ordered_bounds("steering velocity", params.sv_min, params.sv_max)
        _validate_ordered_bounds("velocity", params.v_min, params.v_max)
        if external_scan_hook is not None and not callable(external_scan_hook):
            raise TypeError("external_scan_hook must be callable or None")
        if scan_corruption_hook is not None and not callable(scan_corruption_hook):
            raise TypeError("scan_corruption_hook must be callable or None")
        if not isinstance(scan_only_observation, bool):
            raise TypeError("scan_only_observation must be a bool")
        if scan_only_observation and not params.produce_scans:
            raise ValueError("scan_only_observation requires produce_scans=True")

        super().__init__(num_agents=num_agents)
        self.params = params
        self.external_scan_hook = external_scan_hook
        self.scan_corruption_hook = scan_corruption_hook
        self.scan_only_observation = scan_only_observation
        self.reward_types = frozenset(params.reward_type.split("+"))
        if not self.reward_types or not self.reward_types.issubset(VALID_REWARDS):
            raise ValueError(
                f"Invalid reward type list: {self.reward_types}, "
                f"must be from {sorted(VALID_REWARDS)}."
            )
        # agents
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}

        # choose dynamics model and integrators
        if params.integrator == "rk4":
            self.integrator_func = integrate_rk4
        elif params.integrator == "euler":
            self.integrator_func = integrate_euler
        else:
            raise ValueError(
                f"Chosen integrator {params.integrator} is invalid. "
                "Choose either 'rk4' or 'euler'."
            )

        if params.model == "st":
            self.model_func = vehicle_dynamics_st_switching
            self.state_size = 7
            self.cartesian_obs_indices = (2, 3, 5, 6)
        elif params.model == "st_smooth":
            self.model_func = vehicle_dynamics_st_smooth
            self.state_size = 7
            self.cartesian_obs_indices = (2, 3, 5, 6)
        elif params.model == "ks":
            self.model_func = vehicle_dynamics_ks
            self.state_size = 5
            self.cartesian_obs_indices = (2, 3)
        else:
            raise ValueError(
                f"Chosen dynamics model {params.model} is invalid. "
                "Choose either 'st', 'st_smooth', or 'ks'."
            )

        if params.steering_action_type == "steeringvelocity":
            steering_bounds = (params.sv_min, params.sv_max)
        elif params.steering_action_type == "steeringangle":
            steering_bounds = (params.s_min, params.s_max)
        else:
            raise ValueError(
                f"Chosen steering action type {params.steering_action_type} is invalid. "
                "Choose either 'steeringvelocity' or 'steeringangle'."
            )

        if params.longitudinal_action_type == "acceleration":
            longitudinal_bounds = (-params.a_max, params.a_max)
        elif params.longitudinal_action_type == "velocity":
            longitudinal_bounds = (params.v_min, params.v_max)
        else:
            raise ValueError(
                f"Chosen longitudinal action type {params.longitudinal_action_type} is invalid. "
                "Choose either 'acceleration' or 'velocity'."
            )

        action_low = jnp.array([steering_bounds[0], longitudinal_bounds[0]])
        action_high = jnp.array([steering_bounds[1], longitudinal_bounds[1]])
        self.action_spaces = {
            i: Box(action_low, action_high, (2,)) for i in self.agents
        }

        # scanning or not
        if params.produce_scans:
            self.scan_size = params.num_beams
        else:
            self.scan_size = 0

        # observing others
        if params.observe_others and not scan_only_observation:
            # (relative_x, relative_y, relative_psi, longitudinal_v)
            self.all_other_state_size = 4 * (self.num_agents - 1)
        else:
            self.all_other_state_size = 0

        observation_size = (
            self.scan_size
            if scan_only_observation
            else self.state_size + self.all_other_state_size + self.scan_size
        )
        self.observation_spaces = {
            i: Box(
                -jnp.inf,
                jnp.inf,
                (observation_size,),
            )
            for i in self.agents
        }
        if scan_only_observation:
            self.observation_space_ind = {
                "dynamics_state": [],
                "other_agent_dynamics_state": [],
                "scan": list(range(self.scan_size)),
            }
        else:
            self.observation_space_ind = {
                "dynamics_state": list(range(self.state_size)),
                "other_agent_dynamics_state": list(
                    range(self.state_size, self.state_size + self.all_other_state_size)
                ),
                "scan": list(
                    range(
                        self.state_size + self.all_other_state_size,
                        self.state_size + self.all_other_state_size + self.scan_size,
                    )
                ),
            }

        # load map
        self.track = Track.from_track_name(params.map_name)
        self.track_length = jnp.max(self.track.centerline.s)

        # get a interior point of track as winding number looking point
        start_point_curvature = self.track.centerline.calc_curvature(0.0)
        self.winding_point = jnp.array(
            self.track.frenet_to_cartesian(
                s=0.0, ey=np.sign(start_point_curvature) * 1.5, ephi=0.0
            )
        )[:2]
        # get racing direction
        fp = jnp.array([self.track.raceline.xs[0], self.track.raceline.ys[0]])
        sp = jnp.array([self.track.raceline.xs[1], self.track.raceline.ys[1]])
        fp_winding_vec = fp - self.winding_point
        sp_winding_vec = sp - self.winding_point
        self.winding_direction = jnp.sign(
            jnp.arctan2(
                jnp.cross(fp_winding_vec, sp_winding_vec),
                jnp.dot(fp_winding_vec, sp_winding_vec),
            )
        )

        # set pixel centers of occupancy map
        self._set_pixelcenters()

        # scan params if produce scan
        # if self.params.produce_scans:
        self.fov = self.params.fov
        self.num_beams = self.params.num_beams
        self.theta_dis = self.params.theta_dis
        self.eps = self.params.eps
        self.max_range = self.params.max_range

        angle_increment = self.fov / (self.num_beams - 1)
        self.theta_index_increment = self.theta_dis * angle_increment / (2 * np.pi)
        theta_arr = jnp.linspace(0.0, 2 * jnp.pi, num=self.theta_dis)
        self.scan_sines = jnp.sin(theta_arr)
        self.scan_cosines = jnp.cos(theta_arr)

        self.distance_transform = edt(self.track.occ_map) * self.track.resolution
        self.height, self.width = self.track.occ_map.shape
        self.resolution = self.track.resolution
        self.orig_x = self.track.ox
        self.orig_y = self.track.oy
        self.orig_c = jnp.cos(self.track.oyaw)
        self.orig_s = jnp.sin(self.track.oyaw)

    def _set_pixelcenters(self):
        map_img = self.track.occ_map
        h, w = map_img.shape
        reso = self.track.resolution
        ox = self.track.ox
        oy = self.track.oy
        x_ind, y_ind = np.meshgrid(range(w), range(h))
        pcx = (x_ind * reso + ox + reso / 2).flatten()
        pcy = (y_ind * reso + oy + reso / 2).flatten()
        self.pixel_centers = np.vstack((pcx, pcy)).T
        map_mask = (map_img == 0.0).flatten()
        self.pixel_centers = self.pixel_centers[map_mask, :]

    def _agent_array_to_dict(self, values: chex.Array) -> Dict[str, chex.Array]:
        """Adapt an all-agent array to the legacy dictionary API."""

        if values.shape[0] != self.num_agents:
            raise ValueError("all-agent array must have leading shape [num_agents]")
        return {agent: values[index] for index, agent in enumerate(self.agents)}

    @partial(
        jax.jit,
        static_argnums=(0,),
        static_argnames=("external_scan_hook", "scan_corruption_hook"),
    )
    def step_env_array(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: chex.Array,
        external_scan_ranges: Optional[chex.Array] = None,
        scan_corruption_ranges: Optional[chex.Array] = None,
        *,
        external_scan_hook: Optional[ScanHook] = None,
        scan_corruption_hook: Optional[ScanHook] = None,
    ) -> ArrayStepResult:
        """Step all agents from one ``[agents, 2]`` action array.

        This is the loop-free public transition path.  Optional scan arrays
        must have shape ``[agents, beams]``.  Hook callables are static JAX
        arguments with signature ``hook(key, state, current_ranges)``.
        """

        action_array = jnp.asarray(actions)
        if action_array.shape != (self.num_agents, 2):
            raise ValueError("actions must have shape [num_agents, 2]")

        # 1. state + scan
        x = state.cartesian_states
        us = action_array
        # stop collided cars
        us = jnp.where(state.collisions[:, None], jnp.zeros_like(us), us)
        x_and_u = jnp.hstack((x, us))
        # integrate dynamics, vmapped
        integrator = jax.vmap(self.integrator_func, in_axes=[None, 0, None])
        new_x_and_u = integrator(self.model_func, x_and_u, self.params)
        final_x_and_u = jnp.where(state.collisions[:, None], x_and_u, new_x_and_u)
        state = state.replace(
            last_cartesian_states=state.cartesian_states,
            cartesian_states=final_x_and_u[:, :-2],
            last_frenet_states=state.frenet_states,
            frenet_states=self.track.vmap_cartesian_to_frenet_jax(
                final_x_and_u[:, [0, 1, 4]]
            ),
            step=state.step + 1,
        )
        if self.params.produce_scans:
            state = self.scan_with_hooks(
                key,
                state,
                external_scan_ranges,
                scan_corruption_ranges,
                external_scan_hook=external_scan_hook,
                scan_corruption_hook=scan_corruption_hook,
            )

        # 2. collisions
        state = jax.lax.cond(
            self.params.collision_on, self._collisions, self._ret_orig_state, state
        )

        # 3. dones
        terminated, truncated, state = self.check_done_array(state)

        # 4. rewards
        rewards = self.get_reward_array(state)

        # 5. info
        infos = {}

        return ArrayStepResult(
            observations=self.get_obs_array(state),
            state=state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
        )

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Legacy dictionary transition API retained for compatibility."""

        action_array = jnp.stack(tuple(actions[agent] for agent in self.agents))
        result = self.step_env_array(key, state, action_array)
        done_array = result.terminated | result.truncated
        dones = self._agent_array_to_dict(done_array)
        dones["__all__"] = jnp.all(done_array)
        return (
            self._agent_array_to_dict(result.observations),
            result.state,
            self._agent_array_to_dict(result.rewards),
            dones,
            result.infos,
        )

    @partial(
        jax.jit,
        static_argnums=(0,),
        static_argnames=("external_scan_hook", "scan_corruption_hook"),
    )
    def step_array(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: chex.Array,
        reset_state: Optional[State] = None,
        external_scan_ranges: Optional[chex.Array] = None,
        scan_corruption_ranges: Optional[chex.Array] = None,
        *,
        external_scan_hook: Optional[ScanHook] = None,
        scan_corruption_hook: Optional[ScanHook] = None,
    ) -> ArrayStepResult:
        """Step an all-agent action array and auto-reset when all agents finish.

        ``actions`` has shape ``[agents, 2]``.  The returned termination flags
        describe the terminal transition even when ``state`` and
        ``observations`` have been replaced by the collector's reset state.
        Per-call scan arrays apply to the transition only; constructor hooks
        also run during a random reset.
        """

        step_key, reset_key = jax.random.split(key)
        result = self.step_env_array(
            step_key,
            state,
            actions,
            external_scan_ranges,
            scan_corruption_ranges,
            external_scan_hook=external_scan_hook,
            scan_corruption_hook=scan_corruption_hook,
        )

        if reset_state is None:
            reset_observations, reset_state = self.reset_array(reset_key)
        else:
            reset_observations = self.get_obs_array(reset_state)

        reset_all = jnp.all(result.terminated | result.truncated)
        collector_state = jax.tree.map(
            lambda reset_value, next_value: jax.lax.select(
                reset_all, reset_value, next_value
            ),
            reset_state,
            result.state,
        )
        collector_observations = jax.lax.select(
            reset_all,
            reset_observations,
            result.observations,
        )
        return ArrayStepResult(
            observations=collector_observations,
            state=collector_state,
            rewards=result.rewards,
            terminated=result.terminated,
            truncated=result.truncated,
            infos=result.infos,
        )

    def _initial_state(
        self,
        cartesian_states: chex.Array,
        frenet_states: chex.Array,
    ) -> State:
        """Build a zero-counter state around externally selected poses."""

        expected_cartesian_shape = (self.num_agents, self.state_size)
        if cartesian_states.shape != expected_cartesian_shape:
            raise ValueError(
                "cartesian_states must have shape "
                f"{expected_cartesian_shape}"
            )
        if frenet_states.shape != (self.num_agents, 3):
            raise ValueError("frenet_states must have shape [num_agents, 3]")
        dtype = cartesian_states.dtype
        return State(
            rewards=jnp.zeros((self.num_agents,), dtype=dtype),
            done=jnp.zeros((self.num_agents,), dtype=bool),
            step=jnp.asarray(0, dtype=jnp.int32),
            cartesian_states=cartesian_states,
            last_cartesian_states=cartesian_states,
            frenet_states=frenet_states,
            last_frenet_states=frenet_states,
            num_laps=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            collisions=jnp.zeros((self.num_agents,), dtype=bool),
            scans=jnp.zeros((self.num_agents, self.num_beams), dtype=dtype),
            prev_winding_vector=cartesian_states[:, 0:2] - self.winding_point,
            accumulated_angles=jnp.zeros((self.num_agents,), dtype=dtype),
            last_accumulated_angles=jnp.zeros((self.num_agents,), dtype=dtype),
        )

    def _state_from_cartesian_poses(self, cartesian_poses: chex.Array) -> State:
        pose_dtype = jnp.result_type(cartesian_poses, jnp.float32)
        poses = jnp.asarray(cartesian_poses, dtype=pose_dtype)
        if poses.shape != (self.num_agents, 3):
            raise ValueError("cartesian_poses must have shape [num_agents, 3]")
        cartesian_states = jnp.zeros(
            (self.num_agents, self.state_size),
            dtype=poses.dtype,
        )
        cartesian_states = cartesian_states.at[:, [0, 1, 4]].set(poses)
        return self._state_from_cartesian_states(cartesian_states)

    def _state_from_cartesian_states(self, cartesian_states: chex.Array) -> State:
        state_dtype = jnp.result_type(cartesian_states, jnp.float32)
        cartesian_states = jnp.asarray(cartesian_states, dtype=state_dtype)
        if cartesian_states.shape != (self.num_agents, self.state_size):
            raise ValueError(
                "cartesian_states must have shape "
                f"[{self.num_agents}, {self.state_size}]"
            )
        poses = cartesian_states[:, [0, 1, 4]]
        frenet_states = self.track.vmap_cartesian_to_frenet_jax(poses)
        return self._initial_state(cartesian_states, frenet_states)

    def _state_from_frenet_poses(self, frenet_poses: chex.Array) -> State:
        pose_dtype = jnp.result_type(frenet_poses, jnp.float32)
        poses = jnp.asarray(frenet_poses, dtype=pose_dtype)
        if poses.shape != (self.num_agents, 3):
            raise ValueError("frenet_poses must have shape [num_agents, 3]")
        poses = poses.at[:, 0].set(jnp.mod(poses[:, 0], self.track.s_frame_max))
        cartesian_poses = self.track.vmap_frenet_to_cartesian_jax(poses)
        cartesian_states = jnp.zeros(
            (self.num_agents, self.state_size),
            dtype=poses.dtype,
        )
        cartesian_states = cartesian_states.at[:, [0, 1, 4]].set(cartesian_poses)
        return self._initial_state(cartesian_states, poses)

    def _finalize_reset(
        self,
        key: chex.PRNGKey,
        state: State,
        external_scan_ranges: Optional[chex.Array] = None,
        scan_corruption_ranges: Optional[chex.Array] = None,
        *,
        external_scan_hook: Optional[ScanHook] = None,
        scan_corruption_hook: Optional[ScanHook] = None,
    ) -> Tuple[chex.Array, State]:
        state = state.replace(
            prev_winding_vector=state.cartesian_states[:, 0:2] - self.winding_point
        )
        if self.params.produce_scans:
            state = self.scan_with_hooks(
                key,
                state,
                external_scan_ranges,
                scan_corruption_ranges,
                external_scan_hook=external_scan_hook,
                scan_corruption_hook=scan_corruption_hook,
            )
        return self.get_obs_array(state), state

    @partial(jax.jit, static_argnums=(0,))
    def reset_array(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Reset randomly and return ``[agents, observation_dim]``."""

        # Keep the legacy placement keys stable; scan-only randomness uses a
        # folded-in key so adding scan hooks does not change reset poses.
        s_key, ey_key = jax.random.split(key)
        scan_key = jax.random.fold_in(key, 1)
        # randomly choose first agent location [0, 1] on entire arc length
        first_agent_s_loc = jax.random.uniform(s_key)
        first_agent_s = first_agent_s_loc * self.track.length
        first_agent_ey = jax.random.uniform(ey_key, minval=-0.3, maxval=0.3)
        # set up following agents in a grid pattern
        s_locs = jnp.linspace(
            first_agent_s,
            first_agent_s + 1.0 * (self.num_agents - 1),
            self.num_agents,
            endpoint=True,
        )
        ey_locs = first_agent_ey * jnp.where(
            jnp.arange(self.num_agents) % 2 == 0, 1.5, -1.5
        )
        ephi_locs = jnp.zeros((self.num_agents,))
        initial_states_frenet = jnp.column_stack((s_locs, ey_locs, ephi_locs))
        state = self._state_from_frenet_poses(initial_states_frenet)
        return self._finalize_reset(scan_key, state)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Legacy random reset returning agent-keyed observations."""

        observations, state = self.reset_array(key)
        return self._agent_array_to_dict(observations), state

    @partial(
        jax.jit,
        static_argnums=(0,),
        static_argnames=("external_scan_hook", "scan_corruption_hook"),
    )
    def reset_from_cartesian_poses(
        self,
        key: chex.PRNGKey,
        cartesian_poses: chex.Array,
        external_scan_ranges: Optional[chex.Array] = None,
        scan_corruption_ranges: Optional[chex.Array] = None,
        *,
        external_scan_hook: Optional[ScanHook] = None,
        scan_corruption_hook: Optional[ScanHook] = None,
    ) -> Tuple[chex.Array, State]:
        """Reset from external ``[agents, 3]`` x/y/yaw poses."""

        state = self._state_from_cartesian_poses(cartesian_poses)
        return self._finalize_reset(
            key,
            state,
            external_scan_ranges,
            scan_corruption_ranges,
            external_scan_hook=external_scan_hook,
            scan_corruption_hook=scan_corruption_hook,
        )

    @partial(
        jax.jit,
        static_argnums=(0,),
        static_argnames=("external_scan_hook", "scan_corruption_hook"),
    )
    def reset_from_frenet_poses(
        self,
        key: chex.PRNGKey,
        frenet_poses: chex.Array,
        external_scan_ranges: Optional[chex.Array] = None,
        scan_corruption_ranges: Optional[chex.Array] = None,
        *,
        external_scan_hook: Optional[ScanHook] = None,
        scan_corruption_hook: Optional[ScanHook] = None,
    ) -> Tuple[chex.Array, State]:
        """Reset from external ``[agents, 3]`` s/ey/epsi poses."""

        state = self._state_from_frenet_poses(frenet_poses)
        return self._finalize_reset(
            key,
            state,
            external_scan_ranges,
            scan_corruption_ranges,
            external_scan_hook=external_scan_hook,
            scan_corruption_hook=scan_corruption_hook,
        )

    @partial(
        jax.jit,
        static_argnums=(0,),
        static_argnames=("external_scan_hook", "scan_corruption_hook"),
    )
    def reset_from_cartesian_states(
        self,
        key: chex.PRNGKey,
        cartesian_states: chex.Array,
        external_scan_ranges: Optional[chex.Array] = None,
        scan_corruption_ranges: Optional[chex.Array] = None,
        *,
        external_scan_hook: Optional[ScanHook] = None,
        scan_corruption_hook: Optional[ScanHook] = None,
    ) -> Tuple[chex.Array, State]:
        """Reset from external ``[agents, state_dim]`` dynamic states."""

        state = self._state_from_cartesian_states(cartesian_states)
        return self._finalize_reset(
            key,
            state,
            external_scan_ranges,
            scan_corruption_ranges,
            external_scan_hook=external_scan_hook,
            scan_corruption_hook=scan_corruption_hook,
        )

    @partial(
        jax.jit,
        static_argnums=(0,),
        static_argnames=("external_scan_hook", "scan_corruption_hook"),
    )
    def reset_from_state(
        self,
        key: chex.PRNGKey,
        state: State,
        external_scan_ranges: Optional[chex.Array] = None,
        scan_corruption_ranges: Optional[chex.Array] = None,
        *,
        external_scan_hook: Optional[ScanHook] = None,
        scan_corruption_hook: Optional[ScanHook] = None,
    ) -> Tuple[chex.Array, State]:
        """Inject a complete external State and regenerate its observation."""

        if state.cartesian_states.shape != (self.num_agents, self.state_size):
            raise ValueError("external state has incompatible cartesian_states shape")
        if state.frenet_states.shape != (self.num_agents, 3):
            raise ValueError("external state has incompatible frenet_states shape")
        return self._finalize_reset(
            key,
            state,
            external_scan_ranges,
            scan_corruption_ranges,
            external_scan_hook=external_scan_hook,
            scan_corruption_hook=scan_corruption_hook,
        )

    @partial(jax.jit, static_argnums=[0])
    def get_obs_array(self, state: State) -> chex.Array:
        """Return observations as ``[agents, observation_dim]`` without loops."""

        if self.scan_only_observation:
            return state.scans

        cart_state = state.cartesian_states[:, self.cartesian_obs_indices]
        agent_state = jnp.concatenate((state.frenet_states, cart_state), axis=1)

        if self.params.observe_others:
            observer_indices = jnp.arange(self.num_agents)[:, None]
            compact_indices = jnp.arange(max(self.num_agents - 1, 0))[None, :]
            other_agent_indices = compact_indices + (
                compact_indices >= observer_indices
            )
            other_agent_poses = state.cartesian_states[
                other_agent_indices[..., None],
                jnp.array([0, 1, 4]),
            ]
            agent_poses = state.cartesian_states[:, None, jnp.array([0, 1, 4])]
            relative_poses = other_agent_poses - agent_poses
            relative_yaw = jnp.arctan2(
                jnp.sin(relative_poses[..., 2]),
                jnp.cos(relative_poses[..., 2]),
            )
            other_agent_velocities = state.cartesian_states[other_agent_indices, 3]
            relative_states = jnp.stack(
                (
                    relative_poses[..., 0],
                    relative_poses[..., 1],
                    other_agent_velocities,
                    relative_yaw,
                ),
                axis=-1,
            ).reshape((self.num_agents, -1))
        else:
            relative_states = jnp.empty(
                (self.num_agents, 0),
                dtype=agent_state.dtype,
            )

        observation_parts = (agent_state, relative_states)
        if self.params.produce_scans:
            observation_parts = (*observation_parts, state.scans)
        return jnp.concatenate(observation_parts, axis=1)

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Legacy agent-keyed observation API."""

        return self._agent_array_to_dict(self.get_obs_array(state))

    @partial(jax.jit, static_argnums=[0])
    def get_avail_actions(self, state: State) -> Dict[str, chex.Array]:
        """Returns the available action dimensions for each continuous-control agent."""
        return {
            agent: jnp.ones(self.action_spaces[agent].shape, dtype=bool)
            for agent in self.agents
        }

    @property
    def agent_classes(self) -> dict:
        """Returns homogeneous car agent classes for multi-agent consumers."""
        return {"car": list(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def check_done_array(
        self,
        state: State,
    ) -> Tuple[chex.Array, chex.Array, State]:
        """Return separate ``terminated`` and ``truncated`` agent arrays."""

        winding_vector = state.cartesian_states[:, [0, 1]] - self.winding_point

        # angle differentials, from new winding vectors to previous winding vectors
        # corrected by racing direction
        winding_angles = (
            jnp.arctan2(
                jnp.cross(state.prev_winding_vector, winding_vector),
                jnp.einsum("ij,ij->i", state.prev_winding_vector, winding_vector),
            )
            * self.winding_direction
        )

        state = state.replace(
            last_accumulated_angles=state.accumulated_angles,
            accumulated_angles=state.accumulated_angles + winding_angles,
        )
        state = state.replace(
            num_laps=(state.accumulated_angles / (2 * jnp.pi)).astype(int)
        )
        laps_done = state.num_laps >= self.params.max_num_laps

        # num steps done
        steps_done = state.step >= self.params.max_steps

        terminated = jnp.logical_or(state.collisions, laps_done)
        truncated = jnp.logical_and(~terminated, steps_done)
        done = terminated | truncated

        # update state
        state = state.replace(done=done)
        state = state.replace(prev_winding_vector=winding_vector)

        return terminated, truncated, state

    @partial(jax.jit, static_argnums=[0])
    def check_done(self, state: State) -> Tuple[Dict[str, bool], State]:
        """Legacy combined-done API."""

        terminated, truncated, state = self.check_done_array(state)
        return self._agent_array_to_dict(terminated | truncated), state

    @partial(jax.jit, static_argnums=[0])
    def get_reward_array(self, state: State) -> chex.Array:
        """Return all agent rewards as ``[agents]`` without Python loops."""

        rewards = jnp.zeros((self.num_agents,), dtype=state.cartesian_states.dtype)
        if "time" in self.reward_types:
            rewards = rewards - self.params.timestep * self.params.timestep_ratio
        if "progress" in self.reward_types:
            previous = jnp.mod(state.last_frenet_states[:, 0], self.track_length)
            current = jnp.mod(state.frenet_states[:, 0], self.track_length)
            difference = current - previous
            progress = jnp.where(
                difference > 0.95 * self.track_length,
                difference - self.track_length,
                jnp.where(
                    difference < -0.95 * self.track_length,
                    difference + self.track_length,
                    difference,
                ),
            )
            rewards = rewards + progress
        if "alive" in self.reward_types:
            rewards = rewards - state.collisions.astype(rewards.dtype)
        return rewards

    @partial(jax.jit, static_argnums=[0])
    def get_reward(self, state: State) -> Dict[str, float]:
        """Legacy agent-keyed reward API."""

        return self._agent_array_to_dict(self.get_reward_array(state))

    @partial(jax.jit, static_argnums=[0])
    def _ret_orig_state(self, state: State, key: chex.PRNGKey = None) -> State:
        return state

    def _validated_scan_ranges(
        self,
        name: str,
        ranges: chex.Array,
        dtype,
    ) -> chex.Array:
        scan_ranges = jnp.asarray(ranges, dtype=dtype)
        expected_shape = (self.num_agents, self.num_beams)
        if scan_ranges.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        return scan_ranges

    @partial(
        jax.jit,
        static_argnums=(0,),
        static_argnames=("external_scan_hook", "scan_corruption_hook"),
    )
    def scan_with_hooks(
        self,
        key: chex.PRNGKey,
        state: State,
        external_scan_ranges: Optional[chex.Array] = None,
        scan_corruption_ranges: Optional[chex.Array] = None,
        *,
        external_scan_hook: Optional[ScanHook] = None,
        scan_corruption_hook: Optional[ScanHook] = None,
    ) -> State:
        """Generate and compose all-agent scans without an agent/beam loop.

        All range arrays have shape ``[agents, beams]``.  External ranges are
        finite-minimum-combined with the static map scan.  A corruption array
        replaces that combined scan, while a corruption callable transforms
        the current scan.  Per-call callables override constructor callables.
        """

        state = self._scan_map(state)
        ranges = state.scans
        external_key = jax.random.fold_in(key, 1)
        corruption_key = jax.random.fold_in(key, 2)

        if external_scan_ranges is not None:
            external_ranges = self._validated_scan_ranges(
                "external_scan_ranges",
                external_scan_ranges,
                ranges.dtype,
            )
            ranges = combine_scan_ranges(ranges, external_ranges, self.max_range)

        effective_external_hook = (
            self.external_scan_hook
            if external_scan_hook is None
            else external_scan_hook
        )
        if effective_external_hook is not None:
            hook_state = state.replace(scans=ranges)
            external_ranges = self._validated_scan_ranges(
                "external scan hook result",
                effective_external_hook(external_key, hook_state, ranges),
                ranges.dtype,
            )
            ranges = combine_scan_ranges(ranges, external_ranges, self.max_range)

        # Preserve the simulator's original Gaussian scan noise while keeping
        # all external geometry merges ahead of sensor corruption.  Folded-in
        # hook keys are distinct from this legacy noise key.
        ranges = ranges + jax.random.normal(key, ranges.shape) * 0.01

        if scan_corruption_ranges is not None:
            ranges = self._validated_scan_ranges(
                "scan_corruption_ranges",
                scan_corruption_ranges,
                ranges.dtype,
            )

        effective_corruption_hook = (
            self.scan_corruption_hook
            if scan_corruption_hook is None
            else scan_corruption_hook
        )
        if effective_corruption_hook is not None:
            hook_state = state.replace(scans=ranges)
            ranges = self._validated_scan_ranges(
                "scan corruption hook result",
                effective_corruption_hook(corruption_key, hook_state, ranges),
                ranges.dtype,
            )

        return state.replace(scans=ranges)

    @partial(jax.jit, static_argnums=[0])
    def _scan(self, state: State, key: chex.PRNGKey) -> State:
        """Legacy scan entry point with configured hooks applied."""

        return self.scan_with_hooks(key, state)

    @partial(jax.jit, static_argnums=[0])
    def _scan_map(self, state: State) -> State:
        """Generate the raw static occupancy-map scan."""

        get_scan_vmapped = jax.jit(
            jax.vmap(
                partial(
                    get_scan,
                    theta_dis=self.theta_dis,
                    fov=self.fov,
                    num_beams=self.num_beams,
                    theta_index_increment=self.theta_index_increment,
                    sines=self.scan_sines,
                    cosines=self.scan_cosines,
                    eps=self.eps,
                    orig_x=self.orig_x,
                    orig_y=self.orig_y,
                    orig_c=self.orig_c,
                    orig_s=self.orig_s,
                    height=self.height,
                    width=self.width,
                    resolution=self.resolution,
                    dt=self.distance_transform,
                    max_range=self.max_range,
                ),
                in_axes=[0],
            )
        )
        scans = get_scan_vmapped(state.cartesian_states[:, [0, 1, 4]])
        new_state = state.replace(scans=scans)
        return new_state

    @partial(jax.jit, static_argnums=[0])
    def _collisions(self, state: State) -> State:
        # extract vertices from all cars (n_agent, 4, 2)
        all_vertices = jax.vmap(
            partial(get_vertices, length=self.params.length, width=self.params.width),
            in_axes=[0],
        )(state.cartesian_states[:, [0, 1, 4]])

        # check pairwise collisions
        pairwise_indices1, pairwise_indices2 = jnp.triu_indices(self.num_agents, 1)
        pairwise_vertices = jnp.concatenate(
            (all_vertices[pairwise_indices1], all_vertices[pairwise_indices2]), axis=-1
        )
        # (n_agent!, )
        pairwise_collisions = jax.vmap(collision, in_axes=[0])(pairwise_vertices)

        # get indices that are colliding
        collided_ind = jax.lax.select(
            jnp.column_stack((pairwise_collisions, pairwise_collisions)),
            jnp.column_stack((pairwise_indices1, pairwise_indices2)),
            -1 * jnp.ones((len(pairwise_indices1), 2), dtype=int),
        ).flatten()
        padded_collisions = jnp.zeros((self.num_agents + 1,))
        padded_collisions = padded_collisions.at[collided_ind].set(1)
        padded_collisions = padded_collisions[:-1]

        # check map collisions (n_agent, )
        map_collisions = collision_map(all_vertices, self.pixel_centers)

        # combine collisions
        full_collisions = jnp.logical_or(padded_collisions, map_collisions)

        # if already collided last step also collided this step
        full_collisions = jnp.logical_or(full_collisions, state.collisions)

        # update state
        state = state.replace(collisions=full_collisions)
        return state
