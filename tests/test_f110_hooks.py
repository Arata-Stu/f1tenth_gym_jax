import unittest

import jax
import jax.numpy as jnp

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs import combine_scan_ranges


NO_SCAN_ENV_ID = (
    "Spielberg_2_noscan_nocollision_progress_"
    "acceleration+steeringvelocity_1_5_v0"
)
SCAN_ENV_ID = (
    "Spielberg_2_scan_nocollision_progress_"
    "acceleration+steeringvelocity_1_5_v0"
)


def _fixed_external_ranges(key, state, ranges):
    del key, state
    return jnp.full_like(ranges, 0.5)


def _offset_corruption(key, state, ranges):
    del key, state
    return ranges + 1.0


def _updated_x_corruption(key, state, ranges):
    del key
    return jnp.broadcast_to(
        state.cartesian_states[:, 0, None],
        ranges.shape,
    )


def _wrong_shape_hook(key, state, ranges):
    del key, state
    return ranges[:, :-1]


class TestF110Hooks(unittest.TestCase):
    def test_external_ranges_use_finite_minimum(self):
        static = jnp.array(
            [[4.0, 5.0, 6.0, jnp.nan, jnp.inf, jnp.nan, jnp.inf, 4.0]]
        )
        external = jnp.array(
            [[3.0, jnp.inf, -1.0, 2.0, 1.0, 10.0, 10.0, 10.0]]
        )

        combined = combine_scan_ranges(static, external, max_range=10.0)

        expected = jnp.array(
            [[3.0, 5.0, 6.0, 2.0, 1.0, jnp.nan, jnp.inf, 4.0]]
        )
        self.assertTrue(bool(jnp.allclose(combined, expected, equal_nan=True)))

    def test_external_ranges_require_static_scan_shape(self):
        with self.assertRaisesRegex(ValueError, "match static scan shape"):
            combine_scan_ranges(jnp.ones((2, 4)), jnp.ones((2, 3)), 10.0)

    def test_make_separates_environment_hooks_from_param_overrides(self):
        env = make(
            SCAN_ENV_ID,
            external_scan_hook=_fixed_external_ranges,
            scan_corruption_hook=_offset_corruption,
            scan_only_observation=True,
        )

        self.assertIs(env.external_scan_hook, _fixed_external_ranges)
        self.assertIs(env.scan_corruption_hook, _offset_corruption)
        self.assertTrue(env.scan_only_observation)
        self.assertEqual(env.observation_space("agent_0").shape, (env.num_beams,))
        self.assertEqual(env.observation_space_ind["dynamics_state"], [])
        self.assertEqual(env.observation_space_ind["scan"], list(range(env.num_beams)))

        reset_key = jax.random.key(0)
        observations, state = env.reset_array(reset_key)
        raw_static = env._scan_map(state).scans
        expected = combine_scan_ranges(
            raw_static,
            jnp.full_like(raw_static, 0.5),
            env.max_range,
        )
        scan_key = jax.random.fold_in(reset_key, 1)
        expected = expected + jax.random.normal(scan_key, expected.shape) * 0.01
        expected = expected + 1.0
        self.assertTrue(bool(jnp.allclose(state.scans, expected)))
        self.assertTrue(bool(jnp.allclose(observations, expected)))

    def test_scan_only_observation_requires_scans(self):
        with self.assertRaisesRegex(ValueError, "requires produce_scans"):
            make(NO_SCAN_ENV_ID, scan_only_observation=True)

    def test_constructor_and_hook_result_validation(self):
        with self.assertRaisesRegex(TypeError, "external_scan_hook"):
            make(SCAN_ENV_ID, external_scan_hook=jnp.ones((2, 2)))

        env = make(SCAN_ENV_ID)
        _, state = env.reset_array(jax.random.key(0))
        with self.assertRaisesRegex(ValueError, "hook result"):
            env.scan_with_hooks(
                jax.random.key(1),
                state,
                external_scan_hook=_wrong_shape_hook,
            )

    def test_scan_only_observation_does_not_expose_state_fields(self):
        env = make(SCAN_ENV_ID, scan_only_observation=True)
        observations, state = env.reset_array(jax.random.key(0))

        self.assertEqual(observations.shape, (env.num_agents, env.num_beams))
        self.assertTrue(bool(jnp.allclose(observations, state.scans)))

    def test_scan_arrays_and_callables_have_distinct_composition_rules(self):
        env = make(SCAN_ENV_ID)
        _, state = env.reset_array(jax.random.key(0))
        key = jax.random.key(1)
        raw_static = env._scan_map(state).scans
        external = jnp.full_like(raw_static, 0.5)
        merged = combine_scan_ranges(raw_static, external, env.max_range)
        merged = merged + jax.random.normal(key, merged.shape) * 0.01

        external_state = env.scan_with_hooks(
            key,
            state,
            external_scan_ranges=external,
        )
        self.assertTrue(bool(jnp.allclose(external_state.scans, merged)))

        replacement = jnp.full_like(raw_static, 7.0)
        array_corrupted_state = env.scan_with_hooks(
            key,
            state,
            external_scan_ranges=external,
            scan_corruption_ranges=replacement,
        )
        self.assertTrue(bool(jnp.array_equal(array_corrupted_state.scans, replacement)))

        callable_corrupted_state = env.scan_with_hooks(
            key,
            state,
            external_scan_ranges=external,
            scan_corruption_hook=_offset_corruption,
        )
        self.assertTrue(
            bool(jnp.allclose(callable_corrupted_state.scans, merged + 1.0))
        )

    def test_step_hook_observes_integrated_all_agent_state(self):
        env = make(SCAN_ENV_ID, scan_only_observation=True)
        _, state = env.reset_array(jax.random.key(0))
        moving_states = state.cartesian_states.at[:, 3].set(2.0)
        state = state.replace(cartesian_states=moving_states)
        initial_x = state.cartesian_states[:, 0]

        result = env.step_env_array(
            jax.random.key(1),
            state,
            jnp.zeros((env.num_agents, 2)),
            scan_corruption_hook=_updated_x_corruption,
        )

        self.assertFalse(
            bool(jnp.allclose(result.state.cartesian_states[:, 0], initial_x))
        )
        expected = jnp.broadcast_to(
            result.state.cartesian_states[:, 0, None],
            result.state.scans.shape,
        )
        self.assertTrue(bool(jnp.allclose(result.state.scans, expected)))
        self.assertTrue(bool(jnp.allclose(result.observations, expected)))

    def test_external_pose_reset_casts_to_float_and_wraps_frenet_s(self):
        env = make(NO_SCAN_ENV_ID)
        cartesian_poses = jnp.array([[0, 0, 0], [1, 0, 0]])
        _, cartesian_state = env.reset_from_cartesian_poses(
            jax.random.key(0),
            cartesian_poses,
        )
        self.assertTrue(
            jnp.issubdtype(cartesian_state.cartesian_states.dtype, jnp.inexact)
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    cartesian_state.cartesian_states[:, [0, 1, 4]],
                    cartesian_poses,
                )
            )
        )

        frenet_poses = jnp.array(
            [[env.track.s_frame_max + 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        _, frenet_state = env.reset_from_frenet_poses(
            jax.random.key(1),
            frenet_poses,
        )
        self.assertTrue(bool(jnp.all(frenet_state.frenet_states[:, 0] >= 0.0)))
        self.assertTrue(
            bool(
                jnp.all(
                    frenet_state.frenet_states[:, 0] < env.track.s_frame_max
                )
            )
        )

    def test_complete_state_reset_preserves_external_dynamic_state(self):
        env = make(NO_SCAN_ENV_ID)
        _, state = env.reset_array(jax.random.key(0))
        cartesian_states = state.cartesian_states.at[:, 3].set(
            jnp.array([1.0, 2.0])
        )
        injected = state.replace(
            step=jnp.asarray(3),
            cartesian_states=cartesian_states,
        )

        observations, reset_state = env.reset_from_state(
            jax.random.key(1),
            injected,
        )

        self.assertEqual(int(reset_state.step), 3)
        self.assertTrue(
            bool(jnp.allclose(reset_state.cartesian_states[:, 3], [1.0, 2.0]))
        )
        self.assertTrue(
            bool(jnp.allclose(observations, env.get_obs_array(reset_state)))
        )

    def test_cartesian_state_array_reset_initializes_environment_counters(self):
        env = make(NO_SCAN_ENV_ID)
        _, state = env.reset_array(jax.random.key(0))
        external_states = state.cartesian_states.at[:, 3].set(
            jnp.array([1.0, 2.0])
        )

        _, reset_state = env.reset_from_cartesian_states(
            jax.random.key(1),
            external_states,
        )

        self.assertTrue(
            bool(jnp.allclose(reset_state.cartesian_states, external_states))
        )
        self.assertEqual(int(reset_state.step), 0)
        self.assertFalse(bool(jnp.any(reset_state.done)))
        self.assertFalse(bool(jnp.any(reset_state.collisions)))

    def test_array_step_matches_legacy_dictionary_transition(self):
        env = make(NO_SCAN_ENV_ID)
        _, state = env.reset_array(jax.random.key(0))
        actions = jnp.array([[0.1, 0.5], [-0.1, 0.25]])
        key = jax.random.key(1)

        array_result = env.step_env_array(key, state, actions)
        legacy_actions = {
            "agent_0": actions[0],
            "agent_1": actions[1],
        }
        observations, legacy_state, rewards, dones, infos = env.step_env(
            key,
            state,
            legacy_actions,
        )

        self.assertTrue(
            bool(jnp.allclose(observations["agent_0"], array_result.observations[0]))
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    legacy_state.cartesian_states,
                    array_result.state.cartesian_states,
                )
            )
        )
        self.assertTrue(bool(jnp.allclose(rewards["agent_1"], array_result.rewards[1])))
        self.assertEqual(
            bool(dones["__all__"]),
            bool(jnp.all(array_result.terminated | array_result.truncated)),
        )
        self.assertEqual(infos, array_result.infos)

    def test_array_step_is_vmap_composable(self):
        env = make(NO_SCAN_ENV_ID)
        _, state = env.reset_array(jax.random.key(0))
        batched_state = jax.tree.map(lambda value: jnp.stack((value, value)), state)
        actions = jnp.zeros((2, env.num_agents, 2))
        keys = jax.random.split(jax.random.key(1), 2)

        results = jax.vmap(env.step_env_array)(keys, batched_state, actions)

        self.assertEqual(
            results.observations.shape,
            (2, env.num_agents, env.observation_space("agent_0").shape[0]),
        )
        self.assertEqual(results.rewards.shape, (2, env.num_agents))
        self.assertEqual(results.terminated.shape, (2, env.num_agents))
        self.assertEqual(results.truncated.shape, (2, env.num_agents))

    def test_terminated_and_truncated_are_separate(self):
        env = make(NO_SCAN_ENV_ID)
        _, state = env.reset_array(jax.random.key(0))
        state = state.replace(step=jnp.asarray(env.params.max_steps - 1))

        result = env.step_env_array(
            jax.random.key(1),
            state,
            jnp.zeros((env.num_agents, 2)),
        )
        self.assertFalse(bool(jnp.any(result.terminated)))
        self.assertTrue(bool(jnp.all(result.truncated)))

        collision_state = result.state.replace(
            collisions=jnp.array([True, False]),
        )
        terminated, truncated, _ = env.check_done_array(collision_state)
        self.assertTrue(bool(terminated[0]))
        self.assertFalse(bool(truncated[0]))
        self.assertFalse(bool(terminated[1]))
        self.assertTrue(bool(truncated[1]))

        winding_vector = state.cartesian_states[:, :2] - env.winding_point
        lap_state = state.replace(
            step=jnp.asarray(0),
            collisions=jnp.zeros((env.num_agents,), dtype=bool),
            prev_winding_vector=winding_vector,
            accumulated_angles=jnp.full(
                (env.num_agents,),
                2 * jnp.pi * (env.params.max_num_laps + 0.25),
            ),
        )
        terminated, truncated, _ = env.check_done_array(lap_state)
        self.assertTrue(bool(jnp.all(terminated)))
        self.assertFalse(bool(jnp.any(truncated)))

    def test_array_step_auto_reset_preserves_terminal_flags(self):
        env = make(NO_SCAN_ENV_ID)
        _, state = env.reset_array(jax.random.key(0))
        state = state.replace(step=jnp.asarray(env.params.max_steps - 1))

        result = env.step_array(
            jax.random.key(1),
            state,
            jnp.zeros((env.num_agents, 2)),
        )

        self.assertTrue(bool(jnp.all(result.truncated)))
        self.assertFalse(bool(jnp.any(result.terminated)))
        self.assertEqual(int(result.state.step), 0)


if __name__ == "__main__":
    unittest.main()
