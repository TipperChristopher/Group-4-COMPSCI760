from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

# Filename used for the running observation statistics that accompany a saved
# model. Evaluation must load these, because a policy trained on normalised
# observations sees garbage if handed raw ones.
VECNORMALIZE_FILENAME = "vecnormalize.pkl"

# --------------------------------------------------------------------------
# FROZEN REWARD CONSTANTS
# --------------------------------------------------------------------------
# These define the dependent variable of the study and must be identical in
# every cell of the grid. Do not tune them per-cell.
#
# reward = PROGRESS_WEIGHT * (metres advanced along the centreline)
#          - TIME_COST                      (per step)
#          - CRASH_PENALTY                  (once, on collision only)
#
# Why this shape, and what it replaced:
#
#   The previous reward was `timestep + 0.1*speed - 0.5*|steering|`. Summing
#   `0.1 * speed` at a 0.01 s timestep is just distance times ten, so that term
#   paid 10 per metre travelled, while the `timestep` term paid 1.0 per second
#   elapsed. A car driving straight into a wall covers the same distance
#   whatever its speed, so it banked the same distance reward either way and
#   the only remaining free variable was how long it took. Slower was strictly
#   better. Measured episode returns were 120.21 for a 0.25 m/s crawl against
#   80.91 at 20 m/s, and 100.00 for standing perfectly still.
#
#   Paying for progress along the centreline instead removes the conflict:
#   inside a fixed step budget, the only way to earn more is to cover more
#   track, so faster is monotonically better.
PROGRESS_WEIGHT = 1.0

# Deliberately zero. The survival constant is gone rather than inverted into a
# time cost, because any per-step bleed creates an incentive to crash early to
# stop it: idling to truncation would cost TIME_COST * max_episode_steps, and
# once that exceeds CRASH_PENALTY the agent prefers suicide. That threshold
# moves whenever the step limit changes, which would silently couple the reward
# to an unrelated CLI flag. With a finite horizon the step limit already
# supplies the time pressure, since total progress is what a bounded episode
# can accumulate. Standing still now scores exactly zero and driving scores
# positive, with no configuration that inverts the ordering.
TIME_COST = 0.0

# Fires on collision only, never on lap completion. Small relative to a lap
# (roughly 179 m on the synthetic tracks), so it discourages crashing without
# making the agent so crash-averse that it refuses to move.
CRASH_PENALTY = 5.0

# --------------------------------------------------------------------------
# ACTION SCALING
# --------------------------------------------------------------------------
# The policy acts in [-1, 1]^2 and the wrapper rescales to physical units.
# The raw f1tenth ranges are wildly asymmetric (steering +-0.4189 rad against
# speed spanning 25 m/s), so an SB3 Gaussian policy initialised at mean 0 with
# std 1 saturates steering every single step while needing a ~20 sigma shift of
# its mean to reach top speed. Normalising fixes both.
#
# A useful side effect: the neutral action 0 now maps to the midpoint of the
# speed range, so a freshly initialised policy drives at 10 m/s rather than
# crawling. The default behaviour is to move.
STEERING_LIMIT = 0.4189  # rad, matches params["s_min"] / ["s_max"]
SPEED_MIN = 0.0          # m/s. Reverse is not useful for racing and would earn
SPEED_MAX = 20.0         # negative progress anyway; braking to 0 stays allowed.

# 30 s of simulated time. Long enough for roughly one lap of a synthetic track
# at 6 m/s, short enough that a stuck episode does not eat the step budget.
DEFAULT_MAX_EPISODE_STEPS = 3000


class TrackProgress:
    """Stable, cheap arc-length position along a track centreline.

    Why not just call ``Track.cartesian_to_frenet``:

    *   It ignores its own ``s_guess`` argument. The signature accepts one but
        the body calls ``calc_arclength_inaccurate(x, y)``, which searches every
        segment of the centreline globally. On a circuit that passes close to
        itself (a hairpin, or the pit straight beside the last corner) the
        nearest point can jump to a distant part of the track, producing a
        spurious delta of half a lap in a single step.
    *   It costs about 117 us per call, which is roughly the cost of a whole
        physics step, so it would nearly double environment time.

    Searching a window around the previous match fixes both: it cannot jump to
    a far segment, and it touches ~120 segments instead of ~865.
    """

    def __init__(self, window: int = 60) -> None:
        self.window = int(window)
        self._track = None
        self.length = 0.0
        self._idx = 0
        self._prev_s = 0.0
        self.cumulative_s = 0.0

    def bind(self, track) -> None:
        """Point at a track, rebuilding the cached geometry if it changed.

        TrackPoolWrapper swaps the map on every reset, so this is called each
        episode and must be cheap when the track is unchanged.
        """
        if track is self._track:
            return
        spline = track.centerline.spline
        pts = np.asarray(spline.points, dtype=np.float64)
        self._track = track
        self.points = pts
        self.s = np.asarray(spline.s, dtype=np.float64)
        self.length = float(self.s[-1])
        self.seg_vec = pts[1:] - pts[:-1]
        self.seg_len2 = np.einsum("ij,ij->i", self.seg_vec, self.seg_vec)
        self.seg_len2[self.seg_len2 == 0.0] = 1e-12
        self.seg_len = np.sqrt(self.seg_len2)
        self.n_seg = len(self.seg_vec)

    def _project(self, x: float, y: float, idxs: np.ndarray) -> tuple[int, float]:
        """Nearest point to (x, y) among the given segment indices."""
        p0 = self.points[idxs]
        d = self.seg_vec[idxs]
        wx = x - p0[:, 0]
        wy = y - p0[:, 1]
        t = (wx * d[:, 0] + wy * d[:, 1]) / self.seg_len2[idxs]
        np.clip(t, 0.0, 1.0, out=t)
        ex = wx - t * d[:, 0]
        ey = wy - t * d[:, 1]
        k = int(np.argmin(ex * ex + ey * ey))
        return int(idxs[k]), float(t[k])

    def reset(self, x: float, y: float) -> None:
        """Locate the car globally. Called once per episode, cost is irrelevant."""
        i, t = self._project(x, y, np.arange(self.n_seg))
        self._idx = i
        self._prev_s = self.s[i] + t * self.seg_len[i]
        self.cumulative_s = 0.0

    def update(self, x: float, y: float) -> float:
        """Advance and return metres gained along the centreline this step."""
        lo = self._idx - self.window
        idxs = np.arange(lo, lo + 2 * self.window + 1) % self.n_seg
        i, t = self._project(x, y, idxs)
        self._idx = i
        s = self.s[i] + t * self.seg_len[i]

        delta = s - self._prev_s
        # Lap wraparound: crossing the start line takes s from ~length back to
        # ~0, which without this correction reads as a full lap of negative
        # progress. Anything past half a lap in one step is a wrap, not motion.
        half = 0.5 * self.length
        if delta < -half:
            delta += self.length
        elif delta > half:
            delta -= self.length

        self._prev_s = s
        self.cumulative_s += delta
        return delta

    @property
    def laps(self) -> float:
        """Signed laps completed, including the fractional part."""
        return self.cumulative_s / self.length if self.length else 0.0


class F1TenthSB3Wrapper(gym.Wrapper):
    """
    A custom wrapper to make f1tenth_gym compatible with Stable-Baselines3.
    Downsamples LiDAR and flattens the dictionary observation into a 1D vector,
    rescales actions from [-1, 1], and applies the frozen progress reward.
    """

    def __init__(self, env, max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS):
        super().__init__(env)

        # 108 beams + 5 ego state values = 113-dimensional vector
        self.obs_dim = 108 + 5
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Normalised actions. Rescaled to [steering, speed] in step().
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.max_episode_steps = int(max_episode_steps)
        self._elapsed = 0
        self.progress = TrackProgress()

    # ---------------------------------------------------------------- helpers

    def _rescale_action(self, action):
        """Map [-1, 1]^2 to [steering (rad), speed (m/s)]."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        steering = a[0] * STEERING_LIMIT
        speed = SPEED_MIN + (a[1] + 1.0) * 0.5 * (SPEED_MAX - SPEED_MIN)
        return steering, speed

    def _ego_xy(self):
        core = self.env.unwrapped
        return float(core.poses_x[0]), float(core.poses_y[0])

    # ------------------------------------------------------------------- API

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._elapsed = 0
        # TrackPoolWrapper has already swapped the map by the time this runs,
        # so unwrapped.track is the track this episode will actually use.
        self.progress.bind(self.env.unwrapped.track)
        self.progress.reset(*self._ego_xy())
        info = dict(info) if info else {}
        info["progress_m"] = 0.0
        info["laps"] = 0.0
        return self._process_obs(obs), info

    def step(self, action):
        steering, speed = self._rescale_action(action)
        formatted_action = np.array([[steering, speed]], dtype=np.float64)

        obs, _base_reward, terminated, truncated, info = self.env.step(formatted_action)

        # --- FROZEN REWARD ---
        # The env's own reward is the survival constant (it returns exactly
        # `timestep`) and is deliberately discarded.
        delta_s = self.progress.update(*self._ego_xy())
        reward = PROGRESS_WEIGHT * delta_s - TIME_COST

        # Terminal penalty is gated on the collision flag, not on `terminated`.
        # f110_env sets terminated for a collision OR for two completed laps
        # (`done = collisions[ego] or np.all(toggle_list >= 4)`), so keying off
        # `terminated` would punish finishing the race exactly like crashing.
        collided = bool(np.asarray(obs["collisions"]).reshape(-1)[0])
        if collided:
            reward -= CRASH_PENALTY

        # Step limit. Kept distinct from `terminated` so SB3 bootstraps the
        # value at a timeout instead of treating it as a real terminal state.
        # f110_env never truncates on its own (`truncated = False` is hardcoded
        # and the env is registered with no max_episode_steps), so without this
        # a stationary policy runs forever, reset() never fires, and the track
        # pool never advances, silently nullifying --diversity.
        self._elapsed += 1
        if not terminated and self._elapsed >= self.max_episode_steps:
            truncated = True

        info = dict(info) if info else {}
        info["progress_m"] = self.progress.cumulative_s
        info["laps"] = self.progress.laps
        info["collision"] = collided

        return self._process_obs(obs), float(reward), terminated, truncated, info

    def _process_obs(self, obs):
        scan = obs['scans'][0]

        # Downsample 1080 beams to 108 by taking every 10th beam
        downsampled_scan = scan[::10]

        # Extract 5 ego state values
        ego_state = np.array([
            obs['linear_vels_x'][0],
            obs['linear_vels_y'][0],
            obs['ang_vels_z'][0],
            obs['poses_theta'][0],
            0.0
        ], dtype=np.float32)

        # Concatenate LiDAR and vehicle state into a single 113-dim array
        flat_obs = np.concatenate([downsampled_scan, ego_state])
        return flat_obs.astype(np.float32)


# --------------------------------------------------------------------------
# Observation normalisation
# --------------------------------------------------------------------------
# The 113-d observation mixes LiDAR ranges (0-30 m), body velocities (-5 to
# 20 m/s) and a heading in radians. Those scales differ by an order of
# magnitude, which hurts PPO and SAC differently and would contaminate the
# algorithm comparison. Normalising observations identically for both puts
# them on equal footing.
#
# Rewards are deliberately left unnormalised: the reward is the dependent
# variable of the study, and VecNormalize's running return scaling would make
# the numbers incomparable across cells.


def wrap_vecnormalize(vec_env: VecEnv, training: bool = True) -> VecNormalize:
    """Apply observation-only normalisation. Used identically for PPO and SAC."""
    return VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        training=training,
    )


def vecnormalize_path(save_dir: str) -> str:
    """Canonical location of the statistics file for a run directory."""
    return os.path.join(save_dir, VECNORMALIZE_FILENAME)


def save_vecnormalize(vec_env, save_dir: str) -> str | None:
    """Save running observation statistics next to the model.

    Returns the path written, or None if the env is not normalised.
    """
    if not isinstance(vec_env, VecNormalize):
        return None
    os.makedirs(save_dir, exist_ok=True)
    path = vecnormalize_path(save_dir)
    vec_env.save(path)
    return path


def load_vecnormalize(vec_env: VecEnv, path: str) -> VecNormalize:
    """Restore saved statistics onto ``vec_env`` for evaluation.

    Freezes the statistics and disables reward normalisation so that evaluation
    measures the policy, not a moving normaliser.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No normalisation statistics at {path}. The policy was trained on "
            "normalised observations and cannot be evaluated without them."
        )
    normalised = VecNormalize.load(path, vec_env)
    normalised.training = False
    normalised.norm_reward = False
    return normalised
