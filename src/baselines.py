"""Non-learned reference policies evaluated under the identical protocol.

Two baselines bracket the learned agents:

*   ``RandomPolicy``  - the floor. Uniform samples from the action space.
*   ``GapFollower``   - what a classical reactive rule achieves with no training.

Both expose the same ``predict(obs, deterministic=...) -> (action, state)``
signature that Stable-Baselines3 policies expose, so ``evaluate.py`` drives them
through exactly the same harness, circuits, episode caps, metrics and CSV
columns as a trained model. No special-casing in the evaluation loop.

Observation layout (produced by F1TenthSB3Wrapper._process_obs):

    obs[0:108]  downsampled LiDAR, every 10th of 1080 beams, metres
    obs[108]    linear_vels_x        obs[109]  linear_vels_y
    obs[110]    ang_vels_z           obs[111]  poses_theta
    obs[112]    unused, always 0.0

Actions are the normalised [-1, 1] pair the wrapper rescales to
[steering (rad), speed (m/s)].

Attribution
-----------
``GapFollower`` is a port of the Follow The Gap planner from f1tenth_benchmarks:

    f1tenth_benchmarks/classic_racing/FollowTheGap.py
    https://github.com/BDEvan5/f1tenth_benchmarks

which implements the method of:

    Sezer, V. and Gokasan, M. (2012), "A novel obstacle avoidance algorithm:
    Follow the Gap Method", Robotics and Autonomous Systems 60(9), 1123-1134.

The algorithm is reproduced step for step, with the same method names, so it
can be diffed against the original: ``preprocess_lidar`` (crop the rear beams,
mean filter, clip), an index-width bubble zeroed around the nearest return,
``find_max_gap`` over contiguous non-zero runs, ``find_best_point`` by sliding
window average, ``get_angle`` with the one-half gain, then three speed bands.

Three deviations, each marked at its site:

1.  It reads the 108-beam observation our policies receive rather than the raw
    1080-beam scan, so the baseline is held to exactly the input the agents
    get. Every index constant is a fraction of scan length, so the port
    behaves the same at either resolution.
2.  ``find_max_gap`` falls back to the longest gap when none clears the safety
    threshold. The original returns ``None`` there and raises on unpack.
3.  Output is converted to the normalised action pair this project's wrapper
    expects. The original returns physical ``[steering, speed]``.

Tuning constants are upstream's FollowTheGap.yaml verbatim, not chosen here.

The original computes ``radians_per_elem`` as ``2*pi / len(ranges)`` even though
an F1TENTH LiDAR spans 4.7 rad, not 2*pi. That overstates each beam's angle by
about 1.34x, which the one-half gain in ``get_angle`` then partly cancels, for
a net proportional gain of roughly 0.67 on the true bearing. It is preserved
here deliberately: it is load-bearing tuning, not an incidental bug, and the
ratio is identical at 108 beams as at 1080.
"""

from __future__ import annotations

import numpy as np

# Matches the simulator defaults asserted in the wrapper's observation.
N_BEAMS = 108
RAW_BEAMS = 1080
FOV = 4.7  # rad, 269.3 degrees
MAX_RANGE = 30.0

# Angle of each downsampled beam, measured from straight ahead. Used by
# RandomPolicy's docs and available for analysis; the gap follower deliberately
# uses the original's own index-to-angle conversion instead.
_RAW_STEP = FOV / (RAW_BEAMS - 1)
BEAM_ANGLES = -FOV / 2.0 + np.arange(0, RAW_BEAMS, 10) * _RAW_STEP


class BasePolicy:
    """Minimal stand-in for an SB3 policy."""

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(obs, dtype=np.float64)
        single = obs.ndim == 1
        batch = obs[None, :] if single else obs
        actions = np.stack([self._act(o) for o in batch]).astype(np.float32)
        return (actions[0] if single else actions), None

    def _act(self, obs):
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """Uniform over the normalised action space. The floor for the protocol.

    ``deterministic`` is accepted and ignored: a random policy has no greedy
    mode, and evaluation passes the flag unconditionally.
    """

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def _act(self, obs):
        return self.rng.uniform(-1.0, 1.0, size=2)


class GapFollower(BasePolicy):
    """Follow The Gap, ported from f1tenth_benchmarks. See module docstring.

    Defaults are the upstream FollowTheGap.yaml verbatim. Index constants are
    carried as fractions of scan length so the same settings apply whether the
    port is fed 108 beams or the raw 1080; the numerators are the upstream
    1080-beam values (crop 135 from the source, bubble 160, best-point window
    80, safety threshold 5).

    Note that fast_speed and straights_speed are both 5.0 upstream, so the
    three-band schedule collapses to two speeds in practice: 3.0 m/s when the
    steering command exceeds 0.174 rad, 5.0 m/s otherwise. That is upstream's
    choice, preserved rather than "fixed".
    """

    name = "gap"

    def __init__(
        self,
        n_beams: int = N_BEAMS,
        crop_fraction: float = 135 / 1080,
        bubble_fraction: float = 160 / 1080,
        best_point_fraction: float = 80 / 1080,
        safe_threshold_fraction: float = 5 / 1080,
        preprocess_conv_size: int = 3,
        max_lidar_dist: float = 10.0,
        straights_steering_angle: float = 0.174,
        fast_steering_angle: float = 0.0785,
        corners_speed: float = 3.0,
        straights_speed: float = 5.0,
        fast_speed: float = 5.0,
        max_steer: float = 0.4,
        wrapper_steering_limit: float = 0.4189,
        speed_min: float = 0.0,
        speed_max: float = 20.0,
    ):
        self.n_beams = int(n_beams)
        self.crop = max(1, int(round(crop_fraction * self.n_beams)))
        self.bubble_radius = max(1, int(round(bubble_fraction * self.n_beams)))
        self.best_point_conv_size = max(1, int(round(best_point_fraction * self.n_beams)))
        self.safe_threshold = max(1, int(round(safe_threshold_fraction * self.n_beams)))
        self.preprocess_conv_size = max(1, int(preprocess_conv_size))
        self.max_lidar_dist = max_lidar_dist
        self.straights_steering_angle = straights_steering_angle
        self.fast_steering_angle = fast_steering_angle
        self.corners_speed = corners_speed
        self.straights_speed = straights_speed
        self.fast_speed = fast_speed
        # Upstream clips steering at 0.4 rad, just inside the vehicle's actual
        # s_max of 0.4189. These must stay separate: get_angle clips at
        # max_steer, but the normalised action is scaled by the limit the
        # wrapper will multiply back out. Normalising by 0.4 would hand the
        # wrapper 1.0 for a 0.4 rad command and get 0.4189 rad back, inflating
        # every steering command by 4.7% and quietly undoing upstream's margin.
        self.max_steer = max_steer
        self.wrapper_steering_limit = wrapper_steering_limit
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.radians_per_elem = None

    # ------------------------------------------------------- ported methods

    def preprocess_lidar(self, ranges):
        """Mean-filter, drop the rear beams, and clip. Ported verbatim."""
        # Original: (2 * np.pi) / len(ranges). Preserved; see module docstring.
        self.radians_per_elem = (2 * np.pi) / len(ranges)
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges[self.crop:-self.crop], dtype=np.float64)
        proc_ranges = np.convolve(
            proc_ranges, np.ones(self.preprocess_conv_size), "same"
        ) / self.preprocess_conv_size
        proc_ranges = np.clip(proc_ranges, 0, self.max_lidar_dist)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """Start and end index of the chosen gap in the bubble-masked scan."""
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        slices = np.ma.notmasked_contiguous(masked)
        if slices is None:
            return None
        if isinstance(slices, slice):
            slices = [slices]
        for sl in slices[::-1]:
            if sl.stop - sl.start > self.safe_threshold:
                return sl.start, sl.stop
        # DEVIATION 2: the original returns None here and the caller raises.
        # Fall back to the longest gap so a tight section cannot kill the run.
        longest = max(slices, key=lambda s: s.stop - s.start)
        return longest.start, longest.stop

    def find_best_point(self, start_i, end_i, ranges):
        """Sliding-window average over the gap, then its argmax."""
        averaged_max_gap = np.convolve(
            ranges[start_i:end_i], np.ones(self.best_point_conv_size), "same"
        ) / self.best_point_conv_size
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """Index to steering angle, with the original's one-half gain."""
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return float(np.clip(steering_angle, -self.max_steer, self.max_steer))

    def plan(self, scan):
        """The original's plan(), returning physical [steering, speed]."""
        proc_ranges = self.preprocess_lidar(scan)

        # Find closest point to LiDAR
        closest = proc_ranges.argmin()

        # Eliminate all points inside 'bubble' (set them to zero)
        min_index = closest - self.bubble_radius
        max_index = closest + self.bubble_radius
        if min_index < 0:
            min_index = 0
        if max_index >= len(proc_ranges):
            max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        gap = self.find_max_gap(proc_ranges)
        if gap is None:
            # Every beam blanked: hold the wheel straight and crawl.
            return np.array([0.0, self.corners_speed])
        gap_start, gap_end = gap

        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        steering_angle = self.get_angle(best, len(proc_ranges))
        if abs(steering_angle) > self.straights_steering_angle:
            speed = self.corners_speed
        elif abs(steering_angle) > self.fast_steering_angle:
            speed = self.straights_speed
        else:
            speed = self.fast_speed

        return np.array([steering_angle, speed])

    # ------------------------------------------------------------- adapter

    def _act(self, obs):
        # DEVIATION 1: the 108-beam observation, not the raw 1080-beam scan.
        steering_angle, speed = self.plan(np.asarray(obs[:self.n_beams],
                                                     dtype=np.float64))
        # DEVIATION 3: convert to this project's normalised action pair.
        span = self.speed_max - self.speed_min
        return np.array([
            float(np.clip(steering_angle / self.wrapper_steering_limit, -1.0, 1.0)),
            float(np.clip(2.0 * (speed - self.speed_min) / span - 1.0, -1.0, 1.0)),
        ])


BASELINES = {"random": RandomPolicy, "gap": GapFollower}


def make_baseline(name: str, seed: int = 0) -> BasePolicy:
    """Construct a baseline by name, matching the --baseline CLI values."""
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}. Choose from {sorted(BASELINES)}.")
    return RandomPolicy(seed=seed) if name == "random" else GapFollower()
