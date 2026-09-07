"""Track-pool support for the F1TENTH diversity experiment.

Why this module exists
----------------------
The experiment varies the *number of training tracks* (N = 1, 5, 20, 100) while
holding the *amount of training* constant. The obvious implementation, one
vectorised environment per track, breaks that on two separate counts.

1.  Stable-Baselines3 performs one learning update per ``vec_env.step()`` call,
    and every such call advances all ``n_envs`` environments at once. The number
    of updates therefore scales as ``total_timesteps / n_envs``. Tying
    ``n_envs`` to the diversity level confounds diversity with training amount.

2.  ``f1tenth_gym`` keeps its ray-casting engine in a *class* attribute,
    ``RaceCar.scan_simulator``, created once per process behind an
    ``if RaceCar.scan_simulator is None`` guard. Every ``F110Env`` built inside
    one process therefore shares a single occupancy grid. N environments built
    with N different maps all raycast against whichever map was loaded last,
    while each environment still spawns the car from its own track's centreline.
    The diversity axis was never actually reaching the LiDAR.

``TrackPoolWrapper`` resolves both by keeping exactly one environment alive and
swapping the map underneath it on every ``reset()``.

Map switching
-------------
``F110Env`` exposes ``update_map(map_name)``, which reloads the track from disk
and hands it to the simulator. Two things make it insufficient on its own:

*   It does not rebuild ``env.reset_fn``. That object is constructed once in
    ``__init__`` via ``make_reset_fn(..., track=self.track, ...)`` and is what
    ``reset()`` samples the starting pose from. Calling ``update_map`` alone
    leaves the car spawning on the *previous* track's centreline while the walls
    belong to the *new* map, which puts the car inside a wall on lap zero.
    This wrapper rebuilds ``reset_fn`` after every switch.

*   It reloads the track from disk each time, and ``ScanSimulator2D.set_map``
    then recomputes a Euclidean distance transform over the whole occupancy
    grid (1600x1600 for the synthetic tracks). This wrapper caches both the
    ``Track`` object and its distance transform so a repeat visit to a track is
    a handful of attribute assignments instead of a disk read plus a transform.

Every internal access below is guarded and falls back to the public
``update_map`` path, so an ``f1tenth_gym`` whose internals differ still works,
just more slowly.
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from typing import Iterable, Sequence

import gymnasium as gym
import numpy as np

__all__ = ["StratifiedTrackSampler", "TrackPoolWrapper"]


# Marker attribute written onto the shared ScanSimulator2D so that every
# wrapper in the process can tell which map is *actually* loaded, rather than
# trusting its own idea of what it last requested.
_APPLIED_ATTR = "_track_pool_applied_track"


class StratifiedTrackSampler:
    """Deal tracks from a shuffled deck, reshuffling when the deck runs out.

    This guarantees that over any window of ``len(pool)`` consecutive episodes
    every track is visited exactly once, so a run cannot get unlucky and spend
    most of its budget on a handful of tracks. Sampling uses its own
    ``np.random.Generator`` so that the track order is controlled by the track
    seed alone and is independent of the network-initialisation seed.
    """

    def __init__(self, tracks: Sequence[str], seed: int, stream: int = 0) -> None:
        self._tracks = [str(t) for t in tracks]
        if not self._tracks:
            raise ValueError("Track pool must contain at least one track.")
        self._seed = int(seed)
        self._stream = int(stream)
        # A list seed keeps every stream deterministic in the track seed while
        # giving parallel environments different orderings.
        self._rng = np.random.default_rng([self._seed, self._stream])
        self._order: list[str] = []
        self._cursor = 0
        self.epochs_completed = 0

    @property
    def pool(self) -> list[str]:
        return list(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)

    def next(self) -> str:
        """Return the next track, reshuffling once the deck is exhausted."""
        if self._cursor >= len(self._order):
            self._reshuffle()
        track = self._order[self._cursor]
        self._cursor += 1
        return track

    def _reshuffle(self) -> None:
        idx = self._rng.permutation(len(self._tracks))
        self._order = [self._tracks[i] for i in idx]
        self._cursor = 0
        self.epochs_completed += 1


class TrackPoolWrapper(gym.Wrapper):
    """Cycle one f1tenth environment through a pool of tracks.

    Parameters
    ----------
    env:
        The environment to wrap. May already be wrapped (for example by
        ``F1TenthSB3Wrapper``); the map swap is applied to ``env.unwrapped``.
    tracks:
        Names of the tracks in the pool, as understood by
        ``Track.from_track_name`` (for example ``"synthetic_track_0"``).
    track_seed:
        Seed for the sampler's generator. Keep this separate from the training
        seed so PPO and SAC see identical track orders.
    stream:
        Distinguishes parallel environments while staying deterministic in
        ``track_seed``. Leave at 0 for the single-environment setup.
    cache_size:
        How many tracks to hold in memory. Each entry costs roughly 20 MB (a
        10 MB float32 occupancy grid plus a 10 MB distance transform for the
        1600x1600 synthetic maps), so the default of 8 is about 165 MB. Set to
        0 to disable caching entirely.
    """

    def __init__(
        self,
        env: gym.Env,
        tracks: Iterable[str],
        track_seed: int,
        stream: int = 0,
        cache_size: int = 8,
    ) -> None:
        super().__init__(env)
        self.sampler = StratifiedTrackSampler(list(tracks), track_seed, stream)
        self._cache_size = max(0, int(cache_size))
        # name -> [Track, distance_transform_or_None]
        self._cache: "OrderedDict[str, list]" = OrderedDict()
        self._current_track: str | None = None
        self._applied_track: str | None = None
        self._warned_slow_path = False

    # ------------------------------------------------------------------ API

    @property
    def current_track(self) -> str | None:
        """Name of the track currently loaded, or None before the first reset."""
        return self._current_track

    @property
    def track_pool(self) -> list[str]:
        return self.sampler.pool

    def reset(self, *, seed=None, options=None):
        track = self.sampler.next()
        self._switch_track(track)
        self._current_track = track

        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info) if info else {}
        info["track_name"] = track
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Also stamp step infos: Stable-Baselines3 discards the info returned by
        # an auto-reset inside a VecEnv, so the step info is the only place a
        # Monitor or callback can reliably read the track from.
        info = dict(info) if info else {}
        info["track_name"] = self._current_track
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------- map switching

    def _switch_track(self, name: str) -> None:
        core = self.env.unwrapped
        scan = self._scan_simulator(core)

        # Skip the reload only when the shared simulator still holds this map
        # *and* this wrapper is the one that put it there. The scan simulator is
        # process-wide, so another environment may have swapped it underneath
        # us, and this env's own reset_fn is only valid if we built it.
        already_loaded = (
            self._applied_track == name
            and scan is not None
            and getattr(scan, _APPLIED_ATTR, None) == name
        )
        if already_loaded:
            return

        track = self._load_track(core, name)
        if track is None:
            # Slow path: no Track object available, use the public API.
            core.update_map(name)
        else:
            self._apply_track(core, scan, name, track)

        # Order matters: reset_fn is derived from core.track, so the env's own
        # bookkeeping has to point at the new map before the spawn sampler is
        # rebuilt. Rebuilding first would sample poses from the previous track.
        self._sync_env_track(core, name)
        self._rebuild_reset_fn(core)

        self._applied_track = name
        if scan is not None:
            setattr(scan, _APPLIED_ATTR, name)

    def _apply_track(self, core, scan, name: str, track) -> None:
        """Install ``track`` into the simulator, reusing a cached transform."""
        entry = self._cache.get(name)
        cached_dt = entry[1] if entry is not None else None

        if cached_dt is not None and scan is not None and self._fast_apply(scan, track, cached_dt):
            return

        # Official path. Accepts a Track object as well as a name, so the disk
        # read is still avoided on a cache hit; only the transform is recomputed.
        core.sim.set_map(track)

        if entry is not None and scan is not None:
            dt = getattr(scan, "dt", None)
            if dt is not None:
                entry[1] = dt

    def _fast_apply(self, scan, track, cached_dt) -> bool:
        """Mirror ScanSimulator2D.set_map without recomputing the transform."""
        try:
            occupancy = track.occupancy_map
            origin = track.spec.origin
            scan.track = track
            scan.map_img = occupancy
            scan.map_height = occupancy.shape[0]
            scan.map_width = occupancy.shape[1]
            scan.map_resolution = track.spec.resolution
            scan.origin = origin
            scan.orig_x = origin[0]
            scan.orig_y = origin[1]
            scan.orig_s = math.sin(origin[2])
            scan.orig_c = math.cos(origin[2])
            scan.dt = cached_dt
            return True
        except Exception as exc:  # pragma: no cover - depends on library version
            if not self._warned_slow_path:
                warnings.warn(
                    "TrackPoolWrapper could not apply a cached distance transform "
                    f"({exc!r}); falling back to sim.set_map on every switch. "
                    "Resets will be slower but results are unaffected.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_slow_path = True
            return False

    def _load_track(self, core, name: str):
        """Return a Track object for ``name``, from cache when possible."""
        entry = self._cache.get(name)
        if entry is not None:
            self._cache.move_to_end(name)
            return entry[0]

        try:
            from f1tenth_gym.envs.track import Track
        except Exception:
            try:
                from f1tenth_gym.envs.track.track import Track  # type: ignore
            except Exception:
                return None

        try:
            track = Track.from_track_name(name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load track {name!r}. Check that it exists under the "
                "f1tenth_gym maps directory (run generate_track_pool.py first)."
            ) from exc

        if self._cache_size > 0:
            self._cache[name] = [track, None]
            self._cache.move_to_end(name)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return track

    def _rebuild_reset_fn(self, core) -> None:
        """Re-derive the spawn-pose sampler from the newly loaded track.

        ``F110Env.update_map`` does not do this, and a stale ``reset_fn`` spawns
        the car using the previous track's centreline. That is silent: the
        episode simply starts inside a wall and ends immediately, which would
        look like a policy failure rather than a bug.
        """
        if not hasattr(core, "reset_fn"):
            return  # Older API without a pluggable reset function.

        try:
            from f1tenth_gym.envs.reset import make_reset_fn
        except Exception as exc:
            raise RuntimeError(
                "TrackPoolWrapper needs f1tenth_gym.envs.reset.make_reset_fn to "
                "rebuild the spawn sampler after a map switch, but it could not "
                "be imported. Without it the car spawns on the wrong track."
            ) from exc

        reset_config = dict(core.config.get("reset_config") or {})
        try:
            core.reset_fn = make_reset_fn(
                **reset_config,
                track=core.track,
                num_agents=core.num_agents,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to rebuild the spawn sampler for track {core.map!r}: {exc}"
            ) from exc

    @staticmethod
    def _sync_env_track(core, name: str) -> None:
        """Keep the env's own bookkeeping consistent with the loaded map."""
        scan_track = None
        try:
            scan_track = core.sim.agents[0].scan_simulator.track
        except Exception:
            pass
        if scan_track is not None:
            core.track = scan_track
        core.map = name
        try:
            core.config["map"] = name
        except Exception:
            pass

    @staticmethod
    def _scan_simulator(core):
        """Fetch the process-wide ScanSimulator2D, or None if unavailable."""
        try:
            return core.sim.agents[0].scan_simulator
        except Exception:
            return None
