"""Q5 evidence: SAC 50k steps, same track, MLP, headless. Measure PEAK RSS.
Also reports the replay-buffer memory math for the default 1M buffer."""
import time, threading, warnings, os, sys
import numpy as np, psutil
warnings.filterwarnings("ignore")
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from sb3_wrapper import F1TenthSB3

TOTAL = 50_000
BUFFER = 1_000_000  # SB3 SAC default

proc = psutil.Process(os.getpid())
peak = {"rss": 0}
_stop = threading.Event()
def sampler():
    while not _stop.is_set():
        peak["rss"] = max(peak["rss"], proc.memory_info().rss)
        time.sleep(0.05)

def main():
    env = Monitor(F1TenthSB3("Spielberg"))
    obs_dim = env.observation_space.shape[0]
    # theoretical replay math (obs + next_obs, float32)
    theo = 2 * BUFFER * obs_dim * 4
    print(f"OBS_DIM={obs_dim}")
    print(f"BUFFER_THEORY_GB={theo/1e9:.2f}  (default buffer_size={BUFFER})")

    t = threading.Thread(target=sampler, daemon=True); t.start()
    base_rss = proc.memory_info().rss
    try:
        model = SAC("MlpPolicy", env, buffer_size=BUFFER, verbose=0, device="cpu")
        print("SAC_BUFFER_ALLOC=OK")
        t0 = time.time()
        model.learn(total_timesteps=TOTAL, progress_bar=False)
        dt = time.time() - t0
        print(f"SAC_WALL_SECONDS={dt:.1f}")
        print(f"SAC_FPS={TOTAL/dt:.1f}")
        print("SAC_RAN=OK")
    except MemoryError as e:
        print("SAC_BUFFER_ALLOC=MEMORYERROR", e)
    finally:
        _stop.set(); t.join()
    print(f"SAC_BASE_RSS_GB={base_rss/1e9:.2f}")
    print(f"SAC_PEAK_RSS_GB={peak['rss']/1e9:.2f}")
    print(f"SAC_DEFAULT_BUFFER_USABLE={peak['rss'] < 0.75*psutil.virtual_memory().total}")
    env.close()

if __name__ == "__main__":
    main()
