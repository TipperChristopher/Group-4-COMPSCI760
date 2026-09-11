"""Q4 evidence: PPO 100k steps on one track, MLP, headless. Measure FPS + return trend."""
import time, warnings, numpy as np
warnings.filterwarnings("ignore")
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from sb3_wrapper import F1TenthSB3

TOTAL = 100_000

class ReturnTracker(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.ep_returns = []
    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.ep_returns.append(ep["r"])
        return True

def main():
    from stable_baselines3.common.monitor import Monitor
    env = Monitor(F1TenthSB3("Spielberg"))
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")  # defaults, no tuning
    cb = ReturnTracker()
    t0 = time.time()
    model.learn(total_timesteps=TOTAL, callback=cb, progress_bar=False)
    dt = time.time() - t0
    fps = TOTAL / dt
    print(f"PPO_WALL_SECONDS={dt:.1f}")
    print(f"PPO_FPS={fps:.1f}")
    print(f"PPO_EPISODES={len(cb.ep_returns)}")
    r = cb.ep_returns
    if len(r) >= 6:
        k = len(r) // 3
        first, last = np.mean(r[:k]), np.mean(r[-k:])
        print(f"PPO_RET_FIRST_THIRD={first:.3f}")
        print(f"PPO_RET_LAST_THIRD={last:.3f}")
        print(f"PPO_RET_TREND_UP={bool(last > first)}")
    else:
        print(f"PPO_RET_N={len(r)} (too few episodes to trend)")
    # extrapolations the plan asks for
    for steps, label in [(5_000_000, "one_5M_run")]:
        secs = steps / fps
        print(f"PPO_EXTRAP_{label}_hours={secs/3600:.2f}")
    print(f"PPO_EXTRAP_24runs_hours={24*5_000_000/fps/3600:.2f}")
    env.close()

if __name__ == "__main__":
    main()
