import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
import learning_curve as lc
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

lc.W.CRASH_PENALTY = 40.0
wrap_fn = lambda env, cap: lc.W.F1TenthSB3Wrapper(env, max_episode_steps=cap)

# --- build the EXACT training env learning_curve uses ---
venv = DummyVecEnv([lc.make_env(wrap_fn, lc.TARGET, lc.TRAIN_CAP)])
venv = lc.W.wrap_vecnormalize(venv, training=True)
model = PPO("MlpPolicy", venv, verbose=0, device="cpu", seed=0, ent_coef=0.0)

print("=== 1. SPACES ===")
print("  obs space  :", model.observation_space.shape, "(expect (113,) = 108 LiDAR + 5 ego)")
print("  act space  :", model.action_space.shape, model.action_space.low, "->", model.action_space.high, "(expect 2-dim [-1,1])")
print("  n_envs     :", venv.num_envs, "(expect 1)")
print("  norm_obs   :", venv.norm_obs, " norm_reward:", venv.norm_reward, "(expect True / False)")

print("\n=== 2. REWARD SEMANTICS (is it the progress reward?) ===")
raw = venv.venv.envs[0]  # unwrapped-ish F1TenthSB3Wrapper->Monitor
venv.reset()
# step at MIN speed (a[1]=-1 -> 0 m/s) : standstill -> reward ~ 0
r_still = venv.step(np.array([[0.0,-1.0]],dtype=np.float32))[1][0]
venv.reset()
# step at MAX speed straight (a[1]=+1 -> 20 m/s): should make positive progress
rs=[]
venv.reset()
for _ in range(20):
    rs.append(venv.step(np.array([[0.0,1.0]],dtype=np.float32))[1][0])
print(f"  standstill (0 m/s) step reward : {r_still:+.4f}   (expect ~0)")
print(f"  full-speed straight mean reward: {np.mean(rs):+.4f}   (expect > 0, = progress)")
print(f"  PROGRESS_WEIGHT={lc.W.PROGRESS_WEIGHT}  TIME_COST={lc.W.TIME_COST}  CRASH_PENALTY={lc.W.CRASH_PENALTY}")

print("\n=== 3. PPO HYPERPARAMS IN FORCE ===")
for k in ["learning_rate","n_steps","batch_size","n_epochs","gamma","gae_lambda","clip_range","ent_coef"]:
    v=getattr(model,k,None)
    if callable(v):
        try: v=v(1.0)
        except: pass
    print(f"  {k:14s}= {v}")

print("\n=== 4. SAVED RUN CONFIG (what actually ran) ===")
d=json.load(open("results/learning_curve_cp40_vn_1000000.json"))
print({k:d[k] for k in ["penalty","algo","vecnormalize","steps","ent_coef","completion_bonus","warmup_track"]})

print("\n=== 5. IS THE REWARD OSCILLATING? (training-side ep_rew_mean over 100 eps) ===")
rew=[x["ep_rew_mean"] for x in d["curve"]]
flips=sum(1 for i in range(1,len(rew)) if (rew[i]<0)!=(rew[i-1]<0))
diffs=[rew[i]-rew[i-1] for i in range(1,len(rew))]
dirflips=sum(1 for i in range(1,len(diffs)) if (diffs[i]>0)!=(diffs[i-1]>0))
print("  ep_rew per checkpoint:", [round(x,1) for x in rew])
print(f"  mean={np.mean(rew):.2f} std={np.std(rew):.2f}  (std> mean => not converged)")
print(f"  sign flips (+/-): {flips}   direction reversals (up<->down): {dirflips}/{len(diffs)-1}")
venv.close()
