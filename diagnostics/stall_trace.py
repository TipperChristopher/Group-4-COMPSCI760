import sys; sys.argv=['x']
exec(open('_verify/corner_trace.py', encoding='utf-8').read().split('RUNS = {')[0])
for step in [1000000, 2000000]:
    df, prog, crashed = rollout('PPO_1tracks_s2_G999_s2', step, 0)
    print(f'\nseed 2 @ {step/1e6:.1f}M spawn 0 -> {"crash" if crashed else "no crash"} at {prog:.1f} m, {len(df)} steps')
    for a in range(50, 80, 3):
        z = df[(df.s >= a) & (df.s < a + 3)]
        if len(z): print(f'   {a:>3}-{a+3} m: {len(z):>5} steps  speed {z.speed.mean():5.2f}  cmd_speed {z.cmd_speed.mean():5.2f}  steer {z.steer.mean():+.2f}')
