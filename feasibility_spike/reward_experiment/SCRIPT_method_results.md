# Speaker script — Methodology & Results (our second half)

Covers the slides we authored: 8 (Evaluation Protocol), 10 (The Reward),
12 (PPO vs SAC), 13 (Why PPO Falls Short), 14 (Zero-Shot Results),
16 (Conclusions). Spoken pace assumed ≈ 140 words/min. Lines marked
`[cut to shorten]` can be dropped to bring the section from ~7 min to ~5 min
without losing the argument.

Transitions into/out of teammates' slides (9 Two Bugs, 11 Baselines, 15
Timeline) are the short italic lines.

---

## Slide 8 — Evaluation Protocol (≈ 65 s, 150 words)

Before any results, here is how we make the comparison fair. We split tracks
three ways. We train on synthetic tracks, we pick the best checkpoint on a
held-out synthetic validation set, and we test zero-shot on twenty-three real
circuits the policy never saw.

For scoring we do not use pass-or-fail lap completion, because that floors
every hard track to zero and hides the signal. We use graded fractional laps,
which is how far around the track the car actually got, plus completion, lap
time, and crash rate.

One choice matters a lot here. We select the best checkpoint on validation,
not the final one, because PPO peaks around 0.34 laps and then drifts down to
0.10. Its final policy is a noisy estimate of what it actually learned.

And every policy is evaluated the same way: on the centreline, with five
fixed spawn seeds, averaged per circuit. The reward, the crash penalty, and
normalization are frozen identically for PPO and SAC.

*Transition:* "That protocol only works because we first fixed two bugs in
the pipeline, which the next slide covers." *(hand to teammate, slide 9)*

---

## Slide 10 — The Reward: Design & Why It Changed (≈ 85 s, 200 words)

The reward is the one part of the task definition we deliberately redesigned,
so I want to be precise about it.

What we reward now is simple: metres of progress along the centreline, minus
a penalty when the car crashes. We set the time cost to zero on purpose. The
fixed step budget already creates time pressure, and any per-step cost would
actually reward the car for crashing early to end the episode.

Why did we change it? The original reward paid the car for every second it
stayed alive, plus a distance term, minus a steering tax. If you work out its
optimum, the best thing the car can do is crawl. A standstill scored 100, a
slow crawl scored 263, and driving at twenty metres a second only scored 189.
The steering tax was also net-negative exactly while the car was cornering.
So the old reward paid the agent to do almost nothing.

Now the honest part. Fixing that incentive was necessary, but it did not by
itself make PPO work. All three reward variants plateau around 0.1 laps, shown
on the right. [cut to shorten] What changed in practice is that agents drive
instead of crawling, and SAC completes two laps at 24.7 seconds. The
stability comes from the algorithm, not the reward.

*Transition:* "With a clean reward and protocol, here is what the two
algorithms actually do." *(after baselines, slide 11 → 12)*

---

## Slide 12 — Pilot Results: PPO vs SAC (≈ 55 s, 125 words)

Here are the first results, on a single track, under that frozen protocol.
The two curves tell the story.

PPO, in blue, climbs to about 0.34 laps and then falls back to 0.13 by two
million steps. It cannot hold onto what it learns. SAC, in green, completes
two full laps and then keeps getting faster, dropping its lap time from about
forty seconds down to under twenty-five.

Same reward, same budget, same normalization for both. The only difference is
how each algorithm uses its experience.

I want to be clear about scope: this is a single-track capability check, with
one seed. It tells us each algorithm can or cannot learn the task at all. It
is not the generalization result yet. That is the diversity grid we are
building.

---

## Slide 13 — Why PPO Falls Short (≈ 70 s, 165 words)

So why does PPO fall short? It comes down to how the two algorithms handle
experience.

PPO learns on-policy. It collects a batch, does one update, and throws the
batch away. So a single noisy update can wipe out its best policy, and there
is no memory to recover from it. SAC keeps a replay buffer, so it reuses past
experience and does not forget as easily.

Under a fixed step budget that difference is decisive. Replay lets SAC get
more learning out of every environment step. This is exactly what our
literature survey predicted about how the two reuse experience, so it confirms
the survey rather than surprising us.

One question we expect is: why not just give PPO more parallel environments?
Under a fixed budget, more environments do not add any data. You get fewer,
lower-variance updates, not more experience. [cut to shorten] So the honest
levers for PPO are a denser reward term for braking, and reporting the best
checkpoint instead of the final one.

---

## Slide 14 — Zero-Shot Results: RL vs Baselines (≈ 75 s, 175 words)

Now the zero-shot test, evaluating on tracks the policy never trained on. This
chart groups the four policies per track.

On synthetic tracks SAC is excellent. It completes an unseen synthetic track
on all five seeds. But move to the real circuits, Spielberg and Silverstone,
and SAC crashes. It only gets six to eleven percent around the lap. PPO is
essentially at random level there.

Now look at the orange bars. The gap-follower is a hand-written rule with no
learning at all, and it completes Spielberg. A trained policy losing to a
simple rule off-distribution is the generalization gap this whole project is
about.

The obvious question is whether real circuits are just special. They are not.
We generated a narrow synthetic track with the same spawn wall-distance as the
real circuits, and SAC fails there in the same way, at the same ten percent.
So this is a training-coverage problem. The policy memorized the distribution
it trained on. It is not something magic about real tracks.

---

## Slide 16 — Conclusions & Next Steps (≈ 78 s, 180 words)

To wrap up, three points and the plan.

First, a capability check. Under one identical protocol on one track, SAC
completes laps and keeps improving, while PPO peaks early and then forgets.
That is an on-policy instability under our protocol, not a setup that favours
one side.

Second, the gap. Zero-shot, our RL policies lose to a no-learning baseline on
real circuits. Because that failure reproduces on a narrow synthetic track, we
know it is a coverage problem. The policy memorized its training distribution.

Third, an open question we are actively testing. Does training a million steps
on one track overfit? We saved checkpoints every hundred thousand steps to
check whether earlier ones generalize better.

The fix directions follow directly: run the diversity grid at one, five,
twenty, and a hundred tracks, add real-like geometry to the generator, and
report the best checkpoint. [cut to shorten]

To be clear about status: these are single-track, single-seed pilots. The full
grid is our next step.

---

## Timing summary

| slide | complete | trimmed (drop `[cut]` lines) |
|---|---|---|
| 8 Protocol | 65 s | 55 s |
| 10 Reward | 85 s | 65 s |
| 12 PPO vs SAC | 55 s | 50 s |
| 13 Why PPO | 70 s | 55 s |
| 14 Zero-shot | 75 s | 70 s |
| 16 Conclusions | 78 s | 65 s |
| **total** | **~7.1 min** | **~5.1 min** |

## How long the method + results section *should* be

Ignoring the 8-minute cap: this section is the core of a
methodology-and-results presentation, so it deserves the majority of the
talk. The natural length that does the story justice is **about 6 to 7
minutes** — roughly protocol 1 min, reward 1.5 min, PPO vs SAC 1 min, why PPO
1 min, zero-shot 1.5 min, conclusions 1 min. That pace lets you pause on the
two figures (the PPO-vs-SAC curves and the zero-shot chart) instead of racing
past them, and it leaves room to answer questions well, which is where marks
were lost last time.

Within the 8-minute total, something has to give:
- If the literature section stays (~1.5 min), the second half has to compress
  to about **4.5 to 5 minutes** — use the trimmed column above.
- If the literature section is cut to a single 30-second slide, the second
  half can run at about **5.5 to 6 minutes**, close to its natural length,
  which is the better trade for this presentation.

Recommendation: keep the second half near its complete form and buy the time
from the literature section, since Presentation 2 is graded on methodology,
progress, results, and answering questions — all of which live in this half.
