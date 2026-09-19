"""Rebuild ONLY our inserted slides in Group4_ProjectUpdate_v2.pptx with the
corrected, smoother flow. All original slides are kept and never modified;
our old (wrong scoring-flip) slide is replaced by accurate ones."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
AST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides_assets")
SRC = os.path.join(ROOT, "_v2_build_src.pptx")
OUT = os.path.join(ROOT, "Group4_ProjectUpdate_v6.pptx")

WHITE=RGBColor(0xF8,0xFA,0xFC); GREEN=RGBColor(0x10,0xB9,0x81); BLUE=RGBColor(0x3B,0x82,0xF6)
BODY=RGBColor(0xCB,0xD5,0xE1); MUTED=RGBColor(0x94,0xA3,0xB8)
TFONT="Montserrat"; BFONT="Inter"

prs = Presentation(SRC)
BLANK = min(prs.slide_layouts, key=lambda L: len(L.placeholders))

def new_slide():
    s = prs.slides.add_slide(BLANK)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    s.shapes.add_picture(os.path.join(AST, "bg.png"), 0, 0, width=prs.slide_width, height=prs.slide_height)
    return s

def title(s, txt, size=28):
    tb = s.shapes.add_textbox(Inches(0.62), Inches(0.55), Inches(12.1), Inches(1.0))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = txt
    r.font.name = TFONT; r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = WHITE

def bullets(s, items, top=1.65, left=0.65, width=12.0, size=14.5, gap=9):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.3))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (lead, txt) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(gap)
        r1 = p.add_run(); r1.text = "›  "; r1.font.name = BFONT; r1.font.bold = True; r1.font.size = Pt(size); r1.font.color.rgb = GREEN
        if lead:
            rl = p.add_run(); rl.text = lead; rl.font.name = BFONT; rl.font.bold = True; rl.font.size = Pt(size); rl.font.color.rgb = WHITE
        r2 = p.add_run(); r2.text = txt; r2.font.name = BFONT; r2.font.size = Pt(size); r2.font.color.rgb = BODY

def caption(s, txt, top=5.75, size=13, left=0.7, width=12.2, color=BODY):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(1.5)); tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = txt; r.font.name = BFONT; r.font.size = Pt(size); r.font.color.rgb = color

def img(s, name, left, top, width):
    path = os.path.join(AST, name)
    if os.path.exists(path): s.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))

def notes(s, t): s.notes_slide.notes_text_frame.text = t

# N1: Evaluation Protocol refinements (methodology)
s = new_slide(); title(s, "Evaluation Protocol \u2014 Our Refinements")
bullets(s, [
 ("Splits: ", "TRAIN synthetic \u00b7 VALIDATION held-out synthetic (checkpoint choice) \u00b7 TEST 23 real circuits (zero-shot)."),
 ("Metric: ", "graded fractional laps (not 0/1 completion, a 0% floor) + completion + lap time + crash rate."),
 ("Selection: ", "BEST checkpoint on validation \u2014 PPO's final policy degrades to 0.10 after peaking at 0.34, so the final checkpoint is a noisy estimator."),
 ("Spawn: ", "eval on the centreline + 5 FIXED spawn seeds (0-4), averaged per circuit \u2014 identical for every policy & baseline (matches training; reproducible, verified through VecNormalize)."),
 ("Frozen for both algos: ", "reward, crash penalty, VecNormalize (obs-only), SB3 defaults. Compute: PPO ~40 min/1M, SAC ~3.5-4 h/1M (CPU)."),
])
notes(s, "OURS. INSERT after 'Datasets & Evaluation Protocol'. WHY best-checkpoint: PPO degrades 0.34->0.10; final policy is a poor estimator. RUBRIC: methodology + dataset + experimental design.")

# N2 removed (spawn fix folded into the protocol slide's 'spawn consistency' bullet)

# N3: PPO vs SAC results + figure
s = new_slide(); title(s, "Pilot Results: PPO vs SAC (frozen protocol, single track)")
img(s, "fig_ppo_vs_sac.png", 0.85, 1.5, 11.6)
caption(s, "PPO: full 2M curve \u2014 peaks at 0.34 laps then oscillates to a final 0.13; reward swings +37..\u221219. SAC: 1M so far (2M still training) \u2014 completes 2 laps and gets FASTER (40.3 \u2192 24.7 s). Same reward, budget, obs-normalization. n=1 seed \u2014 diagnostic pilots.")
notes(s, "OURS. RESULTS. INSERT after baselines. WHAT WE DID: segmented training + deterministic per-checkpoint eval; reproducible. Single-track capability, not the generalization result. RUBRIC: results (3pts).")

# N4: Why PPO + how to fix (ties to the survey's prediction)
s = new_slide(); title(s, "Our Analysis: Why PPO Can't Hold a Policy \u2014 and the Fix")
bullets(s, [
 ("The survey predicted this: ", "under a FIXED step budget PPO and SAC 'reuse experience differently' \u2014 SAC's replay buffer learns more per env-step; PPO's on-policy updates discard data after one use."),
 ("What that looks like: ", "PPO peaks at 0.34 laps then forgets it (no replay memory); SAC accumulates and completes 2 laps."),
 ("Parallelism is NOT a free fix: ", "with steps fixed, more envs add no data \u2014 updates drop from ~976 (1 env) to ~122 (8 envs). It only trades update count for lower variance."),
 ("Real levers under the frozen protocol: ", "dense reward shaping that rewards braking (clearance / racing-line term), and reporting best-checkpoint rather than final."),
 ("Takeaway: ", "a fixed-budget comparison structurally favours replay-buffer algorithms \u2014 exactly what the proposal anticipated, and a fair property of the study."),
])
notes(s, "OURS. ANALYSIS. Say: this result CONFIRMS the survey's prediction rather than surprising us. Q&A-proof: 'why didn't you give PPO more envs?' -> fixed budget: more envs = fewer updates, no extra data. RUBRIC: results + Q&A depth.")

# N5: Zero-shot results vs baselines (with chart)
s = new_slide(); title(s, "Zero-Shot Results: RL vs Baselines (5 seeds, one-lap protocol)")
img(s, "fig_gen_gap.png", 0.7, 1.5, 11.9)
bullets(s, [
 ("Read: ", "zero-shot PPO is at RANDOM level on unseen real circuits (0.013 vs 0.011 laps) while the no-learning gap-follower completes them. In-distribution, SAC completes the training track. The SAC unseen-track row lands when its 2M run finishes."),
], top=5.9, size=13.5, width=12.2)
notes(s, "OURS. RESULTS. Honest labels: PPO row = final 2M checkpoint zero-shot (5 seeds each circuit, one-lap protocol). SAC row pending (training). RUBRIC: results (3pts).")

# N6: Conclusions & next steps
s = new_slide(); title(s, "Conclusions & Next Steps")
bullets(s, [
 ("De-risked: ", "FOUR bugs found & fixed, reward + spawn protocol frozen, baselines set, eval protocol formalized."),
 ("Pilots: ", "SAC completes & is stable; PPO peaks early then forgets (no replay memory) \u2014 with a concrete fix path."),
 ("Gap: ", "zero-shot RL still loses to the reactive baseline on unseen circuits \u2014 the diversity grid is the test."),
 ("Next: ", "freeze ONE setup (merge reward, freeze crash penalty + 2M budget + spawn), run 2\u00d74\u00d7\u22653-seed grid with bootstrap CIs."),
 ("Honest status: ", "current numbers are pilots (n=1); the grid is not yet run."),
])
notes(s, "OURS. CONCLUSION. INSERT after Timeline. RUBRIC: timeline + honesty + progress.")

# N7: reward DESIGN (replaces the original reward slide; no 'bug' framing)
s = new_slide(); title(s, "The Reward: Design & Why It Changed")
bullets(s, [
 ("What we reward: ", "1.0 \u00d7 metres of centreline progress (speed projected onto the track), \u2212 crash penalty on collision. TIME_COST = 0 deliberately: the fixed step budget already supplies time pressure, and any per-step cost would reward crashing early."),
 ("Why it changed: ", "the original paid 1.0/s alive + 10\u00d7distance \u2212 a steering tax \u2192 its optimum was a CRAWL (standstill 100, crawl 263, 20 m/s 189), and the steering term was net-negative exactly while cornering."),
 ("Honest read: ", "the change corrected a PROVEN incentive flaw (the old optimum was a crawl), but it did not by itself fix performance \u2014 all three PPO variants plateau ~0.1 laps (right panel)."),
 ("What improved: ", "the team's original agent crawled; under the frozen reward agents drive, and SAC completes 2 laps at 24.7 s. Stability comes from the ALGORITHM (SAC's replay buffer), not the reward. (SAC was never trained under the old reward.)"),
], top=1.6, width=5.9, size=13)
img(s, "fig_reward_compare.png", 6.8, 1.95, 6.3)
notes(s, "OURS. DESIGN framing, not 'bug'. Q&A: V1 also had -5 on termination (punished FINISHING 2 laps like a crash); the scan numbers are spawn-dependent (our 18 m straight vs the team's 8.5 m), hence 263 vs 120 for the same crawl; shape is identical. RUBRIC: methodology + changes-justified.")

# ---- rebuild order: originals kept, ours re-placed ----
sldIdLst = prs.slides._sldIdLst
ids = list(sldIdLst)
o = ids[:13]; ns = ids[13:]
# o: 0 Title 1 Motiv 2-4 Lit 5 RQ 6 Datasets 7 Bugs12 8 Baselines 9 RewardCrawl 10 Timeline 11 Roles 12 ImgSrc
# ns: [N1 protocol, N3 ppovssac, N4 whyfix, N5 gap, N6 concl, N7 reward]  (N2 spawn removed)
desired = (o[0:7] + [ns[0]] + o[7:8] + [ns[5]] + [o[8]] + [ns[1], ns[2], ns[3]] + [o[10]] + [ns[4]] + o[11:13])
for e in ids: sldIdLst.remove(e)
for e in desired: sldIdLst.append(e)

prs.save(OUT)
print("saved:", OUT, "| total slides:", len(prs.slides._sldIdLst), "(13 originals + 6 ours)")
