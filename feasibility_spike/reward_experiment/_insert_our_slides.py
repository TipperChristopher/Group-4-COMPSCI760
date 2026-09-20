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
OUT = os.path.join(ROOT, "Group4_ProjectUpdate_FINAL2.pptx")

WHITE=RGBColor(0xF8,0xFA,0xFC); GREEN=RGBColor(0x10,0xB9,0x81); BLUE=RGBColor(0x3B,0x82,0xF6)
BODY=RGBColor(0xCB,0xD5,0xE1); MUTED=RGBColor(0x94,0xA3,0xB8)
TFONT="Montserrat"; BFONT="Inter"

prs = Presentation(SRC)
BLANK = min(prs.slide_layouts, key=lambda L: len(L.placeholders))

# --- targeted fixes to team slides (authorized): keep wording consistent with our narrative ---
def fix_team_text(replacements):
    for s in prs.slides:
        for sh in s.shapes:
            if not sh.has_text_frame:
                continue
            for para in sh.text_frame.paragraphs:
                for run in para.runs:
                    for old, new in replacements.items():
                        if old in run.text:
                            run.text = run.text.replace(old, new)

fix_team_text({
    "Found and fixed four bugs that would have voided the grid.":
        "Found and fixed two bugs, and redesigned the reward and spawn protocol, before the grid.",
})

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
caption(s, "PPO: peaks at 0.34 laps then falls to 0.13 by 2M. SAC: completes 2 laps and keeps getting faster (40.3 \u2192 24.7 s). Same reward, same budget, same normalization \u2014 the difference is how each algorithm uses experience. (Single-track pilots, one seed.)")
notes(s, "OURS. RESULTS. INSERT after baselines. WHAT WE DID: segmented training + deterministic per-checkpoint eval; reproducible. Single-track capability, not the generalization result. RUBRIC: results (3pts).")

# N4: Why PPO falls short + lap-time figure
s = new_slide(); title(s, "Why PPO Falls Short, and What Would Fix It")
bullets(s, [
 ("PPO forgets: ", "it learns on-policy and throws away each batch after one use, so a noisy update can erase its best policy. SAC keeps a replay buffer and does not."),
 ("A fixed budget helps SAC: ", "under a step budget, replay lets SAC learn more per env-step \u2014 exactly what our literature survey predicted about how the two reuse experience."),
 ("SAC learns controlled speed: ", "same track, same reward \u2014 lap time drops from 40.3 to 24.7 s while it keeps completing (right)."),
 ("If we wanted to help PPO: ", "more parallel envs do not add data under a fixed budget (just fewer, lower-variance updates), so the honest levers are a denser reward term for braking and reporting the best checkpoint."),
], top=1.6, width=6.3, size=13.5)
img(s, "fig_sac_laptime.png", 7.05, 2.3, 6.0)
notes(s, "OURS. ANALYSIS. Say: this result CONFIRMS the survey's prediction rather than surprising us. Q&A-proof: 'why didn't you give PPO more envs?' -> fixed budget: more envs = fewer updates, no extra data. RUBRIC: results + Q&A depth.")

# N5: Zero-shot results vs baselines (with chart)
s = new_slide(); title(s, "Zero-Shot Results: RL vs Baselines (5 seeds, one-lap protocol)")
img(s, "fig_gen_gap.png", 0.7, 1.5, 11.9)
bullets(s, [
 ("What this shows: ", "SAC trained on one synthetic track completes unseen synthetic tracks (track_2: 1.0 laps on all 5 seeds) but crashes on real circuits (0.06-0.11 laps) \u2014 and the same crash reproduces on a NARROW synthetic track with the same spawn scan. PPO is at random level on real circuits; the gap-follower completes Spielberg. A coverage problem, not real-track magic."),
], top=5.9, size=13.5, width=12.2)
notes(s, "OURS. RESULTS. SAC row = 1M checkpoint (paused at 1.2M; 1M->2M only improves lap time). RUBRIC: results (3pts).")

# N6: Conclusions & next steps (staged narrative)
s = new_slide(); title(s, "Conclusions & Next Steps")
bullets(s, [
 ("Capability check: ", "same protocol, one track \u2014 SAC completes laps and keeps improving; PPO peaks early then forgets. An on-policy instability under our frozen protocol, not a setup bias."),
 ("The gap: ", "zero-shot RL loses to the no-learning baseline on real circuits. The failure reproduces on a NARROW synthetic track with the same spawn scan \u2014 a training-coverage problem: the policy memorized its training distribution."),
 ("Open question: ", "does 1M steps on one track overfit? We are testing whether earlier checkpoints generalize better (saved every 100k)."),
 ("Fix directions: ", "the 1/5/20/100 diversity grid, real-like geometry in the generator (narrower tracks, sharper corners), and reporting best-checkpoint instead of final."),
 ("Honest status: ", "these are single-track, single-seed pilots; the grid is not yet run."),
])
notes(s, "OURS. CONCLUSION. INSERT after Timeline. RUBRIC: timeline + honesty + progress.")

# N7: reward DESIGN (replaces the original reward slide; no 'bug' framing)
s = new_slide(); title(s, "The Reward: Design & Why It Changed")
bullets(s, [
 ("What we reward: ", "1.0 \u00d7 metres of centreline progress (speed projected onto the track), \u2212 crash penalty on collision. TIME_COST = 0 deliberately: the fixed step budget already supplies time pressure, and any per-step cost would reward crashing early."),
 ("Why it changed: ", "the original paid 1.0/s alive + 10\u00d7distance \u2212 a steering tax \u2192 its optimum was a CRAWL (standstill 100, crawl 263, 20 m/s 189), and the steering term was net-negative exactly while cornering."),
 ("Honest read: ", "the change removed a proven incentive flaw (the old optimum was a crawl), but it did not by itself improve PPO \u2014 all three variants plateau around 0.1 laps (right)."),
 ("What it changed in practice: ", "the original agent crawled; under this reward agents drive, and SAC completes 2 laps at 24.7 s. Stability comes from the algorithm, not the reward. (SAC was never trained under the old reward.)"),
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
