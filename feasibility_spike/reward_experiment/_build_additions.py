"""Builds ONLY the slides WE worked on (results, our analysis, methodology
refinements, conclusion), in the v2 visual style, to be INSERTED into the
existing deck. Does not reproduce or touch anyone else's slides."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
import os

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "slides_assets")
prs = Presentation()
prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
BG=RGBColor(0x0E,0x1A,0x2B); CARD=RGBColor(0x16,0x24,0x3A); BORDER=RGBColor(0x2A,0x3B,0x55)
WHITE=RGBColor(0xFF,0xFF,0xFF); LIGHT=RGBColor(0xC3,0xCA,0xD6); GREEN=RGBColor(0x2E,0xCB,0x86); BLUE=RGBColor(0x3B,0x82,0xF6)
FONT="Poppins"

def setbg(s): f=s.background.fill; f.solid(); f.fore_color.rgb=BG
def title(s,txt,size=32,top=0.4):
    tb=s.shapes.add_textbox(Inches(0.65),Inches(top),Inches(12.0),Inches(1.0)); tf=tb.text_frame; tf.word_wrap=True
    p=tf.paragraphs[0]; r=p.add_run(); r.text=txt; r.font.size=Pt(size); r.font.bold=True; r.font.color.rgb=WHITE; r.font.name=FONT
    bar=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(0.68),Inches(top+0.92),Inches(1.5),Inches(0.07))
    bar.fill.solid(); bar.fill.fore_color.rgb=GREEN; bar.line.fill.background(); bar.shadow.inherit=False
def subhead(s,txt,left,top,size=16,width=6.0,color=BLUE):
    tb=s.shapes.add_textbox(Inches(left),Inches(top),Inches(width),Inches(0.5)); p=tb.text_frame.paragraphs[0]
    r=p.add_run(); r.text=txt; r.font.size=Pt(size); r.font.bold=True; r.font.color.rgb=color; r.font.name=FONT
def card(s,left,top,width,height,fill=CARD):
    c=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(left),Inches(top),Inches(width),Inches(height))
    c.fill.solid(); c.fill.fore_color.rgb=fill; c.line.color.rgb=BORDER; c.line.width=Pt(0.75); c.shadow.inherit=False
    try: c.adjustments[0]=0.05
    except Exception: pass
    return c
def bullets(s,items,top=1.55,size=16.5,left=0.75,width=12.0,height=5.4):
    tb=s.shapes.add_textbox(Inches(left),Inches(top),Inches(width),Inches(height)); tf=tb.text_frame; tf.word_wrap=True
    for i,(txt,lvl) in enumerate(items):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.space_after=Pt(8)
        r1=p.add_run(); r1.text=("›  " if lvl==0 else "      –  "); r1.font.color.rgb=GREEN if lvl==0 else LIGHT; r1.font.bold=True; r1.font.name=FONT; r1.font.size=Pt(size)
        r2=p.add_run(); r2.text=txt; r2.font.color.rgb=LIGHT; r2.font.name=FONT; r2.font.size=Pt(size-(1.5 if lvl else 0))
def body(s,txt,left,top,width,size=13.5,color=LIGHT,height=3.2):
    tb=s.shapes.add_textbox(Inches(left),Inches(top),Inches(width),Inches(height)); tf=tb.text_frame; tf.word_wrap=True
    p=tf.paragraphs[0]; r=p.add_run(); r.text=txt; r.font.size=Pt(size); r.font.color.rgb=color; r.font.name=FONT
def tag(s,txt):
    tb=s.shapes.add_textbox(Inches(0.65),Inches(7.05),Inches(12),Inches(0.4)); p=tb.text_frame.paragraphs[0]
    r=p.add_run(); r.text=txt; r.font.size=Pt(10); r.font.italic=True; r.font.color.rgb=GREEN; r.font.name=FONT
def img(s,path,left,top,width):
    if os.path.exists(path): s.shapes.add_picture(path,Inches(left),Inches(top),width=Inches(width))
def notes(s,t): s.notes_slide.notes_text_frame.text=t
def S(): s=prs.slides.add_slide(BLANK); setbg(s); return s

# --- A: methodology refinements (OURS) ---
s=S(); title(s,"Evaluation Protocol — Our Refinements")
bullets(s,[("Splits: TRAIN synthetic · VALIDATION held-out synthetic (checkpoint choice) · TEST 23 real circuits (zero-shot).",0),
("Metric: graded fractional laps (not 0/1 completion, which is a 0% floor) + completion + lap time + crash.",0),
("Selection: BEST checkpoint on validation (the final policy is a noisy estimator).",0),
("Spawn: eval on the centreline + pinned seed — matches training, reproducible (see spawn slide).",0),
("Frozen & identical for PPO/SAC: reward, crash penalty, VecNormalize (obs-only), SB3 defaults. Compute: PPO ~40 min/1M, SAC ~3.5-4 h/1M (CPU).",0)],size=15.5)
tag(s,"PLACEMENT: insert right AFTER your 'Datasets & Evaluation Protocol' slide.")
notes(s,"OURS (Aolin/Desmond). WHAT WE DID: added validation split; graded-laps metric; best-checkpoint rule; centreline+pinned-seed spawn. RUBRIC: methodology (1) + dataset (1) + experimental design.")

# --- B: reward crawl (OURS, updated) ---
s=S(); title(s,"Change: The Reward's Optimum Was a Crawl")
bullets(s,[("Old reward paid the SAME distance whether fast or slow → slower was strictly better.",0),
("Measured returns: standing still 100.0, a 0.25 m/s crawl 120.2, 20 m/s only 80.9.",0),
("Not undertraining — the agent had converged on the reward's TRUE optimum. More steps cannot escape it.",0),
("Fix: reward metres of progress along the centreline. Standing still now scores 0.00; faster is better.",0)])
tag(s,"PLACEMENT: this is your existing reward-diagnosis slide — REPLACE it with this updated version.")
notes(s,"OURS (Aolin). WHAT WE DID: measured old-reward returns (100/120/81), replaced with the frozen progress reward. RUBRIC: changes (3pts).")

# --- C: spawn fix (NEW, OURS) + fig ---
s=S(); title(s,"Change: Train/Eval Spawn Mismatch (Our Fix)")
bullets(s,[("Root cause: default spawns on the RACELINE; synthetic tracks have none, so training used the CENTRELINE.",0),
("Eval spawned ~0.8 m off-centre, ~0.3 m from the kerb; a ~1 m random start made one episode a coin-flip.",0),
("Fix: eval on the centreline + pinned seed. Verified the seed propagates through VecNormalize.",0),
("A fairness + reproducibility fix — decisive for borderline policies.",0)],top=1.55,width=6.2,size=15)
img(s,os.path.join(A,"fig_spawn_fix.png"),left=6.95,top=2.15,width=6.05)
tag(s,"PLACEMENT: NEW — insert as 'Bug 4' after the reward slide (or next to your bugs slide).")
notes(s,"NEW (ours). WHAT WE DID: traced raceline fallback, switched eval to centreline + pinned seed, verified seed->VecNormalize (std->0). Honest: fairness fix, not a performance rescue. RUBRIC: changes (3pts).")

# --- D: PPO vs SAC results (NEW, OURS) + fig ---
s=S(); title(s,"Pilot Results: PPO vs SAC (single track, 1M verification run)")
img(s,os.path.join(A,"fig_ppo_vs_sac.png"),left=0.85,top=1.55,width=11.6)
bullets(s,[("SAC completes 2 laps and gets faster (lap time 40.3 → 24.7 s); PPO reaches ~0.34 laps then DEGRADES to ~0.10, reward oscillates +37..-19. Why: PPO is on-policy and forgets its best policy; SAC's replay buffer retains it. Pilots n=1 — diagnostic, not the grid.",0)],top=5.7,size=13.5,width=12.2)
tag(s,"PLACEMENT: NEW — insert in the RESULTS section (after baselines).")
notes(s,"NEW (ours). WHAT WE DID: segmented training + deterministic per-checkpoint eval; reproducible (bit-identical re-run). BE PRECISE: single-track capability, not the generalization result. RUBRIC: results (3pts).")

# --- E: scoring flip (NEW, OUR ANALYSIS) ---
s=S(); title(s,"Our Analysis: The Scoring Rule Flips the Winner")
bullets(s,[("Score by the FINAL policy → PPO 'wins' (0.35 > 0.10 laps).",0),
("Score by the BEST checkpoint → SAC 'wins' (2 laps ≫ 0.35).",0),
("The final policy is a noisy, pessimistic estimator — so the rule matters.",0),
("Decision: freeze BEST-checkpoint-on-validation as the selection rule for the whole grid.",0)])
tag(s,"PLACEMENT: NEW — insert right after the PPO-vs-SAC results slide.")
notes(s,"NEW (ours). WHAT WE DID: observed the flip; adopted best-checkpoint-on-validation. RUBRIC: methodology + results.")

# --- F: conclusions & next steps (OUR CONCLUSION) ---
s=S(); title(s,"Conclusions & Next Steps")
bullets(s,[("This phase de-risked the pipeline: FOUR bugs found & fixed, reward + spawn protocol frozen, baselines set.",0),
("Verification pilots: SAC completes & is stable; PPO is capable early but can't hold a policy (no replay memory).",0),
("The scoring rule can flip the winner → we fixed best-checkpoint-on-validation before the grid.",0),
("Next: freeze ONE setup (merge reward, freeze crash penalty + 2M budget + spawn), run 2×4×≥3-seed grid with bootstrap CIs.",0),
("Honest status: current numbers are pilots (n=1); the diversity grid is not yet run.",0)])
tag(s,"PLACEMENT: REPLACE/augment your 'Timeline & Next Steps' slide with this (keeps your timeline, adds our conclusions).")
notes(s,"OURS. Conclusion of our work. RUBRIC: timeline (1pt) + honesty + progress.")

out=os.path.abspath(os.path.join(HERE,"..","..","..","Group4_OURSLIDES_additions.pptx"))
prs.save(out); print("saved:",out,"| our slides:",len(prs.slides._sldIdLst))
