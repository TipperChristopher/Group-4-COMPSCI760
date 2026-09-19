from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
import os

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "slides_assets")
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

BG     = RGBColor(0x0E, 0x1A, 0x2B)
CARD   = RGBColor(0x16, 0x24, 0x3A)
BORDER = RGBColor(0x2A, 0x3B, 0x55)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT  = RGBColor(0xC3, 0xCA, 0xD6)
GREEN  = RGBColor(0x2E, 0xCB, 0x86)
BLUE   = RGBColor(0x3B, 0x82, 0xF6)
FONT   = "Poppins"


def setbg(s):
    f = s.background.fill; f.solid(); f.fore_color.rgb = BG

def title(s, txt, size=32, top=0.4):
    tb = s.shapes.add_textbox(Inches(0.65), Inches(top), Inches(12.0), Inches(1.0))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = txt
    r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = FONT
    bar = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.68), Inches(top + 0.92), Inches(1.5), Inches(0.07))
    bar.fill.solid(); bar.fill.fore_color.rgb = GREEN; bar.line.fill.background(); bar.shadow.inherit = False

def card(s, left, top, width, height, fill=CARD):
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    c.fill.solid(); c.fill.fore_color.rgb = fill
    c.line.color.rgb = BORDER; c.line.width = Pt(0.75); c.shadow.inherit = False
    try: c.adjustments[0] = 0.05
    except Exception: pass
    return c

def subhead(s, txt, left, top, size=18, width=6.0, color=BLUE):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.5))
    p = tb.text_frame.paragraphs[0]; r = p.add_run(); r.text = txt
    r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = color; r.font.name = FONT
    return tb

def bullets(s, items, top=1.55, size=16.5, left=0.75, width=12.0, height=5.4):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (txt, lvl) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run(); r1.text = ("›  " if lvl == 0 else "      –  ")
        r1.font.color.rgb = GREEN if lvl == 0 else LIGHT; r1.font.bold = True; r1.font.name = FONT; r1.font.size = Pt(size)
        r2 = p.add_run(); r2.text = txt
        r2.font.color.rgb = LIGHT; r2.font.name = FONT; r2.font.size = Pt(size - (1.5 if lvl else 0))
    return tb

def body(s, txt, left, top, width, size=13.5, color=LIGHT, height=3.2):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = txt
    r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = FONT
    return tb

def img(s, path, left, top, width):
    if os.path.exists(path):
        s.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))

def notes(s, t):
    s.notes_slide.notes_text_frame.text = t

def S():
    s = prs.slides.add_slide(BLANK); setbg(s); return s


# 1 TITLE (kept)
s = S()
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.3), Inches(11.7), Inches(2.4)); tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; r = p.add_run(); r.text = "Autonomous Racing:"; r.font.size = Pt(44); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = FONT
p = tf.add_paragraph(); r = p.add_run(); r.text = "Zero-Shot Generalization"; r.font.size = Pt(44); r.font.bold = True; r.font.color.rgb = GREEN; r.font.name = FONT
p = tf.add_paragraph(); r = p.add_run(); r.text = "Group Presentation 2 — Project Update on Methodology & Results"; r.font.size = Pt(19); r.font.color.rgb = LIGHT; r.font.name = FONT
p = tf.add_paragraph(); r = p.add_run(); r.text = "Christopher Tipper · Yi Wei · Zihang Zhang · Aolin Yang · Desmond Li"; r.font.size = Pt(14); r.font.color.rgb = LIGHT; r.font.name = FONT
p = tf.add_paragraph(); r = p.add_run(); r.text = "github.com/TipperChristopher/Group-4-COMPSCI760"; r.font.size = Pt(13); r.font.color.rgb = BLUE; r.font.name = FONT
notes(s, "KEPT (shared). WHAT WE SAY: do RL agents learn to DRIVE or MEMORIZE tracks? PPO vs SAC on zero-shot transfer.")

# 2 MOTIVATION (kept - Christopher)
s = S(); title(s, "Recap: The Overfitting Dilemma")
subhead(s, "Memorization vs. Competence", 0.75, 1.5)
bullets(s, [("Deep RL excels at continuous-control racing, but policies frequently MEMORIZE specific training tracks.", 0),
            ("On unseen geometries performance degrades rapidly — the 'generalization gap'.", 0),
            ("Prior literature often evaluates on the SAME tracks it trained on — testing memorization, not driving intelligence.", 0)], top=2.1)
body(s, "[Keep your original F1TENTH car photo here.]", 8.2, 4.6, 4.4, 12, LIGHT)
notes(s, "KEPT (Christopher's motivation). Re-insert the original car photo from v2.")

# 3 LIT SURVEY - evidence base (kept - Victory/Grant)
s = S(); title(s, "Literature Survey")
subhead(s, "Evidence base", 0.75, 1.5)
card(s, 0.9, 2.3, 3.4, 2.6); c = s.shapes.add_textbox(Inches(0.9), Inches(2.8), Inches(3.4), Inches(1.6)); tf = c.text_frame
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = "72"; r.font.size = Pt(48); r.font.bold = True; r.font.color.rgb = GREEN; r.font.name = FONT
p = tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = "records identified"; r.font.size = Pt(15); r.font.color.rgb = LIGHT; r.font.name = FONT
card(s, 4.7, 2.3, 3.4, 2.6); c = s.shapes.add_textbox(Inches(4.7), Inches(2.8), Inches(3.4), Inches(1.6)); tf = c.text_frame
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = "30"; r.font.size = Pt(48); r.font.bold = True; r.font.color.rgb = GREEN; r.font.name = FONT
p = tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = "papers selected"; r.font.size = Pt(15); r.font.color.rgb = LIGHT; r.font.name = FONT
body(s, "Focus: generalization and fair PPO / SAC evaluation.", 8.5, 3.3, 4.2, 16, LIGHT)
body(s, "Source: Group 4 Literature Survey, Fig. 1", 0.75, 6.7, 8, 11, LIGHT)
notes(s, "KEPT (Victory/Grant's literature survey).")

# 4 LIT SURVEY - key findings (kept - Victory/Grant)
s = S(); title(s, "Literature Survey: Key Findings")
card(s, 1.4, 2.6, 10.5, 1.15); body(s, "Diversity changes experience.", 1.9, 2.85, 9.5, 22, WHITE)
card(s, 1.4, 4.1, 10.5, 1.15); body(s, "PPO and SAC reuse it differently.", 1.9, 4.35, 9.5, 22, WHITE)
body(s, "Source: Group 4 Literature Survey, Sections II–VII and Fig. 2", 0.75, 6.7, 9, 11, LIGHT)
notes(s, "KEPT (Victory/Grant). These two findings are exactly what our study operationalizes — diversity (independent variable) and algorithm (PPO vs SAC).")

# 5 LIT SURVEY - method refinements (kept - connects lit -> our design)
s = S(); title(s, "Literature Survey: Method Refinements")
hdr_l, hdr_r = 0.7, 6.9
subhead(s, "REVIEW FINDING", hdr_l, 1.55, 15, color=LIGHT)
subhead(s, "REFINEMENT IN OUR DESIGN", hdr_r, 1.55, 15, color=GREEN)
rows = [("Exposure & tuning affect comparisons", "Fixed 2M environment steps · matched tuning effort"),
        ("Diversity needs a controlled comparison", "PPO vs SAC · 1 / 5 / 20 / 100 tracks"),
        ("Generalization needs independent evaluation", "Held-out real F1 circuits · multiple seeds")]
y = 2.15
for lf, rf in rows:
    card(s, hdr_l, y, 5.9, 1.15); body(s, lf, hdr_l + 0.25, y + 0.28, 5.4, 15, LIGHT)
    card(s, hdr_r, y, 5.7, 1.15, fill=RGBColor(0x12, 0x2A, 0x22)); body(s, rf, hdr_r + 0.25, y + 0.28, 5.2, 15, WHITE)
    y += 1.35
body(s, "These findings shaped our current experimental design (see next slides).", 0.75, 6.65, 11, 13, GREEN)
notes(s, "KEPT (Victory/Grant) — THIS is the bridge from literature to our methodology. Say: 'the survey's three refinements are implemented directly in the protocol on the next slides.'")

# 6 RESEARCH QUESTION (kept, light update to 2M)
s = S(); title(s, "Core Research Question")
bullets(s, [("Under a FIXED environment-step budget, does zero-shot generalization depend more on:", 0),
            ("ALGORITHM choice (PPO vs SAC)?", 1),
            ("or TRAINING-TRACK DIVERSITY (1 / 5 / 20 / 100 tracks)?", 1),
            ("Implementing the survey's refinements: fixed 2M-step budget, matched (no per-algorithm) tuning, held-out real circuits.", 0),
            ("Diversity is our primary independent variable; algorithm is the second.", 0)])
notes(s, "KEPT/updated. Explicit callback to slide 5's refinements. RUBRIC: objectives (2pts).")

# 7 METHODOLOGY / PROTOCOL (UPDATED - ours)
s = S(); title(s, "Methodology: One Frozen Protocol")
bullets(s, [("Reward: progress along the centreline (frozen); crash penalty frozen identically across all cells.", 0),
            ("Budget: fixed 2M env-steps per cell (these verification pilots use 1M; learning saturates ~500k).", 0),
            ("VecNormalize on OBSERVATIONS only (reward left raw = the dependent variable). Identical for both algos.", 0),
            ("No per-algorithm tuning — stock SB3 defaults. Single env + stratified track-pool sampler.", 0),
            ("TRAIN synthetic · VALIDATION held-out synthetic (checkpoint choice) · TEST 23 real circuits (zero-shot).", 0),
            ("Metric: graded fractional laps + completion + lap time + crash. Compute (CPU): PPO ~40 min/1M, SAC ~3.5-4 h/1M.", 0)], size=15.5)
notes(s, "UPDATED (ours). WHAT WE DID: audited env in code; graded-laps metric (completion rate is a 0% floor); added the validation split; best-checkpoint-on-validation. RUBRIC: methodology (1pt) + dataset (1pt) + experimental design.")

# 8 FOUR BUGS (UPDATED)
s = S(); title(s, "Project Evolution: Four Bugs That Would Have Voided the Grid")
card(s, 0.6, 1.55, 6.05, 5.3); subhead(s, "Bug 1 — Updates scaled with N", 0.9, 1.75, 16)
body(s, "One env per track, so updates = 2M / n_envs (2,000,000 at 1 track but only 20,000 at 100). Fixed: single env + track-pool sampler; verified identical update counts.", 0.9, 2.25, 5.5, 13)
subhead(s, "Bug 3 — Reward optimum was a crawl", 0.9, 4.05, 16)
body(s, "The old reward's best policy was to barely move — measured returns: still 100 · crawl 120 · 20 m/s 81 (next slide).", 0.9, 4.55, 5.5, 13)
card(s, 6.85, 1.55, 5.9, 5.3); subhead(s, "Bug 2 — Shared ray-caster", 7.15, 1.75, 16)
body(s, "RaceCar.scan_simulator is a process-wide class attribute, so multi-track runs were single-track. Fixed by the single-env change; verified 20/20 tracks visited.", 7.15, 2.25, 5.4, 13)
subhead(s, "Bug 4 — Train/eval spawn mismatch", 7.15, 4.05, 16)
body(s, "Eval spawned on the raceline (near the kerb) while training used the centreline (slide after).", 7.15, 4.55, 5.4, 13)
notes(s, "UPDATED. Count now consistent (was 'two' vs 'four'). Bugs 1-2 team; bugs 3-4 ours. RUBRIC: changes (3pts).")

# 9 REWARD CRAWL (UPDATED - Bug 3 detail)
s = S(); title(s, "Change: The Reward's Optimum Was a Crawl")
bullets(s, [("Old reward paid the SAME distance whether fast or slow → slower was strictly better.", 0),
            ("Measured returns: standing still 100.0, a 0.25 m/s crawl 120.2, 20 m/s only 80.9.", 0),
            ("Not undertraining — the agent had converged on the reward's TRUE optimum. More steps cannot escape it.", 0),
            ("Fix: reward metres of progress along the centreline. Standing still now scores 0.00; faster is better.", 0)])
notes(s, "UPDATED (ours). WHAT WE DID: measured old-reward returns (100/120/81), replaced with progress reward. RUBRIC: changes (3pts).")

# 10 SPAWN FIX (NEW - ours) + fig
s = S(); title(s, "Change: Train/Eval Spawn Mismatch (Our Fix)")
bullets(s, [("Root cause: default spawns on the RACELINE; synthetic tracks have none, so training used the CENTRELINE.", 0),
            ("Eval spawned ~0.8 m off-centre, ~0.3 m from the kerb; a ~1 m random start made one episode a coin-flip.", 0),
            ("Fix: eval on the centreline + pinned seed. Verified the seed propagates through VecNormalize.", 0),
            ("A fairness + reproducibility fix — decisive for borderline policies.", 0)], top=1.55, width=6.2, size=15)
img(s, os.path.join(A, "fig_spawn_fix.png"), left=6.95, top=2.15, width=6.05)
notes(s, "NEW (ours). WHAT WE DID: traced raceline fallback, switched eval to centreline + pinned seed, verified seed->VecNormalize (std->0). Honest: fairness fix, not a performance rescue. RUBRIC: changes (3pts).")

# 11 BASELINES (kept/updated - Desmond)
s = S(); title(s, "Initial Results: Baselines (Floor & Ceiling)")
bullets(s, [("Two reference points, SAME protocol as the RL agents.", 0),
            ("Floor — Random completes 0 of 25 episodes.", 0),
            ("Ceiling — classical Follow-the-Gap completes ~68% of circuits.", 0),
            ("Stable limitation: gap-follower fails Silverstone on EVERY seed, crashing at the same hairpin (~26 cm).", 0),
            ("We corroborated it on our harness: finishes Spielberg every seed, crashes Silverstone every seed.", 0)])
notes(s, "KEPT/updated (Desmond). CAVEAT: 68% used a retuned gap-follower (safe_threshold=20); ours uses default — same story. RUBRIC: results (3pts).")

# 12 PPO vs SAC (NEW - results) + fig + memory
s = S(); title(s, "Pilot Results: PPO vs SAC (single track, 1M verification run)")
img(s, os.path.join(A, "fig_ppo_vs_sac.png"), left=0.85, top=1.55, width=11.6)
bullets(s, [("SAC completes 2 laps and gets faster (lap time 40.3 → 24.7 s); PPO reaches ~0.34 laps then DEGRADES to ~0.10, reward oscillates +37..-19. Why: PPO is on-policy and forgets its best policy; SAC's replay buffer retains it. Pilots n=1 — diagnostic, not the grid.", 0)], top=5.7, size=13.5, width=12.2)
notes(s, "NEW (ours). WHAT WE DID: segmented training + deterministic per-checkpoint eval; reproducible (bit-identical re-run). BE PRECISE: single-track capability, not the generalization result. RUBRIC: results (3pts) + Q&A depth.")

# 13 SCORING FLIP (NEW - method finding)
s = S(); title(s, "Methodology Finding: The Scoring Rule Flips the Winner")
bullets(s, [("Score by the FINAL policy → PPO 'wins' (0.35 > 0.10 laps).", 0),
            ("Score by the BEST checkpoint → SAC 'wins' (2 laps ≫ 0.35).", 0),
            ("The final policy is a noisy, pessimistic estimator — so the rule matters.", 0),
            ("Decision: freeze BEST-checkpoint-on-validation as the selection rule for the whole grid.", 0)])
notes(s, "NEW (ours). WHAT WE DID: observed the flip, adopted best-checkpoint-on-validation. RUBRIC: method + results.")

# 14 TIMELINE (updated)
s = S(); title(s, "Timeline & Next Steps")
bullets(s, [("Done — research question locked; harness + track generation; FOUR pipeline bugs found & fixed; baselines; reward + spawn protocol; PPO/SAC verification pilots.", 0),
            ("Next — freeze ONE setup (merge progress reward to main, freeze crash penalty + budget + spawn), then run the main grid.", 0),
            ("Main grid — 2 algos × 4 diversity × ≥3 seeds at 2M steps; bootstrap confidence intervals; generalization-gap plots.", 0),
            ("Honest status: current numbers are pilots (n=1); the diversity grid is not yet run.", 0)])
notes(s, "UPDATED. RUBRIC: timeline (1pt) + honesty.")

# 15 ROLES + Q&A (kept - Victory/Grant unchanged)
s = S(); title(s, "Team Roles & Questions")
bullets(s, [("Christopher — literature synthesis, motivation framing, report assembly.", 0),
            ("Desmond — training pipeline, track-pool sampler, baselines, evaluation protocol.", 0),
            ("Victory — literature survey, method refinements, feasibility.", 0),
            ("Aolin — reward function, spawn configuration, deterministic seeding.", 0),
            ("Grant — literature survey, statistical analysis (CIs), figures.", 0),
            ("Citations: literature = Group 4 survey; images = foxglove.dev, trackdecals.com.", 0)], size=15)
body(s, "AI Acknowledgement: Generative AI was utilized for structuring themes, language polishing, and slide generation, adhering to course guidelines. All experimental design, code, and conclusions are our own.", 0.75, 6.5, 12.0, 11, LIGHT)
notes(s, "KEPT roles (Victory/Grant unchanged). Q&A PREP: (1) PPO unstable? no replay memory + single-env variance + safety-blind reward. (2) progress reward plateaus? rewards speed not braking under a step budget. (3) spawn a tuning knob? no — frozen for all cells. (4) 1M undertraining? no — saturates ~500k; grid uses 2M. (5) n? pilots n=1; grid >=3 seeds + CIs. (6) novel vs replicated? reward/bug/spawn ours; progress reward + gap-follower from literature. RUBRIC: roles(1)+cite(1)+answering(2).")

# 16 IMAGE SOURCES (kept)
s = S(); title(s, "Image Sources")
bullets(s, [("Foxglove — spotlight-building-a-championship-autonomous-f1-racecar-for-f1tenth (cars.webp). Source: foxglove.dev", 0),
            ("Track decals GP image. Source: trackdecals.com", 0),
            ("Result figures (PPO vs SAC, spawn fix) generated by Group 4 from our own experiment logs.", 0)], size=14)
notes(s, "KEPT + added our own figure attribution. RUBRIC: citation (1pt).")

out = os.path.abspath(os.path.join(HERE, "..", "..", "..", "Group4_ProjectUpdate_v3_draft.pptx"))
prs.save(out)
print("saved deck:", out, "| slides:", len(prs.slides._sldIdLst))
