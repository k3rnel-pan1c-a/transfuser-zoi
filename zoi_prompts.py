"""
ZOI VLM prompt templates for planning-relevance scoring.

Two prompts, both using compact JSON output (id + importance only) to avoid
token-cap truncation on frames with many marks:

  EXEMPLAR_PROMPT   — refinement of the compact prompt; adds one concrete
                      driving scenario per score level so the model has anchors.
                      Addresses Gemma 3's score-collapse-to-3 failure.

  EGOPATHCONFLICT_PROMPT — alternative framing: "Does this object conflict with
                      the ego vehicle's path right now?" rather than an
                      abstract importance scale. Forces sharper 0-vs-5
                      discrimination by grounding each level in a specific
                      spatial relationship to the ego trajectory.

Usage in run_vlm_multiframe.py:
    from zoi_prompts import EXEMPLAR_PROMPT, EGOPATHCONFLICT_PROMPT
    prompt = EXEMPLAR_PROMPT.format(object_list="\n".join(object_lines))
"""


# ---------------------------------------------------------------------------
# Prompt 1 — Exemplar-anchored compact prompt
#
# Keeps the same compact output format (id + importance, no reason/role text)
# but adds one concrete example per score level. The examples are designed to
# cover the two main failure modes observed:
#   - score collapse (Gemma 3 returns 3 for everything)
#   - distance blindness (near and far cars rated equally)
# ---------------------------------------------------------------------------
EXEMPLAR_PROMPT = """\
You are a planning-relevance annotator for an autonomous driving system.
The image is the EGO vehicle's front camera. Numbered green markers are already
placed on objects from the simulator's ground truth — you are NOT detecting objects.
For EACH numbered marker rate how planning-relevant that object is to the ego's
near-future driving decisions (braking, steering, yielding) over the next 2-3 seconds.

IMPORTANCE SCALE with one concrete example per level:
  5 = critical — a pedestrian has stepped off the curb directly into the ego lane 10 m ahead,
      or the traffic light at the intersection the ego is entering just turned red.
  4 = high — a cyclist has moved into the ego lane 25 m ahead and is slower than ego,
      or a car is pulling out of a side street directly into the ego path.
  3 = medium — a pedestrian is standing at the edge of a crosswalk 20 m ahead and
      has not yet stepped off the curb, or a slow vehicle is 45 m ahead in the same lane.
  2 = low — a car is waiting at a red light on a perpendicular street to the right,
      or a parked car is on the same side of the road but 15 m ahead in a turn-off bay.
  1 = marginal — a pedestrian is walking on the sidewalk parallel to the road,
      clearly not heading toward the road, or a parked car is 40 m ahead in an
      adjacent lane the ego is not in.
  0 = irrelevant — a parked car on the opposite side of the road 50 m away, or a
      pedestrian on a distant pavement with no sign of approaching the road.

KEY RULES:
- Lane position is the primary signal: objects in a DIFFERENT lane with no sign of merging
  score 0-1, regardless of distance. Reserve 2+ only for objects on or converging toward
  the ego lane.
- Distance matters within the ego lane: objects at >25 m with no converging trajectory
  are at most 2. Objects at >40 m are at most 1.
- Parked, stationary, or clearly off-path objects should default to 0-1, not 3.
- Score 3 only when you genuinely need to keep watching the object; not as a default.

The numbered objects are:
{object_list}

Return ONLY compact JSON, no prose, no markdown, no code fences:
{{"scores": [{{"id": 0, "importance": 3}}]}}
Score every id listed. Use the full 0-5 range."""


# ---------------------------------------------------------------------------
# Prompt 3 — RICH schema (score + structured tags + one-sentence rationale)
#
# One VLM call returns everything the widened supervision channel needs
# (ZOI_CONTEXT.md sec 23.2 Tier-1 tags + Tier-2 rationale) at ~the same
# per-frame cost as the score-only prompt. Keeps the exemplar score anchors
# (they won the sec 18/18a comparison) and the compact-JSON output rule.
# Short keys ("imp"/"path"/"act"/"urg"/"why") keep the output small enough
# to avoid truncation on many-object frames.
# ---------------------------------------------------------------------------
RICH_PROMPT = """\
You are a planning-relevance annotator for an autonomous driving system.
The image is the EGO vehicle's front camera. Numbered green markers are already
placed on objects from the simulator's ground truth — you are NOT detecting objects.

For EACH numbered marker output FIVE fields:
  "imp":  0-5 planning relevance to the ego's driving decisions over the next 2-3 seconds.
  "path": will this object occupy space the ego vehicle needs within ~3 seconds?
          one of "yes", "maybe", "no".
  "act":  the ego's correct response to THIS object alone:
          one of "brake", "yield", "monitor", "ignore".
  "urg":  when it starts to matter: one of "now", "soon", "none".
  "why":  ONE short clause (max 15 words) naming the concrete visual/spatial evidence.

IMPORTANCE SCALE with one concrete example per level:
  5 = critical — a pedestrian has stepped off the curb directly into the ego lane 10 m ahead.
  4 = high — a cyclist has moved into the ego lane 25 m ahead and is slower than ego,
      or a car is pulling out of a side street directly into the ego path.
  3 = medium — a pedestrian is standing at the edge of a crosswalk 20 m ahead and
      has not yet stepped off the curb, or a slow vehicle is 45 m ahead in the same lane.
  2 = low — a car is waiting at a red light on a perpendicular street to the right,
      or a parked car is on the same side of the road but 15 m ahead in a turn-off bay.
  1 = marginal — a pedestrian walking on the sidewalk parallel to the road, clearly not
      heading toward the road, or a parked car 40 m ahead in a lane the ego is not in.
  0 = irrelevant — a parked car on the opposite side of the road 50 m away, or a
      pedestrian on a distant pavement with no sign of approaching the road.

KEY RULES:
- Lane position is the primary signal: objects in a DIFFERENT lane with no sign of merging
  score 0-1, regardless of distance. Reserve 2+ only for objects on or converging toward
  the ego lane.
- Distance matters within the ego lane: objects at >25 m with no converging trajectory
  are at most 2. Objects at >40 m are at most 1.
- Parked, stationary, or clearly off-path objects should default to 0-1, not 3.
- "imp", "path", "act", "urg" must be mutually consistent (e.g. imp 4-5 cannot have
  path "no" and act "ignore").

The numbered objects are:
{object_list}

Return ONLY compact JSON, no prose, no markdown, no code fences, exactly this shape:
{{"scores": [{{"id": 0, "imp": 4, "path": "yes", "act": "brake", "urg": "now", "why": "pedestrian stepping into ego lane 12 m ahead"}}]}}
Score every id listed. Use the full 0-5 range."""
#
# Instead of "how important is this object?" the model is asked "does this
# object conflict with the space the ego vehicle needs right now?". This
# spatial framing is more concrete than an abstract importance scale and
# naturally produces wide score distributions because each level maps to
# a distinct physical relationship.
# ---------------------------------------------------------------------------
EGOPATHCONFLICT_PROMPT = """\
You are a spatial conflict annotator for an autonomous driving system.
The image is the EGO vehicle's front camera. Numbered green markers are placed on
objects from the simulator's ground truth — you are NOT detecting objects.

For EACH numbered marker, answer: does this object conflict with the space the ego
vehicle needs over the next 2-3 seconds?

CONFLICT SCALE (0-5):
  5 = IMMEDIATE CONFLICT — the object is already in or entering the ego's path right
      now; the ego must brake or swerve this instant to avoid a collision.
      Example: pedestrian in the crosswalk the ego is crossing, car braking hard 8 m ahead.
  4 = IMMINENT CONFLICT — the object is about to enter the ego path within 1-2 seconds;
      the ego must begin braking or steering now to be safe.
      Example: cyclist drifting into the ego lane 20 m ahead, car running a red at the
      intersection the ego is about to enter.
  3 = POTENTIAL CONFLICT — the object is near the ego's path and its future motion
      could create a conflict; the ego should adjust speed or position as a precaution.
      Example: pedestrian on the curb of an upcoming crosswalk, car approaching a
      merge point 35 m ahead.
  2 = INDIRECT CONFLICT — the object is not on the ego path but may influence it
      indirectly (e.g. it constrains where the ego can steer if it needs to avoid
      something else, or it may interact with other traffic that then affects ego).
      Example: parked car narrowing the lane 20 m ahead in an adjacent lane,
      car waiting to turn at a side street 25 m ahead.
  1 = NO CONFLICT, MONITOR — the object is close enough to be aware of but poses
      no plausible path conflict given its current state and position.
      Example: pedestrian walking on the pavement alongside the road, parked car
      on the ego side of the road but in a bay clearly off the driving lane.
  0 = NO CONFLICT — the object is spatially irrelevant to the ego's path; removing
      it from the scene would not change the ego's plan at all.
      Example: car parked on the far side of the road facing away, pedestrian on
      a distant pavement not oriented toward the road.

GUIDANCE:
- Ask: "If this object moved unexpectedly toward the ego, how quickly would that matter?"
  Fast = 4-5. In a few seconds = 3. Only if several other things also changed = 1-2. Never = 0.
- Objects beyond 40 m with no converging trajectory: score 0-1.
- Objects in the ego lane and within 20 m: score 4-5 unless clearly stopped and safe.
- Do NOT default to 3. Use 3 only for genuine near-path uncertainty.

The numbered objects are:
{object_list}

Return ONLY compact JSON, no prose, no markdown, no code fences:
{{"scores": [{{"id": 0, "importance": 3}}]}}
Score every id listed. Use the full 0-5 range."""
