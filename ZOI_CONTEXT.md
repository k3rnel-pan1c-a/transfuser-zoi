# VLM-Guided Zone-of-Interest (ZOI) Module — Working Context

Self-contained handoff for working on a remote training server. Captures every
decision + finding from the design sessions so a fresh start (or a fresh Claude)
can continue without re-deriving. Paths are relative to the `research/` workspace
that contains `carla_garage/`, `navsim/`, `PCLA/`, `transfuser_zoi/`.

---

## 1. Goal & hypothesis
Add a Zone-of-Interest module to an end-to-end driving model that learns which
scene **objects/regions are planning-relevant** (not just perception-relevant),
**supervised by a VLM teacher (Qwen2.5-VL)** whose world knowledge exceeds the
small driving model's. Hypothesis: VLM-supervised ZOI improves planning,
especially safety-critical long-tail behavior.

**The claim must beat the right control:** supervised-ZOI vs. *unsupervised*-ZOI
(same tokens, `lambda_zoi = 0`). If they tie, the gain was just extra params and
the thesis fails (a valid negative result).

## 2. Base model (decided)
**TransFuser++ in `carla_garage/`** (user's choice; over plain TransFuser and over
VAD). Backbone = camera+LiDAR fusion. Planner = transformer decoder (`self.join`)
+ GRU waypoint/checkpoint heads. NAVSIM TF++ and PlanT remain possible later
generality experiments, not the POC.

## 3. Architecture findings (carla_garage/team_code) — expensive to re-derive
- BEV coverage `±32 m` (64 m square), `pixels_per_meter = 4`, `lidar_resolution = 256`.
- **Fused feature map is 8×8 → ~8 m per cell.** Hard ceiling: ANY dense ZOI map
  downstream of fusion can't exceed 8 m precision. (Why pooling 64→8 and gating
  the 64×64 are dead ends.)
- **`top_down()` in `transfuser.py:131` has NO skip connections** despite the
  `# FPN fusion` comment — it pure-bilinear-upsamples the 8×8 bottleneck to 64×64.
  So the 64×64 BEV (seg/detect heads) carries no real detail beyond 8×8.
- **Planner memory is a token list.** `model.py:308-331`: fused 8×8 → `change_channel`
  (1×1 conv to `gru_input_size`) → `+ sine pos_enc` → flatten to **64 tokens** →
  permute `[B, tokens, d]`. `extra_sensors` token concat at `model.py:326`,
  `tp_token` at `model.py:349`. wp/checkpoint queries cross-attend via `self.join`.
  **This extensible token list is the seam for ZOI.**

## 4. Chosen design: Route 1 — sparse relevance tokens (NOT a dense grid)
Represent ZOI as K relevance tokens, each carrying a **continuous (x,y) positional
encoding**, appended to the planner memory. Spatial precision lives in the pos-enc
(continuous), so it is immune to the 8 m/cell bottleneck.

**Token vs label — do not confuse:**
- The **token** fed to the planner is dim `d = gru_input_size` (must match memory).
- The **label** `[x, y, importance, class_id]` only supervises the module's small
  aux outputs; it is NOT the token.
- Token built inside the module:
  `token = LayerNorm(content[N,d] + PosEnc(xy)[N,d]) * (1 + sigmoid(importance))`.

**Module (Flavor 1B, recommended):** DETR-style — N learned queries cross-attend to
a genuine high-res mid-stage feature (e.g. 32×32 fused LiDAR stage), each query
outputs `(x, y, importance, [role])`. (Flavor 1A = reuse CenterNet head + importance
head; lighter but couples to detection quality.)

**Wiring (one concat):** after `model.py:331` permute, build `zoi_tokens [B,N,d]`,
gate by `*(1+sigmoid(importance))`, `torch.cat` onto `fused_features` along token
dim — same pattern as `tp_token`. Planner/GRU unchanged. Stash `zoi_xy, zoi_imp`
for the loss. Source 32×32 grid requires `transfuser.py` to also return an early stage.

**Train/inference parity (critical):** queries are learned + self-contained. At
inference there is NO VLM — module emits tokens from features directly. VLM only
supplies the TRAINING target. So closed-loop CARLA runs unchanged.

**Route 2 (fallback only):** dense gate UPSTREAM of bottleneck (16/32×32 inside
encoder) — needs unfreezing encoder stages; heavier. Parked.

## 5. VLM supervision (offline pseudo-labels)
Two generators in `transfuser_zoi/`, **identical output format** so consumer code
is shared. Output per frame: `zoi_labels/XXXX.npy`, shape `[M, 4]` =
`[x, y, importance∈[0,1], class_id]`, ONE ROW PER FILTERED GT BOX, in BEV ego meters.
Importance = VLM score / 5. Position always from CARLA GT (exact).

- **`generate_zoi_labels.py` — Set-of-Marks (RECOMMENDED).** Project filtered GT
  boxes into front image, draw numbered marks, VLM scores each id. No detection,
  no matching → cleanest. On-task (relevance judgment, the VLM's strength).
- **`generate_zoi_labels_vlm_detect.py` — VLM detects, then IoU-match to GT.**
  VLM predicts boxes+importance; match to projected GT AABBs (Hungarian on 1-IoU,
  gate 0.1); transfer importance to matched GT. Noisier; keep as an ablation row.

**GT filtering** mirrors `parse_bounding_boxes` (`data.py:990`): classes
{car, walker, traffic_light, stop_sign}; LiDAR-hit thresholds; red+`affects_ego`
lights only; within ±32 m. Targets MUST match the model's filtered set.

**Projection convention** (replicates `create_projection_grid`, `transfuser_utils.py:595`):
ego CARLA (x-front, y-right, z-up) → subtract `camera_pos=[-1.5,0,2.0]` (rot 0) →
pinhole reorder `[y, z, x]` → intrinsics `K` (fov 110, 1024×512) → divide by depth.
**Vertical (z) sign is subtle — ALWAYS eyeball `--debug_dir` overlays first.**
Scope: front-camera-visible objects only (POC tradeoff; loses occlusion zones).

**Constants to VERIFY in config.py before a full run:** `num_lidar_hits_for_detection_*`,
`min_z`/`max_z` (placeholders in the scripts), RGB ext `.jpg`.

## 6. Data plan (downloading a usable subset of the 380 GB dataset)
Dataset is **~40 per-scenario tarballs** on S3, NOT one blob:
`https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/dataset/${scenario}.tar`
(`tools/download_data.sh` loops all of them = the full 380 GB; don't run as-is.)

Download only ZOI-relevant scenarios (~30-80 GB), e.g.:
`PedestrianCrossing DynamicObjectCrossing OppositeVehicleRunningRedLight
BlockedIntersection StaticCutIn ParkedObstacle` (include ParkedObstacle as the
NEGATIVE / irrelevant-object case). Each tar spans multiple towns → town diversity free.
`curl -sI <url> | grep -i content-length` to check a tar's size first.
Point training at `--root_dir carla_garage/data` (recursively finds route folders).

**Eval hold-out:** hold out a TOWN (Town13/Town12 are standard CARLA-LB2 eval towns;
split-index files already in `carla_garage/data/`). Use `--setting`.

## 7. Dataset gotchas (verified in code)
- **First ~2.5 s of every route is auto-skipped:** `for seq in range(config.skip_first,...)`
  (`data.py:123`), `skip_first = int(2.5*carla_fps)//data_save_freq` (`config.py:377`).
  Mirror this in label gen — start each route at `skip_first`, NOT 0000.
- **Frame 0000 is OOD** (bright clear-noon): weather set via `shuffle_weather`/`set_weather`
  applies one tick late (`data_agent.py:69,347`), AND first-frame LiDAR is half-swept
  (`data_agent.py:248`). Don't label/train on it (skip_first already excludes it).
- **`rgb/` vs `rgb_augmented/`:** viewpoint augmentation (DAgger-style anti-covariate-shift),
  NOT color. `rgb_augmented` = same frame from a camera shifted `[-1,1] m` + rotated
  `[-5,5]°` (`config.py:333-338`); waypoint labels are transformed to match
  (`augmentation_rotation/translation` in measurements). Selected ~50% of the time
  (`augment_percentage=0.5`) at `data.py:527-543`. Color aug is separate, on top.
  **POC decision: label `rgb/` only and set `--augment 0` so geometry matches ZOI
  labels.** Re-enable augmentation (and transform GT box positions accordingly)
  before any CLOSED-LOOP eval.

## 8. Training protocol (decided)
| Component | Init | During ZOI training |
|---|---|---|
| Backbone (fusion) | pretrained | **frozen** |
| Planner (`join` + GRU heads + `target_speed_network` + `*_query`) | pretrained | **fine-tuned** |
| `ZoiModule` (new) | random | **from scratch** |

- **Fine-tune the planner, do NOT train it from scratch** (frozen backbone + random
  planner = worst case; pretrained planner already maps tokens→trajectory).
- Don't freeze the planner either — it must LEARN to consume the new ZOI tokens.
- Load pretrained checkpoint with `strict=False` (ZOI params missing → random init).
  Pretrained models: `https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip`
- **Two LR param groups:** lower LR on pretrained planner (~1e-4), higher on ZoiModule (~1e-3).
- Optional 2-stage warmup: (1) freeze backbone+planner, train ZoiModule on `zoi_loss`;
  (2) unfreeze planner, train jointly.
- **FAIR BASELINE RULE:** the no-ZOI baseline must ALSO be the pretrained planner
  fine-tuned on the SAME subset for the SAME steps (not the released numbers),
  else "ZOI vs baseline" conflates ZOI with extra fine-tuning.

## 9. Loss
Per frame: ZoiModule outputs `zoi_xy [N,2]`, `zoi_imp [N]` (+ optional `zoi_role`).
Targets from `.npy`: `tgt_xy [M,2]`, `tgt_imp [M]`. M ≠ N → Hungarian match:
`cost[i,j] = ||zoi_xy_i - tgt_xy_j||_1 + lam_imp * BCE(zoi_imp_i, tgt_imp_j)`.
Then: `pos = L1(matched xy)`; `imp = BCE(zoi_imp, target)` with unmatched slots → 0.
`zoi_loss = lam_pos*pos + lam_imp*imp`. Total = planning loss + `lambda_zoi * zoi_loss`
(start ~1, N≈20, src grid 32×32).
**Two gradient pathways into the module:** direct (`zoi_loss`) + indirect (planning
loss backprops through the appended tokens). `lambda_zoi=0` = unsupervised-ZOI ablation.

## 10. Evaluation
Primary (planning): **closed-loop CARLA driving score / sub-scores** (collisions,
infractions) and/or NAVSIM PDMS sub-scores; waypoint **L2** secondary.
Secondary (does ZOI work): attention–label **alignment IoU**; **counterfactual probe**
(zero high-ZOI vs random tokens → planning should degrade more for high-ZOI).
**Benefit shows in safety-critical LONG TAIL (collisions), not average L2.**

## 11. Ablation table
1. baseline (no ZOI) — fine-tuned identically
2. + random/uniform ZOI tokens (params control)
3. + learned ZOI, no VLM supervision (`lambda_zoi=0`)
4. + VLM-supervised ZOI (the claim)
5. frozen vs unfrozen backbone (on 4)
6. Set-of-Marks vs VLM-detect labels (on 4)
7. single-frame vs temporal VLM labels (on 4)
Belief test: 4 > 3 > 1 on collision/PDMS sub-scores.

## 12. Why ZOI should help (motivation, honest)
1. Distills VLM world knowledge the small imitation-trained planner lacks.
2. Restores spatial precision the 8 m grid destroys (continuous-coord tokens).
3. Regularizer against shortcut learning — pointed, since carla_garage IS the
   "Hidden Biases of End-to-End Driving Models" paper.
4. Concentrates capacity on the safety-critical long tail.
Risk: transformer attention may already learn relevance implicitly → must beat
the `lambda_zoi=0` control on long-tail metrics, not average L2.

## 13. Files to read (priority order)
Model: `team_code/model.py` (primary), `transfuser.py`, `config.py`.
Data/labels: `data.py` (`__getitem__` 274, `parse_bounding_boxes` 990, `get_targets` 740),
`center_net.py`, `transfuser_utils.py` (projection/transforms).
Train: `train.py`. Yours: `transfuser_zoi/test_zoi_qwen.py`,
`generate_zoi_labels*.py`. Skip: plant*, leaderboard*, scenario_runner*.

## 14. Status / next steps
DONE: design, two label generators written (`transfuser_zoi/generate_zoi_labels*.py`).
DONE: `ZoiModule` (`team_code/zoi_module.py` — `ContinuousPositionEmbedding` + DETR-style
query decoder) + `transfuser.py` now captures and returns a `zoi_feature_grid` (mid-stage
fused LiDAR features, 32×32 by default via `config.zoi_src_stage=1`, channel count exposed
as `backbone.zoi_src_channels`) + wired into `model.py` (instantiated in `__init__` when
`config.use_zoi`, tokens concatenated onto `fused_features` right after the `model.py`
permute, `pred_zoi_xy`/`pred_zoi_imp` added to `forward()`'s return tuple — `train.py` and
`sensor_agent.py` call sites updated to unpack the 2 new (currently unused) values) +
`config.py` flags added (`use_zoi, zoi_num_queries, zoi_src_stage, zoi_num_decoder_layers,
zoi_num_heads, zoi_lambda`). `PositionEmbeddingSine` moved from `model.py` to
`transfuser_utils.py` so both `model.py` and `zoi_module.py` can share it. Standalone
shape/gradient sanity-checked (`carla` package not installed in this sandbox, so the full
backbone — which needs pretrained timm weights — wasn't run end-to-end; verify on the
training server with `config.use_zoi=True`).
DONE (2026-06-28 — model training path fully wired, see §18):
- `zoi_loss` (batched Hungarian) added to `model.py` `compute_loss` as `_compute_zoi_loss`;
  position L1 on matched queries + importance BCE on ALL queries (matched→GT importance,
  unmatched→0 so the gating score learns (ir)relevance). Standalone gradient test passed.
- `CARLA_Data.__getitem__` loads `zoi_labels/XXXX.npy` (pad to `zoi_m_max`, validity mask),
  and applies `augment_route` to the label xy so it tracks viewpoint aug (identity at augment=0).
- `train.py` unpacks `pred_zoi_xy`/`pred_zoi_imp`, passes labels to `compute_loss`, sets
  `loss_zoi` weight = `zoi_lambda`, and builds TWO LR groups (planner at base LR, ZoiModule
  at `base_lr * zoi_lr_multiplier`). `strict=False` load already present.
- `config.py`: `loss_zoi` weight (default 0.0), `zoi_m_max=20`, `zoi_lr_multiplier=10.0` added.
- Ablation knobs fall out: `--use_zoi 1 --zoi_lambda 1.0` = the claim; `--zoi_lambda 0.0` =
  unsupervised control (tokens still flow via the indirect pathway, supervision off);
  `--use_zoi 0` = fair baseline.
TODO (next):
- [ ] Update label generators to start at `skip_first` and read `rgb/` (augment off).
- [ ] Verify the 3 config constants; run `--limit 30 --debug_dir` and eyeball overlays
      (esp. projection vertical sign).
- [ ] Download subset scenarios; download pretrained TF++ weights.
- [ ] On the training server: instantiate `LidarCenterNet` with `use_zoi=True` and run one
      real forward pass to confirm `zoi_src_stage=1` actually yields 32×32 for the chosen
      `lidar_architecture` (computed generically from `feature_info`, but only sanity-checked
      against the default `regnety_032`/256 lidar resolution math, not run). Also confirm the
      DDP `id()`-based param split in `train.py` covers exactly `zoi_module.*`.
- [ ] Train baseline (fair) + ablation rows.

## 15. Session update (2026-06-25) — Kaggle dataset, projection verified, signal-marking fix

Work done against the Kaggle dump `/kaggle/input/datasets/k3rnelpan1ca/carla-garage`
(6 scenarios: PedestrianCrossing, DynamicObjectCrossing, HighwayCutIn,
SignalizedJunctionLeftTurn, ParkedObstacleTwoWays, noScenarios). All new code in
`transfuser-zoi/`; preview artifacts in `/kaggle/working/zoi_preview/`.

### New / changed files (what each does)
- **`zoi_projection.py` (NEW, shared, torch-free).** Single source of truth for the
  projection + Set-of-Marks geometry: `project_ego_to_image(z_sign=-1)`, `keep_box`,
  `visible_marks(classes=VLM_MARK_CLASSES)`, `draw_marks`, `rule_importance`. Imported by
  generate_zoi_labels.py / preview_zoi_projection.py / verify_projection.py /
  run_vlm_multiframe.py (geometry was duplicated before → drift risk). **This is the file
  that holds the projection + traffic-light fix.**
- **`verify_projection.py` (NEW).** Renders the `*_projcheck.jpg` overlays that prove the
  z-sign: GREEN = `z_sign=-1` (correct, lands on objects), RED = `+1` (old bug, floats up),
  on brightened frames. Produced the 6 `<tag>_projcheck.jpg` preview images.
- **`run_vlm_multiframe.py` (NEW).** Multi-frame, multi-model (`gemma3|gemma4|qwen`) runner.
  Its `build_marked()` writes the `<tag>_marked.jpg` Set-of-Marks images and it writes
  `multiframe_{scores.json,compare.md}` + `<tag>__<model>.{json,txt}`. Frames come from
  `_preview_frames.json` (one diverse frame per scenario; selected by an inline snippet).
  The post-fix `_marked.jpg` re-render used an inline loop calling
  `zoi_projection.visible_marks`+`draw_marks` (equivalent to `build_marked`).
- **`sample_zoi_dataset.py` (NEW).** See §6/§16. Builds a carla_garage-loadable train/eval
  tree + label manifest from the Kaggle dump.
- **`generate_zoi_labels.py` (CHANGED).** Now: imports shared geometry; reads `.json` OR
  `.json.gz`; has `--skip_first`/`--stride`; **scores traffic_light/stop_sign by RULE and
  marks only car/walker** (see signal fix below).
- **`preview_zoi_projection.py` (CHANGED).** Imports from `zoi_projection` (works without VLM libs).

### Projection z-sign — VERIFIED CORRECT (was not actually a bug)
`cam=[p_y, -p_z, p_x]`, `p=pos-CAMERA_POS([-1.5,0,2.0])`; `-p_z` (z_sign=-1) puts marks ON
objects, `+p_z` floats them above. Confirmed on 6 diverse frames (`*_projcheck.jpg`). The
old floating marks in `0035_marked.jpg`/`0035_compare.jpg` were the pre-fix version.
TODO §14 "verify projection vertical sign" → **DONE**.

### Signal marking fix (the real issue) — `traffic_light`/`stop_sign`
In this dump a signal box `position` is the **road-level stop-line trigger (z≈0.35m, ~20m
ahead), NOT the visible fixture** → a projected dot lands on the road or ON a car ahead and
misleads the VLM (e.g. junction frame: 4 marks = 2 cars + 2 mis-located lights, one drawn on
the blue car). Fix in `zoi_projection.py`: `VLM_MARK_CLASSES={car,walker}` get VLM marks;
`RULE_CLASSES={traffic_light,stop_sign}` get `rule_importance` (red+affects_ego light→1.0,
stop_sign→0.8) and are NOT shown to the VLM. `keep_box` already restricts signals to
red+affects_ego, so a kept signal is relevant by definition. Data caveat: some visible PARKED
cars aren't in CARLA GT (untracked static) → unmarkable by any model.

### VLM model finding (single frame 0035 + multi-frame)
Flat Qwen baseline = all "3" (no gradient → useless teacher; brightening doesn't fix).
Gemma 3 12B ≈ Gemma 4 12B (both graded/sensible). **Default Gemma 3 12B (4-bit) on T4**,
Gemma 4 only on A100. Run gotchas: needs `bitsandbytes`+`qwen-vl-utils`; the verbose prompt
+256 tokens TRUNCATES JSON on frames with ≥6 marks → parse fails (use the COMPACT
id+importance prompt); `nohup &` orphans hold GPU mem → OOM (use proper backgrounding +
`pkill`/`nvidia-smi --query-compute-apps`).

## 16. Data subset & the "perfect-expert" filter (Kaggle)
carla_garage `data.py` (~L96-110) **silently drops any route whose expert drive wasn't
(essentially) perfect** — it's imitation learning, so imperfect demos teach bad behavior.
EXACT rule: drop if `score_composed < 100` AND infractions aren't ALL min-speed (a real
infraction happened), OR status in {`Failed`, `Failed - Agent crashed`,
`Failed - Simulation crashed`, `Failed - Agent couldn't be set up`}, OR name starts
`FAILED_`. Otherwise KEEP. **`Perfect` AND `Completed` are both kept**; min-speed-only
infractions are forgiven (the cautious expert often drives slow → score<100 but that's fine
to imitate). Dataset statuses: Perfect 662, Completed 577, Failed-timed-out 17, Failed-blocked 2.
Trainable per scenario (corrected): DynamicObjectCrossing 263, SignalizedJunctionLeftTurn
337, noScenarios 350, PedestrianCrossing 95, ParkedObstacleTwoWays 87, HighwayCutIn 81 →
the full **~5k train + ~1k eval** budget is easily met.

NOTE: an earlier version of `sample_zoi_dataset.py` wrongly also required status=="Completed",
discarding all 662 `Perfect` routes and producing the bogus "scarce data" counts (~3,665
frames; SignalizedJunctionLeftTurn 52/347 etc.). FIXED — `is_trainable` now mirrors data.py.
`sample_zoi_dataset.py` replicates this filter, splits train/eval by town (`--eval_town 13`),
fixes 2 format mismatches (dump stores PLAIN json + DOUBLE-nests `<Scn>/<Scn>/<route>`;
data.py wants GZIP `*.json.gz`/`results.json.gz` + immediate-subdir routes → it
gzips boxes/measurements/results, symlinks the rest, flattens to `<Scn>/<route>`), and emits
`_manifest/{train_labels.txt,eval_labels.txt,summary.json}`. Recommended POC: ~5k labeled
frames (debug at ~1-1.5k), stride 5, skip_first 10, Town13 held out, ParkedObstacle as
negatives. Full build not yet run.

### TODO delta from this session
- [x] Projection vertical sign verified; geometry centralized in `zoi_projection.py`.
- [x] Label generators: `--skip_first`/`--stride`, gz-aware, signal rule-scoring.
- [ ] Switch VLM runners to the COMPACT prompt before the real label/compare run.
- [ ] Run `sample_zoi_dataset.py` full build (~212 routes); then generate labels on it.

## 17. Multi-model VLM comparison: InternVL 3.5 vs Gemma 3 vs Gemma 4 (all bf16)
Goal: pick the best teacher VLM. Ran `run_vlm_multiframe.py` on 6 diverse frames (one per
scenario, from `_preview_frames.json`) with the **compact** prompt (id+importance only) →
zero parse failures. Results saved to **`/kaggle/working/zoi_preview_compare/`**
(`multiframe_compare.md`, `multiframe_scores.json`, per-frame `*__<model>.json/.txt`, `*_marked.jpg`).

Env/script changes:
- `run_vlm_multiframe.py`: added `internvl` backend (`OpenGVLab/InternVL3_5-8B-HF`, native
  transformers path, same code as gemma) + a `--prompt {compact,full}` flag (compact default,
  avoids JSON truncation) + `COMPACT_PROMPT_TEMPLATE`. **All backends now full-precision bf16
  for fairness** (gemma3 switched from the `-bnb-4bit` mirror to `unsloth/gemma-3-12b-it`;
  the 12B bf16 models shard across both T4s via `device_map="auto"`).
- **`transformers` upgraded 5.0.0 → 5.12.1**: gemma4's `gemma4_unified` arch is unrecognized
  before 5.10 (`KeyError: 'gemma4_unified'`). InternVL3.5-HF + Gemma3 also load on 5.12.1.
- `merge_compare.py` (NEW): rebuilds the combined `multiframe_{scores.json,compare.md}` from
  the per-frame `*__<model>.json` files (used because gemma4 was re-run separately after the
  transformers upgrade, and the crashed first pass never wrote the summary).

Findings (6-frame screen, NO ground truth — directional, not final):
- All three correctly rank pedestrians + close lead cars highest, far/off cars lowest (all
  real teachers, unlike the flat Qwen baseline from §15).
- **Gemma 3 = weakest:** narrowest range, COLLAPSED to flat 3.0 on all 7 cars in the
  pedestrian frame. Drop it.
- **Gemma 4 = best-calibrated gradient:** clean monotonic-by-distance, full 2→5 range,
  reserves 5 for the closest/critical. But 12B bf16 (~24GB, both T4s; ideal on A100).
- **InternVL 3.5 8B = best practical pick:** WIDEST relevant-vs-irrelevant contrast (uses 1
  for far cars AND 5 for closest → strongest ZOI supervision signal), ~half the params,
  fast (~12-22s/frame), clean parsing. Recommended for labeling thousands of frames on T4.
- Recommendation: **InternVL 3.5 8B** for practicality, **Gemma 4** if compute allows; drop
  Gemma 3. To settle rigorously: label a few hundred frames with the top-2 and compare, or
  build a small gold set.

## 18. Prompt comparison on Gemma 4: compact vs exemplar vs egopathconflict (2026-06-28)
New prompts live in `transfuser-zoi/zoi_prompts.py` (`EXEMPLAR_PROMPT`, `EGOPATHCONFLICT_PROMPT`);
`run_vlm_multiframe.py` already supports `--prompt {compact,exemplar,egopathconflict,full}`,
and `eval_prompts.py` scores a run dir by spread-std / depth-ρ (Spearman of score vs 1/depth) /
top-1-closest / parse-fails. Ran Gemma 4 (`unsloth/gemma-4-12b-it`, bf16, sharded across 2×T4,
transformers 5.12.1) on the 6 `_preview_frames.json` frames (37 objects). Output in
**`/kaggle/working/prompt_eval_gemma4/{compact,exemplar,egopathconflict}/`**.

| prompt | spread std | depth-ρ | range | top1 | parse_fail |
|---|---|---|---|---|---|
| compact | 1.59 | 0.66 | 0–5 | 0.33 | 0 |
| **exemplar** | **1.61** | **0.86** | 1–5 | 0.33 | 0 |
| egopathconflict | 1.10 | 0.64 | 1–5 | 0.33 | 0 |

- **WINNER = exemplar.** Widest spread AND depth-ρ jumps 0.66→0.86 (per-score-level driving
  anchors calibrate distance much better). Histogram is bimodal (clusters at 1 and 4) = sharp
  relevant-vs-irrelevant contrast, the strong supervision signal the importance head wants.
- **egopathconflict backfired:** the "potential conflict" framing made Gemma 4 hedge to the
  middle (collapsed to 2–3–4, std 1.10). Drop it.
- Caveat: 6 frames = directional (same as §17); to settle, label ~100–200 frames compact-vs-
  exemplar and re-run `eval_prompts.py`. Also exemplar's floor is 1 (never 0) — fine since
  importance=score/5 and the gate is sigmoid (relative order matters); tweak prompt if a true
  0 floor is wanted for ParkedObstacle negatives.

### 18a. Closed the cross: InternVL vs Gemma 4 on compact AND exemplar (same 6 frames)
§17 only had InternVL-compact; §18 only had exemplar-on-Gemma4 → never apples-to-apples. Ran
InternVL on both prompts too. Output in `/kaggle/working/prompt_eval_cmp/internvl_{compact,exemplar}/`.

| teacher × prompt | spread std | depth-ρ | top1 | parse |
|---|---|---|---|---|
| Gemma 4 · compact | 1.59 | 0.66 | 0.33 | 0 |
| **Gemma 4 · exemplar** | **1.61** | **0.86** | 0.33 | 0 |
| InternVL · compact | 1.28 | 0.73 | 0.50 | 0 |
| InternVL · exemplar | 1.16 | 0.48 | 0.50 | 0 |

- **Gemma 4 + exemplar wins outright** — best spread AND best depth-ρ; beats InternVL's *best*
  config (compact, 0.73) on both axes.
- **Prompts are TEACHER-SPECIFIC: exemplar helps Gemma 4 but HURTS InternVL** (depth-ρ 0.73→0.48,
  histogram herds 20/37 objects to "2"). Do not reuse one teacher's tuned prompt on another.
- DECISION for the real labeling run: **Gemma 4 + exemplar** (A100 makes its speed penalty moot;
  earlier §17 "InternVL for practicality" lean was compact-vs-compact, superseded once the prompt
  is optimized). FALLBACK if forced onto T4: **InternVL + compact** — NEVER InternVL + exemplar.
- Margin caveat: InternVL-exemplar collapse is robust; the 0.86-vs-0.73 gap is real but small-n
  (37 objects). Bulletproof it with a ~100–200 frame labeled compare if needed, not blocking.

### 18b. Production label generator `generate_zoi_labels_gemma4.py` (NEW, 2026-06-28)
Built + smoke-tested on real Gemma 4 weights. Implements the §18a decision; the old generators
are superseded (Qwen one = dropped teacher + verbose prompt; Gemma-3 one = dropped teacher AND
is STALE — it marks signals (pre-§15 bug) and isn't gz-aware; do not use either for the real run).
- Backend `AutoModelForImageTextToText` (loads gemma4_unified; Gemma-3 class can't), default
  `unsloth/gemma-4-12b-it`, bf16 `device_map=auto`. `dtype=` kwarg confirmed on transformers 5.12.1.
- Prompt `--prompt exemplar` (default; `egopathconflict`/`full` for ablation). Object lines INCLUDE
  depth ("~12 m") to match how exemplar was evaluated in run_vlm_multiframe.
- Signal-correct (shared zoi_projection): traffic_light/stop_sign scored by rule, NOT marked.
- Training-parity selection: `--skip_first 10`, `--stride`. gz-aware. Output `[x,y,imp,class_id]`
  .npy = drop-in for the data.py loader. Per-frame/route/total timing for budgeting.
- Smoke test (2 frames, real weights): model load ~103s, VLM ~12.8s/frame on T4; produced a clean
  graded label (car +15m→0.8, +30m→0.4, behind→0.0). Empty-frame → (0,4) array, loader handles it.
- BUDGET DATA POINT: ~12.8s/frame VLM on T4 → ~5k frames ≈ 9 GPU-h (faster on A100). Matches §19.
- Run: `python generate_zoi_labels_gemma4.py --root_dir <tree> --output_dir <writable> [--stride 5]`.
  Prereq: point at the raw Kaggle dump OR run `sample_zoi_dataset.py` first (full build still pending).

### 18c. Sampler<->labeler link + coverage guard (2026-06-28)
The link between sample_zoi_dataset.py and the labeler was the `_manifest/*.txt` frame list,
but the generators ignored it and re-derived frames via their own skip_first/stride glob (two
independent selection paths -> silent drift; data.py treats a missing .npy as zero-supervision,
no crash). Closed both ways:
- `generate_zoi_labels_gemma4.py --manifest <out_root>/_manifest/train_labels.txt [eval_labels.txt]`
  labels EXACTLY the sampler's frames (needs `--root_dir <out_root>`); skip_first/stride/route are
  then ignored. Single source of truth = the manifest. Frames whose files aren't in the tree are
  counted as MISSING in the summary, not silently skipped.
- `verify_zoi_coverage.py --manifest ... [--labels_root <out_root>]` is the GATE: checks every
  manifest frame has a well-formed `[M,4]` label (finite, importance in [0,1], class_id in {0,1,2,3});
  M==0 valid but counted (zero-supervision frames). Exits 1 on <100% coverage or any malformed, so:
  `python verify_zoi_coverage.py ... && torchrun ... train.py ...`. Tested on missing+malformed
  fixtures (fails) and clean labels (passes).

### 18d. transfuser.py ZOI grid wiring — VERIFIED by code read (no GPU needed)
`zoi_src_stage=1` is correct for the default backbone: start_index=0 (regnety_032 has no extra stem
return layer), loop i maps to feature_info stage i with reductions [4,8,16,32] -> at lidar_resolution
256, i=1 = 32x32 (i=3 = the 8x8 bottleneck, matches sec 3). Grid captured at transfuser.py:184 AFTER
fuse_features (so genuinely image+lidar fused, not raw lidar). `zoi_src_channels` (transfuser.py:104)
indexes the SAME stage and fuse_features preserves channel count -> matches ZoiModule.input_proj by
construction. Robust: a non-default backbone where i=1 != 32x32 won't crash (input_proj+flatten adapt),
just different precision. Still TODO on server: one real use_zoi=True forward (shape-confirm end to
end) + sensor_agent.py inference unpack.

## 19. POC sizing — how much data to tell if ZOI works
Binding constraint: the benefit is in the safety-critical LONG TAIL (collisions), not avg L2
(§1/§10/§12), and rare events need the most data. Stage it; don't jump to closed-loop.

| Stage | Question | Data | Verdict signal |
|---|---|---|---|
| 0 Teacher | VLM label discriminative & geometric? | 6–50 frames (DONE, §17/§18) | high spread, strong depth-ρ, no flat collapse |
| 1 Learnability | Can ZOI head fit the labels? | ~1–1.5k frames (debug budget) | zoi_loss drops; pred imp correlates w/ held-out VLM labels (alignment IoU) |
| 2 Open-loop | Any planning benefit vs control? | ~5k train + 1k eval | row4 > row3 on the safety-critical SUBSET even if avg L2 ties |
| 3 Closed-loop | Drives better? | small CARLA route set, 2–3 seeds | score/collisions, row4 > row3 ≥ row1 |

Minimum viable go/no-go POC:
- **~5k labeled frames** (§16 budget, easily met), stride 5, skip_first 10, **Town13 held out**.
- **Scenario mix > raw count:** bias hard to PedestrianCrossing/DynamicObjectCrossing/HighwayCutIn
  + ParkedObstacle as the NEGATIVE case. Only ~10–30% of frames carry a safety-critical object —
  that subset is what you measure on; cruising frames add ~no ZOI signal.
- **Three arms always:** baseline (row1) / unsup-ZOI λ=0 (row3) / VLM-ZOI (row4), fine-tuned identically.
- Closed-loop: ~20–40 critical routes/arm × 2–3 seeds = directional POC. Publishable collision-rate
  claim needs ~50–100 critical routes/arm (rare Bernoulli events).
- **NON-NEGOTIABLE: the verdict is row4 vs row3, NOT row4 vs row1** — extra params + extra
  fine-tuning help regardless; comparing only to baseline manufactures a false positive (§12).
  row4 ≈ row3 on the long tail = valid negative result (attention already learns relevance).
- Cost driver is LABELING, not training: ~15s/frame → 5k frames ≈ 21 GPU-h/teacher on one T4
  (→ teacher speed matters; A100 makes Gemma 4 viable).

## 20. Honest risk assessment + architecture/benchmark strategy (2026-06-28 discussion)
Recorded because the binding question for this whole project is whether ZOI beats the λ=0 control,
and the answer may well be "no". This section is the sober prior, the cheap go/no-go, and the
architecture/benchmark options — so we don't sink the labeling+training budget on a blind bet.

### 20.1 Why ZOI may NOT improve the model (well-founded skepticism)
- **Redundancy with implicit attention (the core risk, §12).** The `join` decoder already cross-
  attends over BEV tokens — attention IS learned relevance. ZOI adds explicit supervision for
  something the architecture can already learn. If implicit attention captures most of it, ZOI =
  extra params.
- **Thin supervision channel.** The VLM transfers ONE scalar per GT box. Position is already exact
  (GT). So ZOI's entire contribution is a relevance *ranking* — which the imitation target half-
  teaches anyway (the expert braked for the pedestrian → planning loss already encodes "matters").
- **Teacher is weak where it counts.** Even Gemma-4-exemplar's signal is mostly DISTANCE
  (depth-ρ 0.86 = closer→higher), which the model already gets from geometry for free. The non-
  trivial calls (far red light matters, close parked car doesn't) are where the VLM is noisiest.
- **Auxiliary-attention supervision has a mixed-to-poor track record:** usually improves the PROXY
  metric (attention alignment) without moving the TASK metric (driving score). carla_garage is the
  "Hidden Biases" shortcut model — a relevance head doesn't remove the shortcut; the model can
  satisfy zoi_loss AND keep cheating.
- **Modal expected outcome:** row4 ≈ row3 on average metrics; maybe a small, noisy long-tail effect
  hard to establish at ~5k frames. That is the §1 valid-negative-result branch. NOT worthless —
  for a thesis/paper a CLEAN negative (made clean by the row3 control) is a real contribution; for
  shipping a better driver it's a speculative bet with unfavorable odds. Goal determines value.

### 20.2 Cheapest go/no-go: attention-correlation probe (DO THIS FIRST, no training)
Take the RELEASED pretrained model, run on a few hundred frames, extract the planner attention over
BEV tokens (model already has `TransformerDecoderWithAttention` / `tp_attention`), correlate per-
object attention mass with the VLM importance labels.
- Attention ALREADY correlates strongly with VLM relevance → model already knows it → ZOI likely
  redundant → STRONG kill signal; saves ~9–21 GPU-h labeling + all training. Architecture-independent.
- Attention does NOT correlate → genuine headroom → proceed.
- Caveat: not 100% definitive (no-correlation doesn't guarantee ZOI helps), but strong correlation
  is a strong negative. Highest info-per-hour test available. Reuses existing projection/label code.

### 20.3 Does a different architecture change the odds?
- **No, not by itself.** The redundancy-with-implicit-attention risk follows you to ANY attention
  planner (VAD, UniAD, PARA-Drive, PlanT, NAVSIM TF). The control (row4 vs row3) is the real test
  regardless of base model. "Try another architecture" = same bet, different shirt.
- **The axis that DOES matter: explicit object tokens vs feature soup.**
  - TransFuser++ (current): relevance is diffuse (8×8 grid + attention); ZOI tokens are bolted on and
    the planner must LEARN to use them — weakest fit, easiest to ignore.
  - Object-token planners (**PlanT — already in-repo, privileged, takes GT boxes as input**; or
    VAD/UniAD): relevance attaches DIRECTLY to an existing per-object query as an aux head; attention
    is per-object interpretable so you can measure if supervision changed it. Cleaner MECHANISM, but
    they still have planning attention → control risk remains (cleaner, not better odds).
- **The bigger lever is the BENCHMARK, not the architecture.** Worry = "will it improve the model?" →
  fastest answer = more cheap shots on goal = eval cost:
  - carla_garage = closed-loop CARLA: expensive, slow, noisy, simulator-bound, shortcut-learner →
    one hard-to-read bet.
  - **NAVSIM (open-loop PDMS): no simulator, standardized metric, fast train+eval, current hot
    benchmark** (§2 already lists it as a generality experiment). Same ZOI idea on NAVSIM lets you
    test 2 architectures cheaply instead of staking it all on one closed-loop run.

### 20.4 Recommended order
1. Run the §20.2 attention probe FIRST (architecture-independent, hours, no training).
2. If switching, switch architecture AND cheap-eval together: PlanT or a NAVSIM transformer planner
   (clean aux head + fast iteration).
3. Keep the λ=0 control no matter what — it IS the experiment.
4. OPEN QUESTION that changes everything: goal = thesis/paper (clean cross-arch result, even
   negative, is valuable + achievable on NAVSIM) vs working better driver (be more skeptical of the
   whole line regardless of architecture). Decide this before spending the budget.

## 21. Go/no-go probes RUN on Kaggle (2026-06-29) — residual + attention, both implemented
Executed the §20 cheap go/no-go tests BEFORE committing the labeling/training budget. Two new
self-contained scripts in `transfuser-zoi/`; artifacts under `/kaggle/working/zoi_probe*`.

### 21.1 Environment deltas (needed to re-run; some contradict earlier notes)
- **`pip install carla` WORKS on this Python.** Kaggle is py3.12; PyPI has a `carla==0.9.16`
  cp312 manylinux wheel (34 MB). The §5/sandbox assumption "carla not installable" is FALSE here —
  no stub needed for the import itself. (transfuser_utils only uses carla.Vector3D/Location in
  geometry helpers off the preprocessing path anyway.)
- `pip install "laspy[lazrs]"` (lidar .laz), `transformers==5.12.1` (env shipped 5.0.0 → gemma4
  `KeyError: gemma4_unified`; matches §17). `timm` is independent of the transformers bump.
- Pretrained TF++ weights downloaded: `models/pretrained/pretrained_models/{all_towns,
  town13_withheld}/model_0030_{0,1,2}.pth` (+ config.json/args.txt). Used **town13_withheld**
  (matches the held-out-Town13 eval plan). config has tp_attention=0, seq_len=1, use_ground_plane=0
  (lidar=1 channel), transformer_decoder_join=1, gru_input_size=256.

### 21.2 Probe #1 — `analyze_importance_residual.py` (is importance just distance?)
Regress importance on closeness 1/(1+r) AND a behind flag; the leftover variance = what the VLM
adds beyond free geometry. **Filters to VLM-scored classes car=0/walker=1** (traffic_light=2/
stop_sign=3 carry deterministic rule_importance, NOT VLM judgment — excluding them isolates the
test; this class filter was added after a good catch and applies to BOTH probes).
Results (objects, residual = 1 - R^2 of geometry):
- Gemma-4 labels (`zoi_labels_gemma_fast/_test`, ~100-130 obj): **~44-51% residual** → real
  non-distance signal; within-distance-bin importance std ~0.27 (equal-geometry disambiguation).
- Old flat-Qwen `zoi_labels` (1097 obj): only 18% residual; `behind` flag alone explains 82% →
  degenerate teacher (confirms §15 "flat Qwen"). DO NOT use these labels.
- Caveat: necessary-not-sufficient. Big residual could be VLM noise; probe #2 distinguishes
  signal-residual from noise (does attention already track it?).

### 21.3 Probe #2 — `attention_probe.py` (does the planner ALREADY attend to VLM-important objects?)
Runs the RELEASED TF++ on real frames, reads planner cross-attention over the 64 BEV tokens,
correlates per-object attention with VLM importance. **Decisive metric = attention-RESIDUAL vs
importance-RESIDUAL** (both after removing distance), NOT raw corr (both rise with closeness).
Harness recipe (expensive to re-derive — all VERIFIED working on a real forward pass):
- model.py → nav_planner imports `agents.navigation` (CARLA leaderboard control) and data.py
  imports `imgaug` (broken on numpy 2.0) — BOTH stubbed in `_install_stubs()` (neither is on the
  fwd/preprocess path). carla itself is real (pip).
- Preprocessing reused VERBATIM from `CARLA_Data` via `__new__` (skip the heavy dataset-scanning
  __init__, just attach `.config`) → `lidar_to_histogram_features`, `align` (identity at seq_len=1),
  match training exactly. Inputs: rgb=cv2 BGR→RGB float 0-255 (backbone normalize_imagenet /255),
  lidar=laspy.xyz→hist, target_point/speed/command(one-hot) from measurements json.
- Attention "by HOOK not by FLAG": keep tp_attention=0 (so released weights load strict),
  monkeypatch `model.join.layers[-1].multihead_attn` to force need_weights=True and stash weights.
  attn shape [B,11,65]=11 planner queries × (64 BEV + 1 sensor token; NO tp token at tp_attention=0).
  Take attn[:, :64].mean(query)→8×8 grid.
- `xy_to_cell`: ego (x=front,y=right) m → 8×8 cell (±32m, 8m/cell), col=x row=y, token=row*8+col.
  Orientation **VALIDATED** via debug overlays (`zoi_probe/debug/*_attn.jpg`, JET heatmap + dots) and
  camera|BEV side-by-sides (`make_sidebyside.py` → `zoi_probe/sidebyside/`): behind-car→left edge/blue,
  far-lateral→corner/blue, front-pedestrian→center near red peak. Correct.
- PRELIMINARY result (PedestrianCrossing only, basically ONE route, 179 car/walker objects):
  att-vs-importance raw Pearson +0.43; att-vs-closeness +0.43/Spearman +0.53; **att-resid vs
  imp-resid Pearson +0.24, Spearman -0.02**. → attention tracks DISTANCE/path, only weakly the VLM's
  per-object relevance residual → tentative HEADROOM for ZOI ("proceed" lean). FRAGILE: one route,
  Pearson/Spearman disagree (few-point-driven), 8m cells blur objects. NOT a verdict.

### 21.4 Curated probe label set (the real run)
`generate_zoi_labels_gemma4.py` (gemma-4-12b-it, exemplar, shards across 2×T4 ~12GB each, ~13s/frame)
on a balanced mix → `/kaggle/working/zoi_probe_labels/<Scenario>/<route>/zoi_labels/*.npy`:
PedestrianCrossing (stride10) + DynamicObjectCrossing/HighwayCutIn/ParkedObstacleTwoWays(NEG)/
SignalizedJunctionLeftTurn (stride25 for route/town diversity), 40 frames each = ~200 frames.
TODO (immediate): on completion run `attention_probe.py --labels_root zoi_probe_labels` POOLED for the
real residual-corr number; re-run `analyze_importance_residual.py` on the same set (now class-filtered).
Then DECIDE per §20.4: strong att-resid↔imp-resid corr = kill; weak = proceed to Stage-1 learnability.

### 21.5 POOLED probe result (all 5 scenarios, 198 frames, 819 car/walker objects)
`attention_probe.py --labels_root zoi_probe_labels`:
- att vs importance (raw): Pearson +0.36 / Spearman +0.38
- att vs closeness 1/(1+r): Pearson +0.40 / Spearman +0.53
- **att-resid vs imp-resid (THE TEST): Pearson +0.23 / Spearman +0.24** (now AGREE → stable, unlike the
  one-route +0.24/-0.02). Reading: attention dominated by geometry/path; only ~5% of variance (R^2≈0.05)
  of the VLM's non-distance relevance is captured → NOT the redundancy kill-signal → soft "proceed".
- **BUT this number is computed on CONTAMINATED labels — see §22. Re-run after the label fix before trusting it.**

## 22. CRITICAL label-quality bug found (2026-06-29) — silent-zero + dark-image omission + lossy depth
User inspection of `subtle_examples.py` renders caught nonsense labels (a braking in-lane lead car scored
0.0; a far off-axis car scored higher than a close in-lane one). Root-caused with `diag_frame.py`/`diag_show.py`
(re-run Gemma 4 on the offending frames, printing the marked input + exact prompt + RAW output). Findings:

### 22.1 The silent-zero bug (most serious)
`generate_zoi_labels_gemma4.py process_frame`: `targets=np.zeros((n,4))`; importance is ONLY overwritten when the
VLM returns a score for that mark id (`for s in parsed["scores"]: targets[id_to_row[oid],2]=imp/5`). **A marked
object the VLM OMITS from its answer stays 0.0 — indistinguishable from a real "irrelevant".** So a `0.0`
car/walker label means one of: VLM judged irrelevant / VLM forgot the id / projected off-image. The VLM
demonstrably omits ids (and here dropped the SINGLE most relevant object). Prompt is NOT the cause — exemplar
says "For EACH numbered marker..." + "Score every id listed." VLMs are text generators, not form-fillers; "cover
all ids" is a soft constraint they violate at some rate. → false "irrelevant" labels contaminate BOTH probes
(the §21.5 +0.23 included these) and would actively harm training ("ignore the car you're braking for").

### 22.2 PROVEN cause for the example + cheap fix: image was too DARK
Frame `DynamicObjectCrossing/Town01_Rep0_Town01_Scenario3_5_route0_.../0125`: marks id0=lead car @~13m (in lane,
brake lights), id1=far car @~23m. A/B with `diag_show.py` (SAME prompt, SAME marks, only brightness differs):
- DARK input (what pipeline feeds): `{"scores":[{"id":1,"importance":1}]}` → id0 MISSING → silent 0.0 (the bug).
- BRIGHTENED input (alpha 2.6,beta 30): `{"scores":[{"id":0,"importance":4},{"id":1,"importance":1}]}` → id0=0.8
  (CORRECT, matches the 5/6 sibling routes that scored this lead car 0.8-1.0), id1=0.2.
→ Darkness made Gemma unable to see/parse the lead car. Many CARLA frames are night/dusk/rain (weather aug), so
exposure-normalizing the VLM INPUT image likely recovers a large share of dropped objects FOR FREE. (Verified
artifact: `zoi_probe/diag/0125_EVERYTHING.jpg` = verbatim exemplar prompt + dark vs bright inputs + both outputs.)

### 22.3 Lossy object description: depth is forward-distance-only, NO lateral/lane info
`zoi_projection.project_ego_to_image` returns `depth = cam[2] = x + 1.5` (longitudinal dist from camera, camera
at CAMERA_POS x=-1.5), NOT the radial range. The object line is just `a {class} at ~{depth}m`. So for an off-axis
object (0125 id1 ego=(21.6, 21.6)): told "~23 m", actually ~30 m away AND 21 m to the right (other lane). The
exemplar prompt says "Lane position is the PRIMARY signal" yet the VLM is given ZERO lateral/lane data — must infer
lane from the dot's pixel position alone. Object positions x,y,z ARE exact GT in the dataset (`boxes/XXXX.json`
`b["position"]=[x_fwd,y_right,z_up]`), so richer lines cost nothing: e.g. `a car 22 m ahead, 22 m to the right
(~30 m, NOT ego lane)`.

### 22.4 THE FIX (fold all three into the labeler before any re-label / re-probe)
1. **Brighten/exposure-normalize the VLM input image** (auto-gamma/CLAHE or simple alpha/beta). Proven to recover ids.
2. **Completeness guard — never silent-default to 0.** Track returned-vs-marked ids; for any missing id RETRY
   (re-prompt with just the missing ids), then mark any STILL-missing as INVALID (validity mask / NaN), excluded
   from supervision AND from the probe. Also log the omission rate (currently unmeasured).
3. **Richer object lines**: longitudinal + lateral offset (or lane) + true radial range — give the VLM the lane
   signal the prompt says is primary.
Bigger teacher (Qwen2.5-VL-72B / InternVL3-38/78B / Molmo-72B) is an option for scene understanding but needs
A100-80GB and WON'T guarantee completeness — the validity guard is needed regardless. Try brighten+guard+richer
FIRST (cheap) before swapping teachers.

### 22.5 Status / immediate next steps
- New files this session (all in `transfuser-zoi/`): `attention_probe.py`, `analyze_importance_residual.py`,
  `make_sidebyside.py`, `subtle_examples.py`, `diag_frame.py`, `diag_show.py`. Artifacts under `/kaggle/working/zoi_probe/`
  (`debug/` attn overlays, `sidebyside/`, `subtle/` the mined examples, `diag/` the bug evidence inc. 0125_EVERYTHING.jpg).
- Probe labels at `/kaggle/working/zoi_probe_labels/<Scenario>/<route>/zoi_labels/*.npy` (200 frames) are
  CONTAMINATED by §22.1 → DO NOT trust §21.5 (+0.23) until re-labeled with the §22.4 fixes.
- TODO: (1) implement §22.4 in the labeler; (2) re-label the 200 frames + report omission rate; (3) re-run
  `attention_probe.py` on the clean labels for the real go/no-go number; (4) then decide per §20.4.

## 23. Teacher upgrade, richer distillation, and paper positioning (2026-07-06 session)
Strategy review session (no code changes). Three outcomes: new teacher candidates (the "NVIDIA one"
the prof mentioned), a plan to widen the supervision channel with reasoning traces, and — critically —
a prior-art finding that changes how the paper must be framed.

### 23.1 NVIDIA teacher candidates (prof's suggestion — verified July 2026)
- **Cosmos-Reason2** (released 2025-12-19; 2B/8B/32B on HF under `nvidia/Cosmos-Reason2-*`): open
  reasoning VLM for physical AI, #1 open model on Physical AI Bench. Post-trained heavily on driving
  VQA. Natively supports 2D/3D point localization, bounding boxes, trajectories; **video-native**
  (256K ctx). The 8B is in the same T4-friendly class as InternVL 3.5 8B → no A100 dependency.
  **ACTION: run Cosmos-Reason2-8B through the existing bake-off harness** (`run_vlm_multiframe.py`
  + `eval_prompts.py` spread/depth-ρ, then §21.2 residual probe) vs Gemma-4-exemplar. Prompts are
  teacher-specific (§18a) → tune fresh, don't reuse exemplar verbatim. Being a reasoning model it
  emits long CoT before the answer → enforce structured final-answer format + generous max tokens
  or the §15 truncation-parse failures return.
- **Alpamayo-R1-10B** (`nvidia/Alpamayo-R1-10B`, arXiv 2511.00088): driving VLA built ON
  Cosmos-Reason; autoregressively generates "Chain-of-Causation" reasoning traces + trajectory
  tokens, targeted at the long tail. NOT a practical teacher for us (expects 4-camera real-world
  rig, 10 Hz history — domain mismatch with single-front-cam CARLA) but is DIRECT evidence that
  reasoning-trace training improves long-tail action prediction → cite as motivation/related work.

### 23.2 Reasoning-trace supervision — 3 tiers (answers the §20.1 thin-channel critique)
The scalar transfers a few bits/frame at ~13 s/frame of VLM compute, and ~50% of it is distance
(§21.2). Widen the channel, in order of robustness-per-effort:
- **Tier 1 — structured per-object fields (do regardless):** teacher also emits discrete tags per
  object: `crosses_ego_path: yes/no/maybe`, `suggested_ego_response: brake/yield/ignore/monitor`,
  `time_criticality: now/soon/none`. Supervise small classification heads on the matched ZOI
  queries. Cheap to parse/verify; carries the CAUSAL info the scalar doesn't ("matters because it
  will enter my path" vs "matters because it's close").
- **Tier 2 — rationale-embedding distillation (the VLM-AD/DiMA mechanism):** teacher writes a
  one-sentence per-object rationale; embed with a FROZEN text encoder (SigLIP/CLIP text tower or
  sentence-transformer); small projection head on each matched query, cosine loss vs embedding.
  Do NOT put a text-generation head on the module — wrong weight class; embedding regression is
  how VLM-AD gets gains with zero inference cost. Scene-level variant (one rationale → one scene
  token) is the cheaper starting point.
- **Tier 3 — temporal labels:** Cosmos-Reason2 is video-native; a short clip lets the teacher see
  INTENT (pedestrian accelerating toward curb vs standing) — the thing single-frame importance
  cannot encode, and where the long-tail benefit plausibly lives. = ablation row 7, now feasible.
- Every tier adds parse-failure surface → the §22.4 completeness guard generalizes: any object
  whose structured fields fail to parse gets validity-masked, NEVER a silent default.

### 23.3 Distillation recipe (ordered)
1. **§22.4 fixes FIRST** (brighten, retry-then-invalidate, richer object lines) — prerequisite to
   everything; then re-run the attention probe on clean labels (the real go/no-go gate stands).
2. **Teacher bake-off:** Cosmos-Reason2-8B vs Gemma-4-exemplar, per-teacher tuned prompts, decided
   on non-distance residual + spread. Add **self-consistency**: sample teacher 3× at temp>0,
   average scores, use variance as per-label confidence weight (mitigates omissions AND noise).
3. **Importance loss → pairwise ranking + down-weighted BCE.** Absolute VLM calibration is the
   teacher's weakest property (§17/§18 score-scale swings); within-frame relative order is robust.
   Keep some BCE so the sigmoid gate in `token * (1+sigmoid(imp))` stays calibrated.
4. **Widen the channel:** Tier-1 tags now, Tier-2 rationale embeddings as the headline addition.
5. **Extend the ablation table:** importance-only / +tags / +rationale-embedding, all vs the λ=0
   control — shows WHICH part of the distillation carries the gain (the scientific payoff).
6. If Stage-1/2 shows signal → Tier-3 temporal labels.
Richer supervision does NOT change the odds that explicit relevance beats implicit attention — it
changes the effect size IF there is one, and makes a negative result more informative.

### 23.4 Prior art / novelty — CRITICAL for the paper framing
- **RSD — "Risk Semantic Distillation from VLM" (arXiv 2511.14499, Nov 2025) is the closest prior
  work:** Qwen2.5-VL produces per-object risk scores+rankings, distilled via an aux "RiskHead" on
  BEV features into VAD, no VLM at inference, evaluated closed-loop on Bench2Drive long-tail.
  Verified by reading the paper: **they run NO unsupervised control** (no same-head-no-VLM arm; only
  VAD vs VAD+RSD) — exactly the §19 "manufactured false positive" confound.
- **VLM-AD (arXiv 2412.14446):** scene-level freeform reasoning text + structured action labels as
  aux supervision on UniAD/VAD/SparseDrive; gains on nuScenes + CARLA Town05; also no λ=0-style
  control. **DiMA:** MLLM→vision-planner feature distillation. **Alpamayo-R1:** reasoning traces
  inside a full VLA (different mechanism; cite).
- CONSEQUENCE: the claim "we distill VLM relevance into an e2e driver with no inference cost" is
  **occupied territory** (twice). "Our method" framing will not survive review.
- **The distinct, defensible axes (in order of strength):**
  1. **The controlled question as the headline:** "Does VLM relevance supervision teach an e2e
     planner anything its attention doesn't already learn?" — λ=0 control + random-token control +
     fair-baseline rule + attention probe = the contribution. Robust to outcome: row4>row3 is a
     positive nobody cleanly established; row4≈row3 is a clean negative that challenges RSD/VLM-AD's
     implied mechanism.
  2. **Supervision-as-input vs supervision-as-regularizer:** RSD = aux head on BEV features (pure
     loss-side); ours = tokens INJECTED into planner memory, consumed at inference. Nobody has
     isolated this axis → **add ablation row: ZOI supervision through aux head only (no token
     concat) vs full token injection.**
  3. **Which supervision content carries the gain** (§23.3 step 5) — VLM-AD/RSD each pick one
     format, never compare. Object-grounded rationale embeddings specifically are an open slot
     (VLM-AD text is scene-level; RSD is scalar+rank).
- Thesis bar: comfortably cleared (controlled study, honest negatives OK). Conference bar: cleared
  ONLY with the analysis framing; cite RSD + VLM-AD prominently as the methods the controls
  interrogate.
- **Benchmark note:** RSD used Bench2Drive → sharing a benchmark (Bench2Drive or NAVSIM, §20.3)
  for at least one table makes the comparison legible rather than asserted.

### 23.5 TODO delta from this session
- [ ] §22.4 label fixes (unchanged, still first).
- [ ] Add Cosmos-Reason2-8B backend to `run_vlm_multiframe.py` / labeler; bake off vs Gemma-4.
- [ ] Ranking+BCE importance loss; self-consistency sampling + confidence weights in the labeler.
- [ ] Tier-1 structured tags in prompt+parser+labels (extend .npy schema or sidecar); Tier-2
      rationale text + frozen-encoder embeddings; heads on ZoiModule for both.
- [ ] New ablation rows: aux-head-only (no token concat); importance-only vs +tags vs +rationale.
- [ ] Related-work pass: RSD (2511.14499), VLM-AD (2412.14446), DiMA, Alpamayo-R1 (2511.00088).
- NOTE: §23.5 is superseded by the §24 master plan (same items, now sequenced + extended).

## 24. MASTER PLAN (2026-07-06): architecture upgrades + labeling budget + phased execution
Consolidates §22-§23 + the architecture review of `zoi_module.py` into one sequenced plan.
User confirmed MORE LABELING BUDGET is available — but scale AFTER quality+schema are fixed, not before.

### 24.1 Architecture verdict on ZoiModule (reviewed 2026-07-06)
KEEP the DETR-style module (right substrate for §23.2 per-object tag/rationale heads; sees the
genuine 32×32 mid-stage, unlike CenterNet's detail-free 64×64 — §3). But 4 trainability upgrades,
motivated by: **set-prediction from scratch on ~5k frames with a FROZEN backbone is the biggest
training risk.** Plain-language why: the loss must Hungarian-pair 20 unordered outputs with M
labels every step; with random init the pairing flickers (pedestrian graded against slot 7 one
step, slot 12 the next) → queries get contradictory gradients → DETR's infamous slow convergence
(~500 epochs on 118k COCO images). Frozen backbone makes it worse: features never adapt to meet
the queries halfway, the from-scratch module does 100% of the accommodating.
1. **Anchor queries (DAB-style "home addresses").** Each query gets a learnable (x,y) reference
   point; its positional part = `ContinuousPositionEmbedding(ref_xy)` (class already exists);
   `predict_xy` outputs a DELTA from the reference, not an absolute position. Objects then match
   consistently to the nearest-home query → assignment stops flickering from step 0.
2. **Denoising queries (DN-style "training wheels") — nearly free HERE because label xy is exact
   CARLA GT.** Train-time only: append extra queries built from jiggled GT positions, graded
   against their source object BY CONSTRUCTION (no Hungarian). Clean gradients train the SHARED
   decoder + xy/importance heads while learned queries sort out territories. Two masks keep it
   honest: DN queries NEVER enter the planner concat (GT leak) and don't exist at inference
   (parity §4 untouched).
3. **Zero-init injection gate.** `model.py:342` currently concats RANDOM tokens into a PRETRAINED
   planner's memory at step 0 → perturbs it before ZOI knows anything, muddies the fair baseline.
   Fix: learnable scalar gate (init 0, Flamingo/LayerScale style) on `zoi_tokens` → at step 0 the
   model IS exactly the pretrained baseline; planner opens the gate as tokens become useful.
4. **Gating range [1,2] → [0,1].** `token * (1+sigmoid(imp))` (`zoi_module.py:114`) never
   suppresses: an "irrelevant" query still injects at full 1× (≈15 junk tokens/frame at N=20,
   M~2-8). Switch to plain `sigmoid(imp)` so learned irrelevance attenuates; dead-gradient worry
   is covered by the direct zoi_loss path into the importance head. Keep [1,2] as ablation note.
Minor: `norm_first=True` on the decoder layers (pre-norm, stabler small-scale); optionally
restrict each anchor query's cross-attn to a local neighborhood of its reference.
NOT doing: Flavor 1A/CenterNet-head reuse (worse features), content-free tokens as main design
(ablation row only), fancier planner interfaces (§20.3 — risk is redundancy, not bandwidth).
FALLBACK if Hungarian training still unstable after 1+2: dense importance heatmap on the 32×32
(CenterNet Gaussian+focal, machinery in-repo), top-K cells + offset head → tokens. Converges
reliably on small data but per-object rationale/tag heads attach less naturally.

### 24.2 Does more data help, and what to label (user has budget)
More frames help EXACTLY here: (a) Stage-2 verdict power — the measured subset is the ~10-30%
safety-critical frames, so 5k frames ≈ only ~0.5-1.5k critical frames; scaling to 10-15k roughly
triples the frames the verdict is computed on; (b) within-frame pairs for the ranking loss;
(c) walker/rare-event coverage. More data does NOT fix teacher noise (self-consistency does) or
the redundancy risk (only the λ=0 control answers that). Supply is not binding: ~1,213 trainable
routes in the Kaggle dump (§16) >> 15k frames at stride 5.
**SEQUENCING RULE: do not burn the big run early.** One VLM call can return score+tags+rationale
together, so the marginal cost of the §23.2 rich schema is ~zero — but only if the schema exists
BEFORE the big run. Big labeling waits for Phase 2.
Three label products:
- **P1 gold eval set (~300-500 frames):** best teacher available (Gemma-4 or Cosmos-Reason2-32B
  on A100), 3× self-consistency, human spot-check ~100 frames. Uses: alignment-IoU eval,
  teacher-comparison ground truth (kills the recurring small-n caveat, §17/§18a), label-noise est.
- **P2 main train set (~10-15k frames, up from 5k):** winning teacher, rich schema, completeness
  guard; 2-3× self-consistency if A100 (else 1× + retry-on-omission on T4). Mix biased hard to
  PedestrianCrossing/DynamicObjectCrossing/HighwayCutIn + ParkedObstacleTwoWays negatives +
  SignalizedJunctionLeftTurn; small noScenarios share. Town13 held out (unchanged).
- **P3 teacher-compare set (~150-200 frames):** both finalist teachers on identical frames,
  scored against P1 gold → picks the P2 teacher rigorously.
Budget math @T4 ~13s/frame/pass: P2@10k×1 ≈ 36 GPU-h; ×3 ≈ 108 GPU-h (~4.5 T4-days; A100 ~3-4×
faster). If forced to choose: frames > passes for P2 (ranking loss tolerates noise), passes >
frames for P1 (it's the measuring stick).

### 24.3 Phased execution plan (each phase gates the next)
**Phase 0 — label quality + teacher choice (blocks everything).**
0a. Implement §22.4 in labeler: exposure-normalize, completeness guard (retry→invalidate,
    log omission rate), richer object lines (longitudinal+lateral+radial).
0b. Add Cosmos-Reason2-8B backend (§23.1); tune its prompt fresh (§18a rule); structured
    final-answer format + generous max tokens (CoT model).
0c. Re-label the 200 probe frames (both finalists) → P3; re-run `attention_probe.py` +
    `analyze_importance_residual.py` on CLEAN labels.
    GATE: att-resid↔imp-resid still weak (~≤0.3) → proceed. Strong (≳0.5) → §20.4 pivot
    (NAVSIM/PlanT or accept the analysis-paper-of-a-negative framing).
**Phase 1 — architecture + loss upgrades (~1 day code, before any big training).**
1a. §24.1 items 1-4 (+norm_first) in `zoi_module.py`/`model.py`.
1b. Importance loss → within-frame pairwise ranking + down-weighted BCE (§23.3.3);
    self-consistency variance → per-label loss weight.
1c. Standalone shape/gradient test (repeat §14 procedure); then Stage-1 learnability run on
    ~1-1.5k frames (existing-style labels fine here).
    GATE: zoi_loss drops + pred importance correlates with held-out VLM labels → proceed.
**Phase 2 — supervision widening (schema BEFORE the big run).**
2a. Prompt+parser emit score + Tier-1 tags (crosses_ego_path / suggested_ego_response /
    time_criticality) + one-sentence rationale in ONE call; extend .npy schema or sidecar;
    validity-mask any unparsed field (never default). Update `verify_zoi_coverage.py`.
2b. Rationale → frozen text-encoder embedding (SigLIP-text or sentence-transformer), stored
    alongside labels; cosine-loss head + tag heads on matched queries in ZoiModule.
**Phase 3 — the big labeling run (§24.2): P1 gold, then P2 main.**
**Phase 4 — training + extended ablation table.** Rows (all fine-tuned identically, Town13 out):
  1 baseline / 2 random-token / 3 λ=0 unsup / 4 importance-only / 4b +tags / 4c +tags+rationale /
  5 aux-head-only (supervision w/o token concat — the RSD-mechanism row, §23.4 axis 2) /
  6 content-free tokens (posenc×imp only) / 7 [1,2] vs [0,1] gate (cheap, optional).
  VERDICT unchanged: 4x vs 3 on safety-critical subset; 4c vs 4 shows what content carries gain.
**Phase 5 — eval + paper.** Closed-loop per §19; consider one Bench2Drive or NAVSIM table for
  legibility vs RSD (§23.4); write-up = analysis framing (controls as headline).
Kill-switches recap: Phase-0 probe (strong correlation), Phase-1 learnability (can't fit labels),
Phase-4 (4≈3 AND 4c≈3 → clean negative, still a thesis).

## 25. Teacher bake-off kit (2026-07-06) — portable, runs on a friend's machine
Phase-0b/0c implementation. STATUS NOTE: the §22.4 fixes turned out to be ALREADY IMPLEMENTED in
`generate_zoi_labels_gemma4.py` (normalize_exposure, completeness retry guard, describe_object_line
rich lines — §22.5 TODO item 1 is DONE); the bake-off kit reuses the same logic.

### 25.1 The kit (4 self-contained files, no repo/carla needed on the target machine)
`teacher_bakeoff.py` + `eval_bakeoff.py` + `zoi_projection.py` + `zoi_prompts.py` (+
`README_BAKEOFF.md` with env/VRAM/run instructions). Points at the RAW dump (gz-aware,
double-nesting handled). Teachers: `gemma4` (current baseline), `cosmos8b`/`cosmos32b`/`cosmos2b`
(NVIDIA Cosmos-Reason2; Qwen3-VL-based, loaded via Qwen3VLForConditionalGeneration with
AutoModelForImageTextToText fallback; `--load_4bit` for 32B on <70GB), `qwen3vl8b` (Cosmos's BASE
model — isolates NVIDIA's post-training delta), `internvl` (known reference), `mock` (no-GPU
pipeline check, exercises the retry path deterministically).
- Cosmos = reasoning models: model-card `<think>/<answer>` system instruction, 4096-token budget,
  parser strips the think trace and takes the LAST parseable `{"scores":...}` JSON.
- **RICH_PROMPT** added to `zoi_prompts.py`: ONE call returns per object `imp` 0-5 + Tier-1 tags
  (`path` yes/maybe/no, `act` brake/yield/monitor/ignore, `urg` now/soon/none) + `why` (≤15-word
  rationale) — the §23.2 widened channel at ~unchanged per-frame cost. Exemplar anchors kept.
- All §22.4 guards active: exposure norm, rich object lines, retry-then-DROP (never silent 0.0);
  out-of-vocab tags → None (counted, never guessed).
- Frame list pinned in `<out_root>/frames.json` (first run samples balanced per-scenario
  round-robin across routes; later runs reuse → every teacher sees IDENTICAL frames).
- `--passes K` self-consistency (pass 0 greedy + K-1 sampled temp 0.7; imp=mean recorded with
  imp_std, tags=majority).
- Outputs per teacher: `labels/<route_rel>/zoi_labels/*.npy` ([M,4], drop-in for
  `attention_probe.py --labels_root`), `rich_records.jsonl` (full per-object record incl. tags/
  rationale/omissions/timing), `overlays/` scored marks ("MISS" = dropped), `raw/`, `summary.json`.

### 25.2 eval_bakeoff.py metrics (numpy-only)
Per teacher (VLM-scored car/walker only, rule rows excluded per §21.2): spread std, range,
histogram, depth-ρ (tie-averaged Spearman), top1-closest, **nondist_residual** = 1−R² of imp ~
[closeness, |lateral|] (THE signal-quality number — share of teacher judgment not free from
geometry), omission rate, parse fails, tag completeness, `imp_by_path` monotonicity,
contradiction rate (imp≥4 with path=no/act=ignore), rationale coverage/length, s/frame.
Pairwise: Spearman/Pearson on shared objects + top-disagreement table WITH both teachers'
rationales (eyeball who is right). Ranking heuristic: reliability gate (omission>0.10 or parse
fails>5%) first, then residual + 0.5ρ + 0.1spread. Writes `bakeoff_report.md` +
`bakeoff_metrics.json`.

### 25.3 Protocol
1. Friend: mock run (pipeline check) → real teachers (gemma4+cosmos8b minimum; cosmos32b
   `--load_4bit` if VRAM-bound; cosmos2b/qwen3vl8b/internvl optional) → eval → zip `bakeoff_out/`.
2. Us, on return: run `attention_probe.py --labels_root bakeoff_out/<teacher>/labels` per teacher
   — the PROJECT verdict metric (att-resid↔imp-resid per §21.3) on each teacher's labels; a
   teacher whose residual attention correlation is LOW has the most non-redundant signal to teach.
3. Decision updates §18a (current: Gemma-4+exemplar) and picks the P1/P2 teacher for §24.2.
Caveat: bake-off metrics are directional at ~200 frames (same small-n caveat as §17/§18a); the
gold set (§24.2 P1) is what settles it.
