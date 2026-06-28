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
