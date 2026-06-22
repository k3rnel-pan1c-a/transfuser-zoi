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
TODO (next):
- [ ] Update label generators to start at `skip_first` and read `rgb/` (augment off).
- [ ] Verify the 3 config constants; run `--limit 30 --debug_dir` and eyeball overlays
      (esp. projection vertical sign).
- [ ] Download subset scenarios; download pretrained TF++ weights.
- [ ] Write `ZoiModule` (`ContinuousPosEnc` + DETR queries) + `transfuser.py` 32×32 return
      + 3 wiring spots in `model.py` + `config.py` flags
      (`use_zoi, zoi_num_queries, zoi_lambda, zoi_src_stage`).
- [ ] Write batched `zoi_loss` (Hungarian) + `CARLA_Data.__getitem__` loads `.npy`
      (pad to fixed M_max + validity mask).
- [ ] `--load_file strict=False` + two-LR param groups in `train.py`.
- [ ] Train baseline (fair) + ablation rows.
