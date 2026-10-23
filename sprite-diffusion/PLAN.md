# Sprite diffusion: a tiny 32×32 pixel-art model on the NPU

## Latest work (2026-10-23)

- The October 20 handoff below is historical: the move to model-experiments completed in
  `9a034e3`, local branch `sprite-diffusion`, worktree `C:\code\model-experiments-sprite`.
- Owner sampling choice after same-seed 8/16/32/256-step sheets: **16 normal, 8 fast**.
  Some sprites improved dramatically at 32; the 256-step sheets were worse overall.
- Existing dataset verified: 383,103 sprites, 165,184 labelled, 217,919 unlabelled.
  Weak-category labels: structure 4,925, nature 5,244, food 6,309, weapon 7,608.
- Owner priority: fill those four categories with more good designs. Horizontal flips
  already run with probability 0.5. Palette swaps approved as additional augmentation.
- Owner also notes that generated `creature` sheets are almost all people, not monsters.
  Treat monster coverage as weak too. Keep the single output category, but audit and
  track human / monster / unknown subtypes separately for acquisition and later sampling;
  the 100,133 combined creature labels must not be treated as evidence of monster coverage.
- Added on-the-fly, luminance-preserving palette variation for those four categories:
  probability 0.5, hue +/-25 degrees, saturation 0.9-1.1. Outlines, neutrals, highlights,
  alpha and geometry are protected. Text training defaults to no swaps because colour
  captions must stay correct. Preview command: `sprite-train/preview_augmentation.py`.
- Augmented variants are **not new distinct designs**. Keep targeted source collection
  focused on structure/furniture, plants/rocks, actual foods as well as potions, and a
  wider selection of weapon shapes, plus nonhuman monsters. Previous no-AI exclusions
  stay in effect. The real-data palette preview also exposes questionable structure and
  nature labels; clean labels matter alongside count and variety.
- Training code previously sampled uniformly over all sprites; category/subtype modes
  are now implemented and tested (see the update below). Acquire genuinely unseen
  validation packs before the next controlled continuation.
- Best checkpoint remains `C:\dev\sprite-data\runs\q-p2-cat\ckpt.pt` at 100k training
  steps. This change does not retrain or replace it.
- First targeted recovery pass produced a separate pool of **1,056 candidate images**
  absent from the canonical dataset after exact/horizontal-mirror deduplication:
  structure 294, nature 35, food 116, weapon 247, monster/animal creatures 364. The
  filename-family counts are only a design proxy, not verified distinct-design counts.
  Most came from already-downloaded Dungeon Crawl and food archives: the old global
  500-per-pack cap had left out many useful native sprites. Kenney's 16px grid also
  yielded whole 32px trees and beds when its gutters were removed and the authored
  adjacent cells joined.
- Source archive provenance, per-image metadata, complete review galleries and the
  separate NPZ are in `C:\code\model-experiments\artifacts\sprite-data-expansion`.
  New creature rows retain `creature_subtype=monster`; 84 existing monster images also
  have subtype annotations. 442 existing-image label repair **candidates** are recorded,
  not applied. The original pool has not been overwritten or used for new training. Source groups overlap
  the old dataset, so these packs are not a genuinely unseen validation set.
- Quality audit: OGA `trees-13` contains 111 original tree PNGs, all larger than 32px
  (smallest maximum dimension 377px). The old cutter turned that pack into 500 small
  fragments: 147 labelled nature, 89 creature, 264 unlabelled in the raw shard. Proposed
  quarantine for the next build is recorded in `proposed-exclusions.json`; the canonical
  data remains unchanged. New collection excludes masks/shadows, flat auxiliaries,
  oversized images and ambiguous categories. Nature remains the biggest acquisition gap.
- Prepared a separate `trainset-expanded.npz` in that artifact directory: 383,660
  images = 383,103 originals - 499 verified images from the quarantined tree-fragment
  source group + 1,056 recovered images. Removed labels: nature 147, creature 89,
  unlabelled 263. The original `C:\dev\sprite-data\trainset.npz` is unchanged.
- Expanded data retains trusted creature subtypes from pack labels and the new named
  monster sources: human 1,634, monster 2,144, unknown 96,630. Unknown is deliberate;
  generated human bias does not establish the subtype of every training sprite.
  `prepare_training_data.py` and `dataset-preparation.json` record the process.


- Continuation tooling update: explicit `uniform` (compatibility default), `category`,
  and experimental `category-subtype` samplers. Balanced modes allocate 10% unlabelled
  by default and split the remaining mass equally over populated categories. Subtype
  mode splits creature mass over populated trusted human / monster / unknown pools,
  keeping unknowns unknown. No change to the model's single creature output category.
- Real expanded-data audit: 200k draws per mode, structure/nature each ~11.25% versus
  ~1.3% under uniform sampling; trusted monster exposure ~3.75% versus 0.56% in subtype
  mode. This is exposure, not quality or additional designs. Small trusted subtype pools
  may overfit; compare a matched uniform control with the same augmentation settings.
- `--init-from` starts a separate category-only run with existing model/EMA weights and
  fresh optimizer/step; sliced runs resume their own checkpoint. Sampler distribution
  and RNG state are saved; incompatible sampling changes on resume are rejected.
- Statistical tests and CPU initialization/resume smoke checks pass. Audit and smoke
  outputs: `C:\code\model-experiments\artifacts\sprite-training-next`. No real model
  training has been launched. Unseen validation source acquisition remains outstanding.

---

Status (2026-10-05):
- Phase 1d is done on branch `feat/sprite-diffusion` (uncommitted): the labeller bake-off
  and the self-training experiment, which was negative.
- Next is 1e, the harvest to the data targets.
- Decisions 3, 4 and 5 are taken; decisions 1 and 2 are open.

Goal: a small diffusion model that draws pixel art from a label and a short description.
- Output: 32×32 RGBA, with a size tag, so 16×16 sprites can still be requested.
- Labels cover characters, monsters, items, weapons, food, terrain, structures and
  effects.
- It trains on this laptop's iGPU and runs on the NPU.

Constraints:
- The user accepts ripped game sprites as training data.
- Weights trained on ripped data stay local. They do not go on a public HF repo or ship in
  the app.
- Raw data lives outside the repo. Tools go in `utilities/`.

---

## What we already know (measured 2026-10-01)

**The course dataset is small.** DeepLearning.AI's "How Diffusion Models Work" set is
`sprites_1788_16x16.npy` (HF mirror `ashis-palai/sprites_image_dataset`).
- It has 89,400 entries but only **1,722 unique** sprites, each base sprite repeated ~50×.
- RGB only, no alpha, white background.
- Unique sprites per class: front characters 120, monsters 642, food 120, items 680,
  side-facing characters with weapons 160.

It is a usable seed set and gold-label source, but too small on its own.

**The Spriters Resource (TSR) can be crawled with plain HTTP.**
- No Cloudflare challenge. robots.txt only disallows `/browse/`, `/random`, the sort
  parameters, and the account and editing pages.
- Game page: `/<platform>/<game>/`. It lists sheets by title, grouped under sections.
- Sheet page: `/<platform>/<game>/asset/<id>/`. Its image is
  `/media/assets/<dir>/<id>.png`, an RGBA PNG.

Sheet titles alone already map to categories. On NES Zelda: "Overworld Enemies",
"Bosses", "Items & Weapons", "NPCs", "Overworld Tileset", "Dungeon Tileset", "Link".

**TSR sheets are messy.** Samples are in the scratchpad (`tsr_sheet.png`). They contain:
- text labels drawn into the sheet ("Octorok", "Moblin"), which are useful as hints and as
  name labels;
- credit boxes, palette swatches and VRAM dumps;
- tiles drawn inside grid lines;
- solid-colour backgrounds in several colours;
- multi-part sprites and sprite-plus-projectile pairs.

Extraction needs real work, and it is most of Phase 1.

**Sprite sizes: most sprites are bigger than 16×16.**
- Measured with `utilities/sprite-data/size_probe.py` on 9 games, ≤18 sprite sheets each,
  tile/map/UI sheets skipped.
- Each sheet was cut into connected components on its background, deduplicated, and
  counted as the cumulative share that fits each canvas:

| platform | unique sprites | ≤16×16 | ≤16×32 | ≤32×32 | ≤64×64 |
|---|---:|---:|---:|---:|---:|
| NES (Zelda, SMB3, FF) | 1,787 | 28% | 47% | 76% | 92% |
| SNES (ALttP, Chrono Trigger, Secret of Mana, EarthBound) | 8,839 | 14% | 33% | 73% | 97% |

- The 16×32 and 32×32 buckets are clean SNES characters, NPCs and monsters (contact sheets
  `gain_16x32.png`, `gain_32x32.png` in the session scratchpad).
- On SNES the 32×32 bucket is twice the size of the 16×32 one, because monsters and
  action poses are wide.
- Error sources:
  - tightly packed sprites merge into one component, which inflates sizes;
  - detached weapons and projectiles split off, which deflates sizes;
  - many components are animation frames of the same character.
- GB/GBC was left out: its sample was dominated by 56×56 Pokémon battle sprites.
- At ≤32×32 the sample yields ~1,600 sprites per SNES game from 18 sheets.
- Raw crawl cache: `C:\dev\sprite-data\tsr_cache`.

**Training throughput on this laptop.** Settings: klein-style blocks, head dim 128, 64
tokens, batch 256, bf16 autocast, AdamW. Script: `utilities/sprite-train/bench.py`.

| model | iGPU (C:\dev\igpu-venv, ROCm gfx1150) | CPU |
|---|---:|---:|
| h256 × 8 blocks, 6.8M | 620 samples/s | 175 |
| h384 × 10 blocks, 19.2M | 325 samples/s | 68 |
| h512 × 12 blocks, 41M | 154 samples/s | 36 |

- A 32×32 canvas at patch 2 is 256 tokens, so expect roughly 4× slower (unmeasured).
- The CPU is ~5× slower than the iGPU, so it is not the training device.

---

## Phase 1: data (the next step)

Tools go in `utilities/sprite-data/`. Data goes in `C:\dev\sprite-data\` with subfolders
`raw/`, `sprites/` and `manifest.parquet`.

Most of this phase's time is unattended machine time:
- the crawl at 1 request/s;
- the labelling batch.

Nothing waits on the user. There is no hand labelling (user, 2026-10-01), and contact
sheets are optional QA.

### 1a. Crawl TSR (`crawl_tsr.py`)
- Platforms, in order of yield at ≤32×32: SNES and NES first, then GBA, Genesis,
  GB/GBC, Master System, Game Gear.
- For each game, record the section heading, sheet title, game and platform for every
  sheet. Store the PNG plus a JSON sidecar.
- Also fetch map sheets. They are the source of the 32×32 terrain chunks (1b).
- Rate: 1 request/s. Cache everything and make the crawl resumable. Paths disallowed by
  robots.txt are never requested.
- Measure yield on a 50-game SNES sample before crawling a whole platform.

### 1b. Extract sprites up to 32×32 (`extract.py`)
1. **Background.** Take alpha == 0. Otherwise flood-fill the dominant colour from the
   sheet border.
2. **Pixel scale.** Some sheets are upscaled 2× or 3×. Find the GCD of run lengths and
   reduce by it.
3. **Grids.** Detect regular 16 px lattices, including ones with 1 px gutters or grid
   lines. Slice tilesets cell by cell.
4. **Terrain.** Cut 32×32 windows from map sheets, aligned to their 16 px tile grid.
   These are 2×2-tile chunks, so they carry tile transitions. Cap near-identical windows,
   such as plain grass.
5. **Components.** Otherwise:
   - find 8-connected foreground components;
   - merge components whose bounding boxes are ≤1 px apart;
   - keep a component if its bbox is ≤32×32 and ≥5×5.
6. **Junk filters:**
   - font glyphs: ≤8×8, one or two colours, in horizontal runs;
   - palette swatches: solid rectangles;
   - credit boxes and VRAM dumps: large, or full of glyphs;
   - framed boxes behind sprites (seen on EarthBound sheets).
7. **Place.** Put each sprite on a 32×32 transparent canvas. Centre it horizontally and
   bottom-align it, so characters stand on the same baseline. Record its **footprint**:
   the smallest of 16×16 / 16×32 / 32×32 that holds it.
8. **Dedupe.**
   - Exact duplicates: hash the RGBA bytes.
   - Palette swaps: the same alpha mask and luminance rank pattern. Keep at most 4 per
     group; they double as augmentation.
   - Animation frames: keep them all, and group them (1d labels each group once).

Per-sprite records carry provenance: platform, game, sheet id, section, title and bbox.
Nothing is downsampled; anything over 32×32 is dropped.

### 1c. Clean packs (`packs.py`): the gold-labelled subset
These packs are grid-aligned and labelled by folder and file name:
- the course set (1,722);
- DawnLike 16×16 (CC-BY 4.0);
- the Kenney 16×16 packs (CC0): Tiny Dungeon, Tiny Town, 1-Bit Pack, Pixel Platformer;
- Ninja Adventure (CC0);
- 0x72 DungeonTileset II (CC0).

Pack licences are from memory, so verify them on download. They hardly matter here, since
the weights stay local.

They do two jobs: training data, and the answer key for checking the auto-labeller.

Also inspect `evilsocket/alucard-sprites` on HF (100k–1M sprites, 128×128, CLIP-captioned,
source not stated). If it was nearest-upscaled from small sprites, it reduces exactly to
the native size and is worth adding. If it was resampled, skip it.

### 1d. Label (`label.py`): fully automated

There is no hand labelling (user, 2026-10-01).
- The answer keys come from data that labels itself: packs and the course set.
- On TSR, label noise is estimated from agreement between independent labellers, not
  measured against truth.
- Label noise blurs the conditioning; it does not break training.

**Bake-off result (2026-10-03): no local vision-language model is worth running.**
SigLIP 2 plus sheet metadata wins on every field. Details below.

**Fields (revised by the bake-off):**
- **category** (one of): creature (people, monsters, animals, bosses), item, weapon, food,
  terrain, structure, nature (trees, rocks, plants), effect (projectiles, magic,
  explosions), ui. `ui` is extracted but excluded from training. Character and monster
  were merged on 2026-10-04 (decision 5).
- **role** (optional): hero / npc / enemy / boss, from the TSR section.
- **facing**: front / back / left / right / none.
- **era**: 8-bit or 16-bit, taken from the platform.
- **footprint**: from 1b.
- **colours**: the 1–2 dominant hues, computed from pixels.
- **name** (optional): from sheet titles that are names ("Crono", "Taros & Moblin").
- **Dropped: subject, caption, pose.** No local model gets them right at this size:
  - subject: ≤17% top-1;
  - pose: ≤6%.

**Sources (each label records its source and a confidence):**
1. **Deterministic, no model:** era, footprint, colours, game, and name.
2. **Category seeds** from section and title keywords: "Enemies" → monster, "Tileset" →
   terrain/structure, "NPCs" → character, "Items" → item.
   - Tile cells (tilesets, maps) get their category from the seed alone; every pixel
     labeller scored ≤33% on them.
3. **Frame groups.** Cluster each sheet's sprites by shape and palette, so one group is
   one character's animation frames. Group-level fields are labelled once per group.
4. **Facing.** Two sources:
   - **Mirror pairs within a group:** two sprites that mirror each other are left/right;
     a self-mirror sprite is front, back or none.
   - **A SigLIP facing probe** (70% on 4-way pack keys), trained with flipped copies.
5. **SigLIP 2 votes on category**, run on every sprite:
   - siglip2-so400m zero-shot (69% on whole sprites, ~0.5 s/sprite on the iGPU, so ~14 h
     per 100k);
   - a linear probe on SigLIP embeddings, trained on packs plus high-precision TSR seeds.
6. **Reconciliation.**
   - Category is a 2-of-3 vote between the seed, so400m zero-shot, and the probe (the probe
     breaks ties).
   - With no majority, the sprite is kept but unlabelled. It trains only in the
     unconditional (CFG-dropout) slot.
   - Votes among the pixel models alone don't help: see "Ensembles" below. The vote is
     only useful because the seed's errors are independent of the pixel models' errors.

**Automated answer keys (built by `bakeoff_keys.py`):**
- **Sources:** the course set, DawnLike, Ninja Adventure and DungeonTileset II, 9,597
  sprites in all.
- **Category:** pack folder, file and sheet names, plus the course set's classes.
- **Facing:** Ninja Adventure's walk/idle/attack sheets (columns are front/back/left/right),
  DT2 heroes (face right), and the course set's front and side classes.
- **Eval set:** 622 sprites. 449 are whole sprites; 173 are tile cells, i.e. grid cells
  spanning their full width or height, which are often slices of a bigger object.
- **TSR (no truth):** 240 sprites from 104 cached sheets, scored by seed / probe / model
  agreement.

Packs are cleaner than TSR and drawn in a different style, so pack accuracy is optimistic
for TSR.

**Labelling is local only. No cloud APIs (user, 2026-10-01).** It is offline data prep, so
the iGPU and CPU are fair game alongside the NPU.

**Bake-off results (2026-10-03).** Whole sprites only. Category is the mean of the
per-category accuracies over character, monster, item, weapon, food and effect.

| entrant | category | char+monster merged | facing | pose | subject top-1 / top-5 | s/sprite |
|---|---:|---:|---:|---:|---:|---:|
| siglip2-so400m zero-shot | **69%** | **74%** | 56% | – | – | ~0.5 (0.72 contended) |
| siglip2-base zero-shot | 57% | 61% | 61% | – | – | 0.033 |
| siglip2-base probe (leave-one-pack-out) | 52% | 54% | **70%** | – | – | 0.033 |
| qwen3vl-it:4b, NPU (FLM 1.0.2) | 45% | 49% | 28% | 6% | 10% / 28% | 4.6 (clean) |
| gemma3:4b, NPU (quarter set) | 43% | 48% | 40% | 6% | 17% / 37% | 9.6 |
| Florence-2-large, caption → category | 34% | – | – | – | 14% / 37% | 2.3 |

- **The VLMs are slower and worse than SigLIP on every field.**
  - qwen3vl answers "front" for 229 of 230 facing questions.
  - gemma3 confidently invents scenes (a course hero became "a rusty robot tending a
    broken circuit board").
  - Neither names objects reliably.
- **Tile cells:** every entrant scores ≤33%. Metadata only.
- **TSR, qwen3vl with sheet-title hints:**
  - seed = VLM 82%, seed = probe 43%, probe = VLM 40%;
  - the VLM mostly repeats the hint, so it is not an independent voter;
  - a probe trained on packs alone transfers poorly to TSR's art.
- Timings marked "contended" ran while the NPU job was also running.
- Raw outputs and reports: `C:\dev\sprite-data\bakeoff\` (`report-sprites.md`,
  `report-tiles.md`, `out\`).
- Scripts in `utilities/sprite-data/`: `bakeoff_keys.py`, `bakeoff_tsr.py`,
  `bakeoff_vlm.py`, `bakeoff_siglip.py`, `florence_caption.py`, `bakeoff_score.py`,
  `itch_fetch.py`, `spritekit.py`.

**Ensembles (2026-10-04, `bakeoff_ensemble.py`, `report-ensemble.md`).** Do combined
labellers beat the best one alone? No. Whole sprites, character+monster merged:
- so400m alone: 74% balanced.
- 3- and 5-model majority votes: 68–72%.
- Soft fusion of the SigLIP probabilities: 69–73%.
- The oracle (any of the five right) is 83%: below the 85% gate, because the models
  share blind spots.
- Why: so400m is far ahead, and the others either share its errors (the other SigLIP
  models) or are much weaker (qwen3vl, Florence), so they outvote it when it is right.
- As a confidence filter, three-SigLIP agreement keeps 56% of sprites at 83%.
- Conclusion: more voters won't reach the gate; a domain-adapted model (self-training)
  is needed.

**Next experiment: self-training the probe.**
- The gap to the 85% gate is mainly style transfer: the probe never saw TSR art.
- Steps:
  1. Crawl ~100 more games.
  2. Take the high-precision seeds (sections "Playable Characters", "Enemies",
     "Weapons", "Items", …).
  3. Train the probe on packs plus those seeds, using so400m embeddings.
  4. Score it on held-out games: agreement with seeds, and accuracy on the pack keys.
- If a linear probe stalls below 85%, fine-tune SigLIP's last blocks on the same labels.

**Self-training run (started 2026-10-04).** Every step can be paused (stop the process)
and resumed (run the same command); nothing is lost.

| step | command (from the repo root) | resumes because |
|---|---|---|
| crawl ~100 games | `python utilities/sprite-data/crawl_tsr.py >> C:\dev\sprite-data\selftrain_crawl.log` | fixed game order, every request cached |
| cut sprites | `python utilities/sprite-data/selftrain_extract.py` | skips sheets already in `sprites.jsonl` |
| embed, train, score | `C:\dev\igpu-venv\Scripts\python.exe utilities/sprite-data/selftrain.py` | embeddings saved in 512-sprite chunks |
| same with so400m | as above, with `SIGLIP=google/siglip2-so400m-patch14-384` | same |

Notes:
- Seeds come from `tsrkit.SEED_RULES`. Output goes to `C:\dev\sprite-data\selftrain\`.
- Early crawl: SNES sheets give mostly creature seeds. Item, weapon, food and effect
  seeds are scarce, so watch their counts.

**Self-training, first result (2026-10-04, siglip2-base):**
- Data: 100 games crawled (1,598 sheets) gave 22,697 seeded sprites. Seeds: creature
  19,890, effect 1,224, item 1,184, weapon 359, food 40. TSR rarely gives items, weapons
  or food a sheet of their own.
- Results:
  - packs only (leave-one-pack-out): 57%;
  - packs + TSR seeds: 55%;
  - TSR seeds only: 39%;
  - held-out games, agreement with seeds: item 19%, weapon 24%.
- Cause, from a random sample of each seed:
  - the cutter kept **drawn-on text** (weapon names, single letters) and **coloured
    backing boxes**, so most "weapon" seeds were glyphs;
  - "creature" seeds also carry projectiles from enemy sheets. That is inherent seed noise.
- Fixes, both kept for production extraction:
  - `tsrkit.strip_box` makes a uniform outer ring transparent;
  - `selftrain_clean.py` drops text two ways: SigLIP zero-shot "text" beating every object
    prompt by more than 0.015, and, per sheet, any cut-out whose colours all lie inside that
    sheet's text palette (single letters in the same font).
- After the fixes (`sprites2.jsonl` + clean): 20,542 seeded sprites.
  - packs + TSR seeds: still 57% (creature 79, item 54, weapon 59, food 62, effect 32);
  - packs only: 57%;
  - held-out-game seed agreement: item 26%, weapon 34%.
- The palette pass first dropped real NES sprites (their 3 colours fit a sheet's font
  palette). It now also requires SigLIP to lean towards text.

**Verdict: self-training on sheet seeds does not move accuracy.**
- Why: the seeds are 89% creature, and the scarce non-creature seeds stay noisy
  (projectiles on enemy sheets, heroes in "Weapons" sections, leftover boxes).
- By the stated rule, skip the overnight so400m self-training run. Fine-tuning on the
  same labels would inherit the same noise.
- New approach to labelling the training set:
  - pack sprites keep their true labels;
  - a TSR sprite is labelled only where its seed and so400m zero-shot agree (two
    independent sources);
  - everything else trains unlabelled, in the unconditional slot.
- Items, weapons and food then come mostly from the packs; creatures and effects from TSR
  in bulk.
- Next measurement: so400m zero-shot over the 20.5k TSR sprites (~2.5 h on the iGPU,
  resumable). It gives seed/so400m agreement per class, i.e. how much TSR data ends up
  labelled. It is also the production labelling pass itself, not a throwaway.


**Gate (unchanged):**
- ≥85% category accuracy on the pack keys, and ≥75% three-way agreement on TSR.
- Below that, tighten the keyword rules, or drop that category's conditioning.

**Review page** (`review.py`): contact sheets per category and label source. Optional;
nothing in the pipeline waits on it.

**Florence-2 probe (2026-10-01).** Setup:
- 44 sprites: 20 course sprites with known class, 24 cut from NES/SNES sheets.
- Each rendered 8× nearest-neighbour on grey, run on the iGPU in fp16 with 3 beams.

Results:
- **Characters:** usable. "a man in a military uniform", "a boy in a blue and white
  striped shirt", "a man holding a pitchfork" (a Moblin with a trident).
- **Items and food:** poor. A gem became "a heart", a shield "a brown bear", a barrel
  "a cup of coffee", an orange "a square with a red cross".
- **Facing:** never mentioned.
- **`<MORE_DETAILED_CAPTION>`:** unusable. It invents sources ("a character from the video
  game Minecraft", "Bowser") on most sprites.
- **Category, inferred from the large model's short caption:** roughly 2 in 3, below the
  85% gate.
- **Speed:** 1.4 s (base) and 1.8 s (large) per sprite at batch 1. Batched, greedy decoding
  should be much faster, but that is unmeasured.
- **Bonus:** it reads drawn text. The extractor's "Unused" glyph cluster came back as
  "the text 'Unused'", so it doubles as a junk filter.

### 1e. Harvest to the data targets (added 2026-10-05)

**The shortage is labelled sprites in the rare categories, not sprites in general.**

| category | labelled pack sprites | TSR sheets seeded (of 1,598) |
|---|---:|---:|
| creature | 3,304 | 1,444 |
| terrain / structure / nature | 2,020 / 1,825 / 667 | tile sheets not seeded |
| item | 1,223 | 51 |
| food | 243 | 2 |
| weapon | 164 | 15 |
| effect | 151 | 86 |

**Two kinds of data, two jobs.**
- **Total volume** teaches what every category shares: outlines, shading, palettes, how
  a 32 px figure is drawn. Labels aren't needed for it, and the more the better.
- **Labels per category** only teach the model which part of that knowledge to use.
  - Reference point: CIFAR-10 is 32×32 photos at ~5k per class, and class-conditioned
    diffusion produces recognisable classes from it. Sprites are simpler than photos.
  - So ~5k labelled per category is the target. 1–2k is thin but workable.
  - 10× that (20k+ food) likely does not exist in free and ripped pixel art combined.

**Count distinct designs, not sprites.**
- Animation frames, flips and palette swaps inflate counts without adding designs.
- The 20.5k cleaned TSR sprites are probably 3–5k distinct characters.
- Too few designs makes the model copy its training set. That copying is the failure to
  watch for.
- `count_designs.py`: within each sheet, merge sprites whose SigLIP embeddings are
  near-duplicates (threshold set by eye on one contact sheet). Then dedupe across sheets
  on exact alpha mask + luminance pattern (the 1b palette-swap rule).
- Every count from here on reports both sprites and distinct designs.

**Targets (the new Phase 1 gate, below):**
- total ≥200k sprites, ≥50k distinct designs, labelled or not;
- ≥5k labelled sprites (≥1.5k distinct designs) in every category.

**Blocked 2026-10-08: TSR now answers every request with a Cloudflare managed challenge**
(HTTP 403, `Cf-Mitigated: challenge`).
- We do not bypass it. `tsrkit.get` now stops the crawl on it instead of retrying.
- What we have stays usable: 1,598 cached sheets, i.e. the 20.5k cleaned sprites.
- Volume has to come from sources that allow bulk download:
  - Hugging Face pixel-art datasets (check `alucard-sprites` and others, 1c);
  - the Source 2 packs.
- `run_harvest.ps1 -Crawl` keeps the TSR path for if access returns.
- A scheduled task `SpriteHarvest` runs the so400m labelling pass alone
  (`run_harvest.ps1`). Logs go to `C:\dev\sprite-data\logs`. `run_harvest.ps1 -Stop`
  pauses it; re-running it resumes.
- Labeller speed: so400m patch16-256 scores 67% vs patch14-384's 69% (unmerged, whole
  sprites). It runs at 170 ms/sprite with eager attention (ROCm SDPA: 245), so it is the
  production labeller.

**OpenGameArt via the HF mirror (probed 2026-10-09, `oga_probe.py`).**
- Mirror: `nyuuzyou/OpenGameArt-<licence>`. 2D art is 15,548 entries in 64 archives
  (42 GB). Per entry: title, tags and description.
- Probe results:

| archive | unique sprites ≤32 | food | weapon | item | effect | creature |
|---|---:|---:|---:|---:|---:|---:|
| CC0 2D_Art_06 (densest by tags) | 4,379 | 84 + most of 1,691 unmatched | 802 | 1,094 | – | 245 |
| CC0 2D_Art_00 (typical) | 21,206 (11,620 from one generator pack) | 45 | 142 | 940 | 2,101 | 318 |

- Food, weapons, items, creatures and effects are clean native pixel art at 16–32 px.
  Few images are upscaled.
- Problems to fix in the harvester:
  - Cap sprites per pack: one procedural pack gave 11.6k near-identical glyphs.
  - Keyword labels are noisy: UI frames came out as "item". Keep the seed == so400m
    agreement rule.
  - ~40% of archive files don't match a metadata record by file name. Mostly food in
    archive 06, so the matching is worth fixing.
- Rough yield, extrapolated from 2 of 64 archives: 150k–500k raw sprites before design
  dedupe.
- Disk: 51 GB free on C:. Process one archive at a time and delete each zip after
  extraction (needs the user's OK).

**OpenGameArt harvest run (started 2026-10-16, `harvest_oga.py`).**
- The arbiter session told me to build and run it, with a 10 GB disk floor, the source
  zips kept, and local commits only.
- Code lives in worktree `C:\code\OpenFlowLM-sprite` on `feat/sprite-diffusion`, because
  the main checkout is shared with other sessions' branches.
- All 64 archives (40 GB) are in `C:\dev\sprite-data\oga\zips`.
- Stages:
  - `cut` (system python): one shard per archive in `oga/cut/`:
    - `<lic>-<part>.npz` holds the 32×32 canvases;
    - `.jsonl` holds metadata: pack, title, tags, author, licence, seed, upscale,
      footprint.
  - `label` (igpu-venv): so400m embeddings as float16 `.emb-*.npz`, plus `.labels.jsonl`
    (zero-shot category, margin, text margin, label).
  - `report` totals it all.
- Both stages resume by shard. Both wait (and re-check every 10 min) whenever C: drops
  under 11 GB free.
- Choices made without the user, all reversible because the zips are kept:
  - **Per-pack cap of 500 sprites**, sampled at random across the whole pack. It stops
    generator packs (11.6k glyphs from one). Re-cut with a different cap if large real
    icon packs get clipped.
  - **Shards, not one PNG per sprite.** About 4 KB per file on disk would eat the 4 GB of
    headroom.
  - **Text is scored, not dropped.** The so400m text margin is stored per sprite. The
    threshold (0.015) was tuned on SigLIP base, so the filter is applied at training
    time, once checked on a contact sheet.
  - **Seed rule:** the pack's title and tags must name exactly one category; otherwise
    the file or folder name must. A `ui` category catches interface packs, which are
    never labelled.
  - **File matching:** archive members are matched by the decoded file name of the
    download URL. This fixed all 122 unmatched files in archive 06.

**Harvest results, FULL (2026-10-17 17:15; all 64 archives cut and labelled).**
These replace the PARTIAL block below; its extrapolations were too generous for food.
- 380,437 unique sprites ≤32×32 from 7,139 packs. 221k carry a seed (the rest are
  unlabelled or `ui`); 97k carry a *label* (seed = so400m zero-shot).
- Distinct designs (merged within a pack at so400m cosine ≥0.95): **153,736 (40%)**; at
  0.93: 111,991; at 0.97: 215,086. Palette-swap groups alone: 89% stay distinct.
- **Both Phase 1 targets hold:** ≥200k sprites (380k), ≥50k designs (154k).
- `trainset.npz` (`build_trainset.py`, text margin >0.015 and `ui` packs dropped, plus the
  9.6k bake-off pack sprites): **356,828 sprites in 8,723 groups, 97.4k labelled.**

| category | labelled sprites | OGA designs @0.95 | vs 5k target |
|---|---:|---:|---|
| creature | 46,352 | 12,376 | ✓ |
| terrain | 12,018 | 3,802 | ✓ |
| effect | 10,479 | 3,801 | ✓ |
| item | 9,693 | 4,145 | ✓ |
| weapon | 7,474 | 2,332 | ✓ |
| structure | 4,705 | 1,349 | just under |
| nature | 4,454 | 1,545 | under |
| food | 2,271 | 1,431 | **well under** |

- **Weak categories (decision for the user):** food (2.3k), nature (4.5k) and structure
  (4.7k). Food is the real gap: the OGA mirror simply has few food sprites, and the
  seed = so400m rule kept 2,028 of 4,009 seeded. Options and the recommendation (wait for
  the scaling test) are unchanged, but food is now the priority if sources are added.
- Labelling ran 04:00–17:10 under `dibs`, yielding the gpu on request. The TSR label pass
  (20.5k sprites) was started afterwards.

**Food, itch.io packs and relabelling (2026-10-18).** User asked for more food and said
food should include potions ("most games have a big hunk of meat on a bone and nothing
else"); later that weapon, then "everything other than creature", looked weak.
- **Food now means food and drink, potions included.** Changed in the SigLIP prompts
  (`siglipkit.py`), the pack seed rules (`harvest_oga.py`), and the true-labelled packs
  (`build_trainset.py`: `dawnlike:Potion`, `dt2:flask`, `ninja:items-Potion`).
  Potions were most of the missing food: 2.3k → 6.3k labelled.
- **itch.io (`harvest_itch.py`):** 901 candidate packs from the food-ish tags, 214 cut
  into 6,178 sprites, mostly mixed icon packs.
  - **677 packs carry the `no-ai` tag and were skipped.** Not downloaded, so reversible. The
    tag usually means "made without generative AI", not an explicit training opt-out. Skipped
    on purpose, out of respect for those artists. **Decision for the user.** I recommend
    keeping them out.
  - Rate limiting (HTTP 429) is handled with backoff; the site's pages were reachable with
    no challenge.
- **Labeller: `relabel_v2.py`** (calibrated so400m-only labels). Calibrated on the 9.6k
  true-labelled pack sprites, so400m zero-shot reaches 85% precision only for creature
  (0.98 at any margin) and, barely, item and terrain. Weapon, food, effect, nature and
  structure cannot be labelled by so400m alone. Result: creature +31k sprites, item +4.7k,
  terrain +0.7k; nothing else.
- **Probe: `probe_v3.py`** (linear probe on the 71k agreement labels, tested on the packs):
  **53% balanced, so no per-class threshold reaches 85% except creature.** Likely domain
  gap between OGA art and the packs (effect 28%, nature 12%). It adds only the "probe
  agrees with the pack's seed" labels (+8k). Probe-only labels are creature only (12k).
- Label sources are recorded per sprite (`label_src`: agree / family / so400m /
  probe-agree / probe); the older label files are kept as `.labels.v1.jsonl` and
  `.labels.v2.jsonl`.
- **Training set now (`build_trainset.py`):** 383,103 sprites; labelled: creature 100.1k,
  terrain 14.5k, item 14.2k, effect 12.3k, weapon 7.6k, food 6.3k, nature 5.2k, structure
  4.9k.
- **Distinct designs @0.95 (OGA + itch only; packs add a few hundred):** creature 28.3k,
  item 7.1k, terrain 4.7k, effect 4.3k, food 3.3k, weapon 2.3k, nature 1.9k, structure
  1.4k.
- **Assessment:** all non-creature categories clear ~5k labelled (structure just under),
  and distinct designs are ≥1.4k, but creature is 7–20× the others. Weapon (2.3k designs)
  and structure (1.4k) are the thinnest. Training will sample classes evenly rather than by
  count. The scaling test shows whether the thin ones suffer.
- **User offered to generate sprites.** If used: real pixel art ≤32×32 with transparent
  background (downscaled and palette-reduced if made with an image generator), tagged
  `synthetic`, at most ~1/3 of a category's labels, in their own shard so they can be
  dropped. Worth targeting at weapon, structure and nature. Not yet done; needs an ingest
  step for files the user drops in a folder.

**Pilot training (2026-10-18, `runs/pilot-p4`, patch 4, 24.2M, 383k sprites, batch 256).**
- Ran 3,000 of 4,000 steps (~770k samples) at 174 samples/s when quiet, 43 when the
  machine was busy, then **stopped by me because it was using too much CPU for the user to
  work.** It resumes from `ckpt.pt` at step 3000 (`train_dibs.ps1 -Out ... --patch 4
  --steps 4000 --ckpt-every 1000`).
- Grids at 1k/2k/3k: loss 1.63 → 0.10. Samples are still noise-like blobs, but with class
  colour statistics (terrain green/blue, effects bright). No sprite shapes yet, so it
  says nothing about data scaling. Expected this early, and EMA 0.999 lags.
- **A 3M-sample scaling run (11.7k steps) may also be under-trained.** Decide the step
  budget after seeing grids at ~10k steps.
- **Finished 4,000 steps (2026-10-18 18:23), loss 0.092, ~105 samples/s for the last
  1,000 steps (45 min).** `grid-0004000.png`: figures with heads and bodies in the
  creature row, rounded outlined objects in the food row, clearly green terrain, tan
  structures. Shapes are emerging but nothing is clean yet. Expect many more steps.
  Extending to 20k steps is ~11 h of GPU in slices at this rate (resume with
  `--steps 20000`).
- **The user's machine was unusable while training ran** (CPU 100% with other sessions'
  jobs). The user asked only for a preemptible `dibs` lock, so other sessions can butt in
  (`train_dibs.ps1` yields on request). An idle-time gate was tried on my own initiative
  and reverted at the user's request; do not add input-based gating.

**RunPod scaling test (2026-10-19; user approved sending the data off the machine).**
- Pod: RTX 4090, community cloud, $0.34/hr, created with `--terminate-after` 8 h and
  deleted when done. Needs `--ports 22/tcp` (and `--public-ip` on community) for SSH.
  Setup: `pip install --break-system-packages numpy pillow` on the PyTorch 2.9 image.
  Total ~80 min, well under $1. The other two pods on the account are other sessions'.
- Speed: **~4,000 samples/s** (the laptop managed 43–174), so 20k steps (5.1M samples)
  takes ~21 min. `utilities/sprite-train/pod_run.sh` queues 100%, 50%, 25% of the data
  at patch 4, 20k steps each, batch 256, grid every 2,000 steps.
- Final loss 0.076 (100%). Results in `C:\dev\sprite-data\runs\scale-p4-f*` (grids,
  `samples-final.npz`, logs; the 100% `ckpt.pt` is kept for the NPU port).
- **The model draws real sprites by 20k steps** (grids at 4k were still blobs):
  creatures with hair and clothes, swords and axes, jars and cakes, flames, doors and
  furniture, bushes. By eye the three runs look close; 25% already works, and more data
  mostly adds variety (e.g. the 25% food row is nearly all potion bottles; the 100% row
  has jars, cakes and bread).
- Scoring (`eval_scaling.py`, so400m recognition, nearest-neighbour copying, diversity vs
  real sprites; 256 EMA samples per category, CFG 3, 16 steps) was paused for another
  session's NPU benchmarks and rerun after they finished. Full table:
  `runs/report-scaling.md`.

| run | creature | item | weapon | food | effect | terrain | structure | nature | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real sprites | 95% | 92% | 94% | 88% | 86% | 77% | 70% | 77% | 85% |
| 25% data | 95% | 68% | 92% | 91% | 91% | 85% | 48% | 86% | 82% |
| 50% data | 97% | 74% | 85% | 91% | 85% | 88% | 80% | 71% | 84% |
| 100% data | 98% | 46% | 84% | 66% | 84% | 84% | 79% | 79% | 78% |

  - **Recognition is close to real sprites** (generated 78–84% vs real 85% mean) at all
    three sizes.
  - **Copying: 0% in every run and category.** The nearest training sprite is never
    within 0.02 rms (medians 0.25–0.57), so nothing was memorised.
  - **Diversity is at or above real sprites.** Item (1.12 vs 0.67 real) and effect
    (0.87 vs 0.64) at 100% are far above real, which looks like noise rather than
    variety.
  - **Structure is the one category that clearly improved with more data** (48% at 25%
    → 80% at 50% and 79% at 100%).
  - **The test is confounded: all runs had the same 20k steps.** That is ~13 epochs of
    the full data but ~53 of the 25% subset. The full run drops on item (46%) and food
    (66%), the categories with the most distinct designs. Likely under-trained, not
    worse for having more data. 256 samples per category gives ±3% noise, one seed.
  - **Next:** train the full data for 4× the steps (80k; ~85 min, <$1 on the pod), and
    then the final patch-2 (256-token) model. A fair data comparison needs equal epochs.

**100k-step runs, and captions (2026-10-19).** The user said the 20k-step model was
"terrible, only a few images coherent". Findings:
- **Sampling settings are not the problem:** 50 Euler steps vs 16, CFG 1.5/3/5, EMA vs raw
  barely change the samples (`runs/sampling-check/`).
- **Colour counts:** real sprites have a median 9 colours (7 after discarding the low 3
  bits); generated samples 308 exact, **60 at 16 levels per channel vs 7 for real**. So the
  model gives smooth shading instead of flat palette colours. The raw 308 is inflated by
  invisible ±1-level noise, which I first overstated. Snapping to 12 colours
  (`snap.py`) flattens it a little but does not fix shapes.
- **Captions** (`captions.py`, saved as `caption` in `trainset.npz`): real text per sprite
  = file/folder name + pack tags (OGA/itch) or sheet title + platform + role (TSR) or
  group name (packs) + pixel-derived colour words. Noisy but real; 6.9 words on average.
  57k sprites come from single-sprite files, ~46k with a readable file-name subject.
- **Two 100,000-step runs on the full data** (RunPod secure RTX 4090, $0.89/hr, ~4.9k
  samples/s, 88 min each, ~$2.70; pod deleted):

| run | creature | item | weapon | food | effect | terrain | structure | nature | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real sprites | 95% | 92% | 94% | 88% | 86% | 77% | 70% | 77% | 85% |
| 20k steps, category-only | 98% | 46% | 84% | 66% | 84% | 84% | 79% | 79% | 78% |
| **100k, category-only** | 98% | 86% | 96% | 93% | 89% | 91% | 83% | 94% | **91%** |
| 100k, + caption words | 80% | 67% | 84% | 70% | 83% | 73% | 60% | 69% | 73% |

  - **Training length was a big part of the problem.** 100k category-only samples are
    recognised more often than real sprites (91% vs 85%) and look clearly coherent.
    No copying (0% in both runs).
  - **The caption run scores lower on category recognition** but its prompts are real,
    noisy captions, so it follows the words, not the category. Its value is control
    (`sample_text.py`, `runs/prompt-test.png`):
    - **colour words work well** (red apple, red potion, gold key, green zombie, gray
      knight, orange fire, green grass);
    - **some subjects work** (axe, apple/tomato, key, zombie, bow-ish branches);
    - **others don't** (sword is tiny junk, book is red blobs, bread and cake are pale
      shapes, dragon is brick texture).
    Subject knowledge is limited because few sprites have a clean subject word and the
    word embeddings are learned from scratch.
  - Without a size tag many text samples come out tiny (the size tag is available but
    unused in these tests).
- Files: `runs/q100k-cat`, `runs/q100k-text` (grids every 5k, `ckpt.pt`, `samples-final.npz`),
  `runs/report-q100k-X.md`.
- **Open:** a final model. Candidates: more steps (250k+), patch 2 or a wider model
  (h512×12), cleaner subject captions, size tag in training prompts. Category-only is the
  quality model; text-conditioned is the control model.
- **Quality run started (2026-10-19 19:40 UTC, user: "quality first"):** patch 2 (256 tokens),
  category-only, 100k steps, full data (`pod_run3.sh`, `runs/q-p2-cat`). RunPod A100 SXM secure
  ($1.79/hr; no 4090 in stock), 1,350 samples/s, ~5.3 h, ~$9.5. Same samples seen as `q100k-cat`,
  so it compares directly. If it wins, extend from `ckpt.pt` (constant lr, so resume is clean).
- **Result (2026-10-19 20:00 local, ~$9.5, pod deleted):** recognition 92% vs 91% for patch 4
  (creature 100, item 90, weapon 96, food 95, effect 93, terrain 93, structure 78, nature 93);
  within noise at 256 samples per category, and above real sprites (85%), so this metric is
  saturated. Copying 0%. By eye on the same seed (`runs/show50-p2.png` vs `show50-100k.png`):
  crisper outlines, real terrain tiles, readable chests and cupboards; new flaw: stray single
  bright pixels on foliage edges. Colour noise unchanged (median exact colours 73–432 vs ~9 real).
  Remaining gap is colour noise, not shape.
- **Arbiter ruling (model-experiments-b5, 2026-10-19): B first** (measure the NPU), A only if
  patch 2 fits an NPU target and loss still falls; C rejected; tiny effects parked.
  - `snap2` (`snap.py`): per-sprite palette from opaque pixels; median colours 234 -> 10 (real ~9).
  - Stray-pixel cleanup (`stray_test.py`, one attempt): counted strays 8.5 -> 0.9 per sprite,
    but real sprites score 17.4 (eyes, highlights), and the visible foliage strays sit on the
    silhouette edge and stay. Not adopted.
  - Loss: training loss still falls ~0.0005 per 10k steps at 100k (0.0646); no validation split.
  - **NPU, GEMMs only** (`npu_gemm_bench.ps1`, dit_gemm, 64-sprite sheet + CFG = 128 sprites,
    hidden 384 padded to K 512 / N 1024): per block, patch 2 = 31-33 + 10-11 ms, patch 4 =
    6-8 + 2.5-3.5 ms. Per sampling step (10 blocks): **patch 2 ~420 ms, patch 4 ~100 ms**.
    GEMM-only sheet time at 16 steps: patch 2 ~6.8 s, patch 4 ~1.6 s (8 steps: 3.4 / 0.8 s).
    Attention (~10% of flops at patch 2, needs the batched dit_fa) and elementwise extra.
    Padding wastes 1.6x (lin1) and 2.7x (lin2) at hidden 384. The earlier 0.5-1 s/sheet
    estimate was optimistic. No NPU target in the plan: asked the owner.

**Harvest results, PARTIAL (2026-10-17 02:30; cut 61/64 archives, labelled 16/64).**
Superseded by the full numbers above.
- Cut so far: 358,532 unique sprites ≤32×32 after the per-pack cap.
- Labelled so far: 79,479 sprites from 1,885 packs. 21k of them (26%) carry a label;
  the agreement rule is strict.
- Distinct designs (`oga_designs.py`, merged within each pack by so400m cosine):
  - palette swaps alone: 92% of sprites stay distinct;
  - cosine ≥0.97: 58%; ≥0.95: 42%; ≥0.93: 31%; ≥0.90: 20%.
  - Threshold set by eye on `oga/check-design-pairs.png`:
    - ≥0.95 merges animation frames and colour variants of one design (correct);
    - 0.93–0.95 is mostly the same plus a few genuinely different shapes;
    - <0.90 pairs are different designs.
  - **Distinct-design threshold: 0.95** (reversible).
- Extrapolated to all ~360k sprites: ~150k distinct designs (≥0.95), ~110k at 0.93.
  **The Phase 1 targets of ≥200k sprites and ≥50k designs hold with room to spare.**
- Labelled per category so far, as sprites / designs at 0.95:

| category | sprites | designs |
|---|---:|---:|
| creature | 7,981 | 2,748 |
| effect | 4,113 | 1,363 |
| weapon | 3,075 | 735 |
| item | 2,039 | 1,103 |
| terrain | 1,668 | 677 |
| food | 800 | 569 |
| structure | 658 | 246 |
| nature | 655 | 243 |

  - The 16 archives are ~22% of the sprites, so ×4.5 for a rough full-set figure.
  - Food (~3.6k labelled), structure and nature (~3k each) look short of the 5k
    labelled target. Every other category clears it.
  - Pack sprites (bake-off corpus, 9.6k with true labels) are on top of this.
- **Text threshold: drop at text margin > 0.015** (4.6% of sprites), set by eye on
  `oga/check-text-margin.png`:
  - above it: letters, digits and UI glyphs, with the odd real sprite;
  - 0–0.015: mixed, mostly UI shapes.
  - Applied at training-set build time, so it stays reversible.
- The labeller runs at below-normal priority until it is done (~16 h at 0.2 s/sprite).

**Source 1: TSR crawl for volume (`crawl_tsr.py`, existing; blocked, see above).**
- Raise the quota from 100 games to ~1,000, across SNES, NES, GBA, Genesis, GB/GBC,
  Master System and Game Gear. Also raise `--per-sheet` (30 now).
- Yield so far: 100 games → 22.7k sprites before cleaning, so ~1,000 games gives ~200k.
- Time: ~33 requests per game at 0.8 s ≈ 30 s/game, so ~8–9 h for 1,000 games.
  Unmeasured at this scale; the crawl log gives the real rate after the first hour.
- Resumable: every response is cached, and `sheets.jsonl` / `sprites2.jsonl` append.
  Stop the process at any time; re-running the same command continues.
- Then `selftrain_extract.py` → `selftrain_clean.py` (text and box removal).
- Labels: only where the seed and so400m zero-shot agree (1d); everything else trains
  unlabelled.

**Source 2: single-subject asset packs for labels (`harvest_packs.py`, new).**
- Sources:
  - itch.io free asset packs, found by tags: pixel-art + food / weapons / items / icons
    / effects / spells. `itch_fetch.py` already downloads free packs without a key.
  - OpenGameArt collections and tags, CC0 / CC-BY / OGA-BY only.
  - Kenney (CC0).
- **Label rule:** a pack is labelled by its title and page tags only when they name one
  category ("16×16 food icons", "RPG weapon pack", "spell effects"). Mixed packs label
  per file, by keywords in the file or folder name (`apple.png`, `sword_03.png`).
  Otherwise the sprite is kept unlabelled.
- Cutting: grid detection first (most packs are 16 or 32 px grids), components second
  (the 1b cutter).
- Rescaling: packs at 2×/3×/4× get reduced by the run-length GCD (1b step 2). Native
  sprites over 32×32 are dropped.
- Record per pack: URL, licence, title, tags, and the category rule that fired.
- Polite rate (1 request/s), cached, resumable like the TSR crawl. Downloads are kept;
  nothing is deleted.
- Check: on 20 packs, contact sheets per assigned category. This is a one-time check of
  the rule, not a review step in the pipeline.
- Spend: stop harvesting a category once it reaches the target, or when 50 new packs add
  <5% more distinct designs.

**Source 3: label more of what we have.**
- so400m zero-shot over all TSR sprites (the 1d "next measurement"). It is resumable,
  and it is the production labelling pass.
- Pack sprites already labelled `item` that so400m calls food or weapon at high margin
  stay `item`. Pack keys win.

**Augmentation is not variety.** Flips and palette swaps help training but never count
towards the targets.

### Phase 1 gate
Report counts by category × era × footprint, as sprites and distinct designs.
- Targets: total ≥200k sprites / ≥50k designs; ≥5k labelled / ≥1.5k designs per
  category.
- A category short of its target can still train. It goes into Phase 2 flagged, and the
  scaling test decides whether chasing it is worth it.

---

## Phase 2: train (iGPU)

- **Model.** Rectified flow on 32×32 RGBA, values in [-1, 1].
  - The DiT copies klein's single-stream block, so Phase 3 reuses `dit_gemm` and `dit_ew`
    by shape: adaLN modulation, fused QKV + SwiGLU in, QK RMSNorm + 2D RoPE, head dim 128.
  - Start at h384 × 10 (19M) with patch 2 (256 tokens).
  - Ablations: patch 4 (64 tokens, ~4× cheaper) and h256.
  - Re-run `bench.py` at 256 tokens first to size the training run.
- **Conditioning** (per decision 1):
  - The tag vector goes through an MLP: category, facing, pose, era, footprint, colours.
  - The caption goes through embeddinggemma-300m, frozen; the server already runs this
    model for `/v1/embeddings`. Its pooled vector goes through a second MLP.
  - Both are added to the timestep embedding.
  - Each is dropped independently 10% of the time, for CFG.
  - A pooled vector binds attributes weakly ("red sword, blue shield"). Per-token
    cross-attention is the upgrade if that matters.
- **NPU numerics in training.** Use bfp16ebs8 fake-quant on the GEMM weights, and the
  hardware's exp2 approximation in softmax (`utilities/dit-ref/fa_emul.py` models it
  bit-exactly). The trained model then matches what the NPU computes. Cheap to add; drop
  it if it slows training more than ~20%.
- **Augmentation.** Horizontal flip (swapping facing left/right), plus the capped palette
  swaps from 1b.
- **Evaluation:**
  - sample grids per category and footprint at 8, 16 and 32 Euler steps;
  - a nearest-neighbour search against the training set, to measure how much the model
    copies;
  - a fixed-seed grid saved per checkpoint.
- **Scaling test (before any harvest beyond the Phase 1 targets):**
  - Train the same model on 25%, 50% and 100% of the data, each split by distinct design
    so the subsets share no designs. Same number of training steps for all three.
  - Measure per category on 512 samples each:
    - **recognised:** so400m zero-shot agrees with the requested category;
    - **copying:** the share of samples whose nearest training sprite (SigLIP cosine)
      is above a near-duplicate threshold, plus the exact pixel-match rate;
    - **diversity:** mean pairwise SigLIP distance among samples, against the same
      measure on the training sprites.
  - Decision per category:
    - still improving from 50% to 100% (recognised up, copying down): harvest more for
      it;
    - flat: stop, more data won't pay.
  - Cost: 30k sprites × 100 passes ≈ 2.5 h at 64 tokens. At 256 tokens it is ~4× that;
    re-measure with `bench.py`.
- **Checkpoint.** The "good enough" call is the user's eye on the grids.
- Tools go in `utilities/sprite-train/`.

**Status (2026-10-17): code written, smoke-tested on CPU, waiting for the iGPU.**
- `build_trainset.py` builds `C:\dev\sprite-data\trainset.npz` from:
  - the bake-off packs (true labels);
  - labelled OGA shards, dropping text margin > 0.015 and `ui` packs;
  - TSR labels once `label_tsr.py` has run.
  - Partial build: 86,079 sprites, 31k labelled (creature 11.6k, effect 4.4k,
    terrain 3.7k, weapon 3.3k, item 3.2k, structure 2.5k, nature 1.3k, food 1.1k).
- `train.py`:
  - model: the klein-style block as specified above;
  - conditioning: category + footprint embeddings added to the timestep embedding, each
    dropped 10% of the time;
  - training: logit-normal t, flip augmentation, EMA 0.999;
  - checkpoints: `ckpt.pt` every 2,000 steps (resumes on restart), plus a fixed-seed grid
    (one row per category + an unconditional row, 16 Euler steps, CFG 3);
  - `--frac`: nested subsets by group for the scaling test.
- CPU smoke test: 0.6M model, 300 steps; loss falls, it resumes, grids are written.
- Choices made without the user (reversible):
  - **Decision 1 taken as recommended: tags only** (category + footprint). No captions:
    there is no caption source.
  - **NPU fake-quant deferred** until there is a model worth porting.
  - **Groups** (pack, game, or pack-group) stand in for design splits in the scaling test.
    Packs rarely share designs.
- Order once the iGPU is free:
  1. finish OGA labelling (~16 h, running);
  2. `label_tsr.py`. The scheduled Oct 8 run failed early (exit 1, no logs) and never
     labelled anything; run it directly instead of debugging the wrapper.
  3. rebuild the training set;
  4. `bench.py` at 256 tokens to size runs;
  5. the scaling test (25/50/100%).
  - Not benchmarking now: the labeller holds the iGPU, so timings would be wrong.

**Bench (2026-10-17 ~03:00, `bench_train.ps1`).** Run with the labeller paused. Other
sessions' CPU/NPU jobs were running, and the cutter for its first ~5 minutes.

| config | params | tokens | samples/s |
|---|---:|---:|---:|
| h384 × 10, patch 2 | 24.2M | 256 | **19** |
| h384 × 10, patch 4 | 24.2M | 64 | **81** |

- **`train.py` is ~4× slower than `bench.py`'s bare blocks** (325 samples/s for 19M at
  64 tokens). Suspects, unprofiled:
  - the per-parameter EMA loop and non-fused AdamW;
  - fp32 RoPE and RMSNorm;
  - the host→GPU copy of every batch;
  - CPU contention from other sessions, since ROCm on Windows issues every kernel from
    the CPU.
- **Scaling-test estimate at today's speed:** 3 runs × 3M samples each (the plan's
  "30k × 100 passes" budget, kept fixed across the 25/50/100% runs):
  - patch 2: 44 h per run, **~132 h total**;
  - patch 4: 10 h per run, **~31 h total**.
- Neither is acceptable as is. **Next: profile and speed up `train.py` before any
  training**:
  - fused AdamW, `torch._foreach` EMA, data kept on the GPU as uint8;
  - bf16 RoPE/norms; check SDPA vs eager on ROCm (eager won for SigLIP).
  - Target: within 1.5× of `bench.py`, i.e. ≥200 samples/s at 64 tokens and ≥50 at 256.
  - Then: patch 4 for the scaling test (relative comparisons don't need full
    resolution), patch 2 for the final model.

**Speed-up work (2026-10-17 04:00–04:55, `prof_train.py`, patch 4 / 64 tokens, labeller
paused).**

| window | ceiling (bench.py bare blocks) | train.py settings | ratio |
|---|---:|---:|---:|
| 1 (04:10) | 248 | 131 baseline; 106–120 with host-side changes | 0.53 |
| 2, start (04:30) | 123 | 81 | 0.66 |
| 2, end (04:50) | 77 | 54 | 0.70 |

- **The machine's throughput drifts 3× within an hour.** The ceiling fell from 248 to
  77, while other sessions ran CPU/NPU jobs (bench2.py, kokoro, qwen3-embed); thermals
  are possible too.
  - Absolute samples/s are therefore not comparable across windows.
  - Only rows measured back-to-back are, and even within window 2 the drift was 1.5×.
- Findings:
  - Host-side changes (no per-step sync, data on GPU, foreach EMA, fused AdamW): no
    measurable gain. The step is GPU-bound.
  - Switching off RoPE + QK norm (timing only): +16% (81 → 94). Those extras are not
    the gap.
  - **Ours is 0.53–0.70× the bare blocks**, which is within ~1.5–1.9×. The remaining cost
    is the work the model genuinely does: adaLN modulation, QK norm, RoPE, patching.
  - Rotate-half RoPE, joint q/k and addcmul are numerically identical paths. They read
    slower, but that sits inside the drift. Not adopted; defaults unchanged.
- **Verdict: stop optimising.** We're close to the ceiling, which is the arbiter's stop
  rule. The 200 samples/s target is above the ceiling itself on this machine under
  current load.
- **What actually decides training time is machine load.** At the ratio above:
  - a quiet machine (ceiling ~325 at 64 tokens) should give ~170–220 samples/s, so
    ~4–5 h per 3M-sample run, **~13–15 h** for the patch-4 scaling test;
  - a busy one gives 3×+ longer.
  - Training should run when other sessions aren't using the iGPU, NPU and memory
    bandwidth (overnight, coordinated through the arbiter). Re-measure the ceiling at
    the start of the run.

**Categories under the 5k-labelled target (decision for the user).** Projected from
16 of 64 archives, plus the bake-off packs:
- **food ~3.6k labelled (~2.6k designs): the weakest;**
- **structure ~3k;**
- **nature ~3k.**
- Options:
  - (a) accept: classifier-free guidance still conditions on them, just with less
    variety;
  - (b) add sources: food/nature-tagged itch.io packs, or relax the label rule for
    those three to "pack keyword alone";
  - (c) wait for the scaling test to show whether they improve with more data.
- Recommend (c): it costs nothing, and the zips are kept, so (b) can follow.

## Phase 3: NPU port as a sheet generator

**One sprite gains nothing on the NPU.**
- One sprite is ~80 GFLOP over 8 steps (19M params, 256 tokens).
- On the NPU that is ~90 dispatches per step with half the array idle: ~150 ms, about
  the same as the CPU.
- **A 64-sprite sheet** gives 64 × 256 = 16,384 rows per GEMM and fills all 32 cores.
  Estimate: 0.5–1 s per sheet at 8 steps, less with patch 4 or a step-distilled model.
  To be measured.

Kernel work:
- **`dit_gemm`.** New instruction streams for the small-K shapes (K = 384, 1152). Check
  the K and N alignment constraints. It should need no new xclbin, because loop bounds are
  runtime parameters.
- **`dit_fa`: batched short sequences.** Each sprite's 256 tokens may attend only to
  themselves (block-diagonal). This is the one real kernel change.
  - One sprite's K/V per head is 128 KB, which does not fit one core.
  - The batched mode therefore streams K/V in chunks, as `dit_fa` already does, and resets
    its running max and sum at each sprite boundary.
- **`dit_ew`: modulation.** klein broadcasts one modulation vector over all rows.
  - **v1: one label per sheet** (e.g. 64 monsters). The modulation is then identical
    across the batch, and klein's kernel works unchanged.
  - Per-sprite labels later: a per-row-group modulation stride.
- **RoPE.** Repeat the 16×16 position table across the batch.
- **Caption embedding.** Runs once per sheet on the NPU through the existing embedding
  path; nothing on the host.
- **Host.** A schedule module like `klein_pipeline.py`, run as one hardware context with
  runlists (`compose_elf.py`).
  - Output is a PNG sheet.
  - Alpha threshold and an optional palette snap run as an NPU elementwise op, not on the
    host.
- **Spec.** Create `specs/sprite-diffusion/` (prefix `SPRITE`) at this phase, once there
  is engine behaviour to specify.

### Phase 3 implementation plan (2026-10-19, arbiter ruling: build now, patch as a parameter)

**Goal.** A 64-sprite sheet (one category per sheet, CFG 3) generated end to end on the NPU from
`runs/q-p2-cat/ckpt.pt`, then the CPU palette snap (`snap2`) and a PNG. Patch size (2 or 4) and
step count (8 or 16) are parameters. Output to the owner: full sheet time at 8/16 steps for
patch 2 and patch 4, plus one sample sheet each.

**Spec impact.** New spec `specs/sprite-diffusion/spec.md` (prefix `SPRITE`), created in step 1:
- `SPRITE-SHEET`: a sheet command writes a PNG of 64 RGBA 32×32 sprites of one category
  (manual: open the PNG, check 64 sprites of the asked category).
- `SPRITE-NPU-PARITY`: the NPU velocity for one step matches the PyTorch reference within the
  dit chain gates (rel_fro ≤ 3e-2, row cosine > 0.999) (test, run on hardware: deterministic
  given fixed inputs, silent when wrong, cheap with saved goldens).
- `SPRITE-PALETTE`: snapped sprites have at most 16 colours (test: pure function).

**What already fits (measured or read in the code):**
- `dit_gemm`: all block GEMMs, padded (hidden 384 → K 512 / N 1024). Measured today.
- `dit_fa`: head dim 128 matches the model. Constraints: tokens % 512, heads % 2, keys % 64.
- `dit_ew`: `ln_mod`, `res_ln_mod`, `qk` (RMSNorm + RoPE), `euler`. The row element size is a
  build define (`EW_EL`, 3072 for klein).

**Kernel work (the real cost):**
1. **`dit_fa` block-diagonal mode.** One pass is 512 query rows = 2 sprites at patch 2, 8 at
   patch 4. New runtime parameter `seq`: each core walks only the key chunks of its own
   sprite (`[s·seq, (s+1)·seq)`), and K/V stream per pass, not per whole sequence. No
   topology change; the cores skip chunks. Heads: 3 → pad to 4 with a zero head (heads % 2),
   or run 2 + 1 with a zero partner. Pick by test.
2. **`dit_ew` at row size 384.** A second build with `EW_EL=384` (its own xclbin, its own
   streams). Check the core-count split (`N_COLS * PER_COL` elements per stream) and the
   parameter-vector fill at the small size.
3. **Sprite RoPE mode in `qk`.** The model's RoPE is 2-D: 32 frequency pairs from the row,
   32 from the column, interleaved, θ = 10000 (`dit.py:rope_table`). klein's mode has 4 axes
   of 16 pairs, so it cannot be reused. Add `rope: "sprite"` with an 8 or 16-entry fp32
   table per axis (positions 0..15 at patch 2, 0..7 at patch 4).
4. **`cfg_euler` op in `dit_ew`.** `x + dt·(vu + cfg·(vc − vu))` from two velocity views
   and the latent. Replaces the host CFG combine.

**No host compute in the loop** ([[no-cpu-compute-in-npu-pipelines]]):
- The conditioning vector depends only on (t, category, footprint). With a fixed step
  schedule, every block's modulation (shift, scale, gate) and the final modulation are
  precomputed at export time per (step count, category) and stored with the weights:
  9 categories × (8 + 16) steps × 11 modulations × 1152 values, ~5 MB bf16. No timestep
  MLP runs at sampling time.
- Patchify/unpatchify are fixed permutations: fold them into the `inp` and `outp` weight
  layouts (K 16/64 → 512, N 384 → 1024 padding) plus a strided view, so the latent stays
  in a GEMM-ready layout for the whole sampling loop.
- The host writes the initial noise once and reads the final latent once.

**Host program.** `open_kernels/sprite_pipeline.py`, on the model of `klein_pipeline.py`:
one hardware context, the three sets composed into one ELF (`compose_elf.py`), the whole
sampling loop as a runlist. Weights from a new exporter `utilities/sprite-train/export_npu.py`
(bfp16 packing with `pack.pack_b`, padding, modulation tables, a manifest).

**Steps, each with a gate before the next:**
1. Spec skeleton; PyTorch reference that dumps one step's per-op goldens (fixed noise, fixed
   category) at patch 2 and patch 4.
2. `dit_ew` EL=384 build + sprite RoPE + `cfg_euler`; each op passes `compare.py` at the
   bf16 floor against the goldens.
3. `dit_fa` block-diagonal mode; passes against PyTorch SDPA per sprite (rel_fro ≤ 3e-2).
4. Exporter + one full block on the NPU against the golden block output.
5. Whole step, then the whole sampling loop; `SPRITE-NPU-PARITY` on step 1, and a sheet
   that looks like the PyTorch sheet for the same noise.
6. Timing under dibs (`npu`), alternating patch 2 / patch 4, at 8 and 16 steps; four sample
   sheets; report to the owner.

**Arbiter review (model-experiments-b5, 2026-10-19): approved with two cuts and a checkpoint.**
- Cut 1: `cfg_euler` deferred. CFG and Euler run on the host for now (a few ms per step
  against ~400 ms of GEMMs). This is a known exception to [[no-cpu-compute-in-npu-pipelines]].
- Cut 2: no composed ELF / runlist yet. One dispatch chain per step; measure the dispatch
  overhead; build the runlist only if it is a real share after the owner sets a target.
- Gate order: goldens → ew ops (EL 384) → sprite RoPE → fa block-diagonal → one block →
  whole loop → timing (patch 2/4 × 8/16 steps) + sheets → report.
- Checkpoint: after "one block", send the arbiter time spent and the estimate for the rest;
  if the rest is > 2 days, stop and report to the owner.
- `SPRITE-NPU-PARITY` must skip on a machine with no NPU.

**Phase 3 result (2026-10-20 03:00, branch `feat/sprite-diffusion` @188370c2).** All gates passed.
- NPU vs fp32 reference, one full evaluation: patch 2 velocity rel 1.6e-2, row cos 0.9994
  (PASS); patch 4 rel 2.0e-2, row cos 0.9989 (just under the 0.999 row gate; bf16 drift; the
  numpy model of the same dataflow matches to 4e-7).
- Full 64-sprite sheet (food, seed 1), two rounds, host CFG/Euler, one dispatch per op,
  three hardware contexts:

| | 8 steps | 16 steps |
|---|---:|---:|
| patch 2 | 5.0 s | 9.9 s |
| patch 4 | 2.2 s | 4.6 s |

  - Per step ~620 ms (patch 2) vs the GEMM-only estimate of ~420 ms; small elementwise ops
    (qk ~10 ms a call) and context switches make up the rest.
  - Sheets: `C:\dev\sprite-npu
pu-sheets-food.png`. 8 steps look as good as 16. Patch 2 is
    crisper; patch 4 softer. All four read as food.
- Next (owner decides from these numbers): a target. Cheap speed-ups if needed: 8 steps
  (2x, no visible loss), one composed ELF (removes ~60 context switches a step), the
  dit_gemm SwiGLU epilogue and qk fusion, hidden 512 on any retrain.
- `model-experiments-b5` (arbiter) was unreachable at the checkpoint (2026-10-20 ~03:00).
- **Owner target set (2026-10-20): patch 2 at 8 steps, ~5 s per 64-sprite sheet, is "GREAT".**
  No speed work is needed; the composed ELF and fusions stay optional.

**Risks.**
- `dit_fa`'s L1 budget is 59 KB of 64; the per-core key range adds RTP fields only, but
  watch the program memory (16 KB) of `dit_ew` when the sprite RoPE helpers are added
  (the qwen helpers had to be `noinline` for this reason).
- Small streams are dispatch-bound: at patch 4 one step is ~100 ms of GEMM over ~60
  dispatches; dispatch overhead may dominate the elementwise ops.
- The bfp16 GEMM and hardware exp2 drift (klein's LPIPS study) may show at 32×32, where
  one pixel matters. Step 5 compares sheets by eye and by colour count.

**Not in scope** (arbiter): a new GEMM tile for the padding waste (only if the full time
misses the owner's target); the stray-pixel fix; more training. Next retrain, if any:
hidden 512 and a held-out validation split.

## HANDOFF (2026-10-20): move to model-experiments, not started

Owner decision: this work leaves OpenFlowLM. It moves to `C:\code\model-experiments`, where each
experiment lives on its own branch. Model support in oflm comes only after a working model.
- Kernels: the modified `dit_fa` (block-diagonal seq mode) and `dit_ew` (EL 384, sprite RoPE,
  residual bias) go into model-experiments too, not an OpenFlowLM branch.
- The repo is PUBLIC: create a local branch only (`git switch -c sprite-diffusion --no-track
  origin/main`, in a separate worktree because `main` there has other sessions' uncommitted
  edits). Do not push until the owner decides what may be public (harvesters, ripped-data notes).
- Source of truth now: worktree `C:\code\OpenFlowLM-sprite`, branch `feat/sprite-diffusion`
  @188370c2 (local only; includes a local merge of origin/main). Move: `utilities/sprite-data/`,
  `utilities/sprite-train/`, `specs/sprite-diffusion/`, the two kernel designs, this plan.
  The runner still needs an OpenFlowLM checkout for dit_gemm, the harness and build_design.py:
  point to it with an env var instead of `parents[2]`.
- Data and outputs stay in `C:\dev\sprite-data` and `C:\dev\sprite-npu` (not in git).
- Paused before the move because the owner had to close the laptop (2026-10-20).

## Phase 4 (optional): surface
`oflm image --model sprites` or a dedicated subcommand, and packaging. The model stays
local-only while it is trained on ripped data.

---

## Decisions for the user

1. **Conditioning: tags only (recommended now), or tags + captions?**
   - The bake-off found no local model that writes trustworthy captions: subject top-1
     ≤17%. So the caption path has no data source.
   - Recommend tags: category, facing, era, footprint, colours, plus an optional name
     embedded with embeddinggemma.
   - Revisit if a captioner appears.
2. **NPU v1: one label per sheet, or per sprite?**
   - Per-sheet needs no `dit_ew` change.
   - Recommend per-sheet first.
3. **Canvas: 32×32 RGBA plus a footprint tag.** Taken 2026-10-01.
4. **Labeller: SigLIP 2 + sheet metadata.** Taken by the bake-off, 2026-10-03.
   - The VLMs (qwen3vl-it:4b, gemma3:4b) and Florence-2 are out: slower and less accurate
     on every field.
   - The Florence fine-tune contingency is moot.
5. **Character + monster merged into `creature`.** Taken 2026-10-04 (user).
   - The sheet's role (hero / NPC / enemy / boss, from the TSR section) is kept as a
     separate `role` tag where the section gives one.
   - Merged accuracy, so400m zero-shot: 74% balanced, 85% on creature.
