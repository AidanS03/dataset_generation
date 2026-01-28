# Minimal Isaac Sim Replicator Dataset Generator

This folder contains a small, config-driven Isaac Sim Replicator pipeline that generates synthetic images plus 2D annotations suitable for training detection models.

- Generator script: `my_replicator/min_replicator.py`
- Default config: `my_replicator/config/min_config.yaml`
- Local target assets: `my_replicator/assets/`
- Local HDRI backgrounds: `my_replicator/backgrounds/`

It writes per-frame images + PoseWriter JSON, and can optionally post-process those into COCO (Detectron2-friendly) and/or YOLO label formats.

## Install Isaac Sim

Follow NVIDIA’s official installation guide (includes download links and setup steps):

- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/quick-install.html

This repo is laid out like an Isaac Sim distribution workspace (it includes `python.sh`, `apps/`, etc.). If you are using a separate Isaac Sim install, you can still copy the `my_replicator/` folder into your Isaac Sim directory (or adjust commands accordingly).

## Quick start (generate a small dataset)

From the repo root:

```bash
./python.sh my_replicator/min_replicator.py \
  --config my_replicator/config/min_config.yaml \
  --frames 200 \
  --output my_replicator/output_run1
```

Outputs are written to the `--output` directory. You should see:

- `000000.png`, `000001.png`, ...
- `000000.json`, `000001.json`, ... (PoseWriter DOPE-style JSON)
- `metadata.txt`
- `annotations_coco.json` (if COCO export is enabled)
- `yolo/` (if YOLO export is enabled)

## Where to put a new target object

Place your target object as a USD file under:

- `my_replicator/assets/<your_object>.usd` (or `.usdc` / `.usda`)

Then point the config at it via `OBJECT_USD`.

### Tips for preparing a good target USD

A “good” target asset for dataset generation typically has:

- Reasonable real-world scale (meters) and consistent unit usage
- A sensible pivot/origin (ideally at the object base or center)
- Clean geometry (no extreme outliers that break bounding boxes)
- Materials/textures embedded or resolvable on your machine

If the object origin is **not** at its bottom, you’ll usually want to set an offset height (see `CUP_HEIGHT_M`) so the object doesn’t intersect the ground.

## Config: setting a new target object

Edit `my_replicator/config/min_config.yaml`:

1) Set the target USD path:

```yaml
OBJECT_USD: "/absolute/path/to/isaac-sim/my_replicator/assets/my_object.usd"
```

2) Set the label name you want exported:

```yaml
CLASS_NAME: "my_object"
```

3) Adjust scale if needed:

```yaml
OBJECT_SCALE: [1.0, 1.0, 1.0]
```

4) Adjust spawn bounds so the object stays visible:

```yaml
OBJ_POS_MIN: [-0.2, -0.2, 0.0]
OBJ_POS_MAX: [ 0.2,  0.2, 0.2]
```

If you see many frames with empty annotations, the most common reason is “the object is out of view / too occluded / too small”. Tighten bounds and/or reduce distractors.

## Backgrounds (HDRI dome)

This pipeline supports dome-light HDRI backgrounds.

Background HDRIs used for this project were downloaded from Poly Haven (Indoor HDRIs):

- https://polyhaven.com/hdris/indoor

Download `.hdr` or `.exr` files and place them in:

- `my_replicator/backgrounds/`

Then configure:

```yaml
USE_DOME_LIGHT: true
DOME_TEXTURES_BASE: "/absolute/path/to/isaac-sim/my_replicator/backgrounds"
```

### Auto-discovery vs curated list

- If `DOME_TEXTURES` is **omitted or empty**, the script auto-discovers **all** `.hdr` / `.exr` files in `DOME_TEXTURES_BASE` and randomly samples from them each frame.
- If you want to pin to a curated subset, set:

```yaml
DOME_TEXTURES:
  - some_hdr_name_without_extension
  - another_hdr_name_without_extension
```

## Distractors (visual clutter, not annotated)

Distractors are extra objects added to create clutter/occlusions. They are not exported as labels.

### Recommended mode: Isaac asset path auto-discovery

To randomly sample from all YCB Axis-Aligned props shipped with Isaac assets:

```yaml
DISTRACTOR_COUNT: 50
DISTRACTOR_ASSET_PATH: /Isaac/Props/YCB/Axis_Aligned/
# Leave DISTRACTOR_FILENAMES unset/empty for auto-discovery
```

Notes:

- This requires access to Isaac Sim’s assets root (often via Omniverse content / Nucleus / the online content mirror).
- If you’re offline, prefer the local-folder mode below.

### Local folder mode (offline-friendly)

If you have a real filesystem directory containing USDs:

```yaml
DISTRACTOR_DIR: "/absolute/path/to/some/folder/of/usd_files"
```

### Distractor placement controls

These knobs directly affect clutter density and occlusion:

- `DISTRACTOR_POS_MIN` / `DISTRACTOR_POS_MAX` (meters)
- `DISTRACTOR_ROT_MIN` / `DISTRACTOR_ROT_MAX` (degrees)
- `DISTRACTOR_SCALE_MIN` / `DISTRACTOR_SCALE_MAX`

## Camera controls

The most important camera options are:

- `CAM_USE_LOOK_AT`: if `true`, the camera is oriented using a look-at target.
- `CAM_USE_OBJECT_POS_AS_LOOK_AT`: if `true`, the look-at target follows the sampled object position.
- `CAM_POS_MIN` / `CAM_POS_MAX`: camera position bounds (meters).
- `CAM_LOOK_AT_OFFSET_MIN` / `CAM_LOOK_AT_OFFSET_MAX`: jitter around the object for look-at targeting.

Practical guidance:

- For stable annotations, keep `CAM_USE_LOOK_AT: true` and `CAM_USE_OBJECT_POS_AS_LOOK_AT: true`.
- If you randomize camera pose too widely, you’ll get many frames with the target off-screen.

## Lighting controls

Two lighting systems can be used together:

- Dome HDRI lighting (`USE_DOME_LIGHT`, `DOME_LIGHT_INTENSITY`)
- Extra key/fill lights (scaled by `LIGHTS_SPHERE_INTENSITY_SCALE`)

If your scene looks blown out, try lowering `LIGHTS_SPHERE_INTENSITY_SCALE` first.

## Dataset export (COCO / YOLO)

The generator always writes PoseWriter per-frame JSON (DOPE-style cuboid keypoints). Optionally, it converts those outputs into standard detection dataset formats.

### COCO (Detectron2-friendly)

Enable COCO export:

```yaml
DATASET_EXPORT: coco
COCO_FILENAME: annotations_coco.json
```

Result:

- `<output>/annotations_coco.json`

This file contains `images`, `annotations` (bbox), and `categories`.

### YOLO

Enable YOLO export:

```yaml
DATASET_EXPORT: yolo
YOLO_DIRNAME: yolo
```

Result:

- `<output>/yolo/images/` (images)
- `<output>/yolo/labels/` (`.txt` labels)
- `<output>/yolo/data.yaml`

### Both

```yaml
DATASET_EXPORT: both
```

### Class filtering

By default, exports only the configured `CLASS_NAME`.

To export multiple classes (if your PoseWriter JSON contains multiple object classes), set:

```yaml
DATASET_INCLUDE_CLASSES: ["class_a", "class_b"]
```

### Flush/writer timing

Isaac Sim writers can be asynchronous. If your export runs before all JSONs are fully written, increase:

```yaml
DATASET_EXPORT_WAIT_S: 5.0
```

## Important config options (cheat sheet)

**Core / output**

- `CONFIG.headless`: run without GUI (useful for long dataset jobs)
- `WIDTH`, `HEIGHT`: dataset resolution
- `OBJECT_USD`: path to target asset
- `CLASS_NAME`: exported label name
- `--frames`: number of frames to write (CLI)

**Target randomization**

- `OBJ_POS_MIN`, `OBJ_POS_MAX`: target translation bounds (meters)
- `OBJ_ROT_MIN`, `OBJ_ROT_MAX`: target rotation bounds (degrees)
- `OBJECT_SCALE`: global scale multiplier for target USD
- `CUP_HEIGHT_M`: Z offset to keep object above ground (rename-friendly: it’s just an object height offset)

**Background randomization**

- `USE_DOME_LIGHT`: enable HDRI dome
- `DOME_TEXTURES_BASE`: folder containing `.hdr` / `.exr`
- `DOME_TEXTURES`: optional curated list (names without extension)

**Distractors**

- `DISTRACTOR_COUNT`: number of clutter objects
- `DISTRACTOR_ASSET_PATH`: Isaac virtual asset folder (auto-discover USDs)
- `DISTRACTOR_POS_*`, `DISTRACTOR_ROT_*`, `DISTRACTOR_SCALE_*`: clutter pose/scale

**Dataset export**

- `DATASET_EXPORT`: `coco | yolo | both | none`
- `COCO_FILENAME`
- `YOLO_DIRNAME`

## Troubleshooting

### “COCO/YOLO export is empty”

Check a frame JSON like `<output>/000000.json`. If you see:

```json
{"objects": []}
```

then PoseWriter did not record the target for that frame. Common causes:

- Target is off-screen: tighten `OBJ_POS_*` and/or `CAM_POS_*`
- Target too small: move the camera closer or increase `OBJECT_SCALE`
- Too much occlusion: reduce `DISTRACTOR_COUNT` or expand `DISTRACTOR_POS_*` so clutter isn’t always blocking

### Dome textures look blank / warnings about subframes

Replicator may warn that `RTSubframes` must be >= 3 when randomizing dome textures. The script/Replicator will often auto-adjust; if you still see blank backgrounds, try increasing subframes in your Isaac/Kit settings and/or reduce the frequency of dome changes.

### Performance

- Headless mode (`CONFIG.headless: true`) is usually faster for long runs.
- Larger `WIDTH`/`HEIGHT`, heavy distractors, and ray tracing settings can slow generation significantly.

## Reproducibility (optional)

If you need perfectly repeatable datasets, consider adding explicit random seeds (Python `random`, NumPy, and Replicator) and keeping a copy of the config with your outputs.

---

Questions or improvements you might want next:

- Add automatic train/val/test splitting for COCO/YOLO
- Export instance segmentation masks (if you decide to annotate distractors or add semantic segmentation)
