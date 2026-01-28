# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal Isaac Sim Replicator script that generates 5 images and corresponding JSON
in a local output folder. Keeps things simple: one cube, a camera, and basic lighting.
"""

import os
import argparse
import yaml
import numpy as np
import random
import json
from pathlib import Path

from isaacsim import SimulationApp


def main():
    parser = argparse.ArgumentParser("Minimal Replicator")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config/min_config.yaml"))
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "output"))
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    def _normalize_class_name(raw: str, mode: str) -> str:
        if raw is None:
            return ""
        raw = str(raw).strip()
        if mode == "full":
            return raw
        # default: first token (handles assets that encode multiple comma-separated labels)
        return raw.split(",", 1)[0].strip()

    def _bbox_from_projected_cuboid(points, width_px: int, height_px: int):
        """Compute [x, y, w, h] bbox from DOPE projected cuboid points."""
        if not points or not isinstance(points, list):
            return None
        xs = []
        ys = []
        for p in points:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                continue
            xs.append(float(p[0]))
            ys.append(float(p[1]))
        if not xs or not ys:
            return None
        x0 = max(0.0, min(xs))
        y0 = max(0.0, min(ys))
        x1 = min(float(width_px), max(xs))
        y1 = min(float(height_px), max(ys))
        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)
        if w <= 1e-6 or h <= 1e-6:
            return None
        return [x0, y0, w, h]

    def _export_coco_and_or_yolo(output_dir: str, width_px: int, height_px: int, expected_frames: int | None = None):
        """Convert PoseWriter DOPE JSONs into COCO and/or YOLO datasets."""
        export_mode = str(cfg.get("DATASET_EXPORT", "coco")).strip().lower()
        if export_mode in ("", "none", "off", "false"):
            return

        class_mode = str(cfg.get("DATASET_CLASS_MODE", "first_token")).strip().lower()
        if class_mode not in ("first_token", "full"):
            class_mode = "first_token"
        class_mode_internal = "full" if class_mode == "full" else "first_token"

        include_classes = cfg.get("DATASET_INCLUDE_CLASSES")
        exclude_classes = cfg.get("DATASET_EXCLUDE_CLASSES")
        include_set = None
        exclude_set = None
        if isinstance(include_classes, list) and include_classes:
            include_set = {str(c).strip() for c in include_classes}
        else:
            # Safer default: only export the configured primary class.
            primary = _normalize_class_name(str(cfg.get("CLASS_NAME", "")), class_mode_internal)
            if primary:
                include_set = {primary}
        if isinstance(exclude_classes, list) and exclude_classes:
            exclude_set = {str(c).strip() for c in exclude_classes}

        out_path = Path(output_dir)
        if not out_path.exists():
            return

        # PoseWriter writes files like 000000.png and 000000.json.
        # Writers can flush asynchronously, so wait briefly for the expected count.
        try:
            import time

            if isinstance(expected_frames, int) and expected_frames > 0:
                deadline = time.time() + float(cfg.get("DATASET_EXPORT_WAIT_S", 5.0))
                while time.time() < deadline:
                    image_files = [p for p in out_path.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
                    if len(image_files) >= expected_frames:
                        break
                    time.sleep(0.1)
        except Exception:
            pass

        image_files = sorted([p for p in out_path.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
        if not image_files:
            print(f"[min_replicator] DATASET_EXPORT requested but no .png found in {output_dir}")
            return

        # Build COCO structures.
        coco = {
            "info": {"description": "isaac-sim replicator export", "version": "1.0"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }
        category_name_to_id = {}
        next_category_id = 1
        next_annotation_id = 1

        def get_category_id(name: str) -> int:
            nonlocal next_category_id
            if name not in category_name_to_id:
                category_name_to_id[name] = next_category_id
                next_category_id += 1
            return category_name_to_id[name]

        # YOLO output settings
        yolo_dir_name = str(cfg.get("YOLO_DIRNAME", "yolo")).strip() or "yolo"
        yolo_root = out_path / yolo_dir_name
        yolo_images = yolo_root / "images"
        yolo_labels = yolo_root / "labels"
        yolo_use_symlinks = bool(cfg.get("YOLO_USE_SYMLINKS", True))

        do_yolo = export_mode in ("yolo", "both")
        do_coco = export_mode in ("coco", "detectron", "detectron2", "both")

        if do_yolo:
            yolo_images.mkdir(parents=True, exist_ok=True)
            yolo_labels.mkdir(parents=True, exist_ok=True)

        for img_id, img_path in enumerate(image_files, start=1):
            stem = img_path.stem
            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                # some pipelines may omit JSON; skip
                continue

            try:
                frame = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[min_replicator] WARNING: Failed reading {json_path}: {e}")
                continue

            coco["images"].append(
                {
                    "id": img_id,
                    "file_name": img_path.name,
                    "width": int(width_px),
                    "height": int(height_px),
                }
            )

            objects = frame.get("objects", [])
            yolo_lines = []

            for obj_data in objects:
                raw_class = obj_data.get("class", "")
                class_name = _normalize_class_name(raw_class, class_mode_internal)
                if not class_name:
                    continue

                if include_set is not None and class_name not in include_set:
                    continue
                if exclude_set is not None and class_name in exclude_set:
                    continue

                bbox = _bbox_from_projected_cuboid(obj_data.get("projected_cuboid"), width_px, height_px)
                if bbox is None:
                    continue

                if do_coco:
                    category_id = get_category_id(class_name)
                    coco["annotations"].append(
                        {
                            "id": next_annotation_id,
                            "image_id": img_id,
                            "category_id": category_id,
                            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                            "area": float(bbox[2] * bbox[3]),
                            "iscrowd": 0,
                        }
                    )
                    next_annotation_id += 1

                if do_yolo:
                    # YOLO expects class indices 0..N-1.
                    # We'll derive that from COCO categories (stable per run).
                    category_id = get_category_id(class_name)
                    cls_idx = category_id - 1
                    x, y, w, h = bbox
                    cx = (x + w / 2.0) / float(width_px)
                    cy = (y + h / 2.0) / float(height_px)
                    nw = w / float(width_px)
                    nh = h / float(height_px)
                    yolo_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if do_yolo:
                # Link/copy the image into YOLO layout.
                dst_img = yolo_images / img_path.name
                if not dst_img.exists():
                    try:
                        if yolo_use_symlinks:
                            dst_img.symlink_to(img_path)
                        else:
                            dst_img.write_bytes(img_path.read_bytes())
                    except Exception:
                        # fallback to copy
                        try:
                            dst_img.write_bytes(img_path.read_bytes())
                        except Exception:
                            pass

                (yolo_labels / f"{stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        if do_coco:
            coco["categories"] = [
                {"id": cid, "name": name, "supercategory": "none"}
                for name, cid in sorted(category_name_to_id.items(), key=lambda kv: kv[1])
            ]
            coco_filename = str(cfg.get("COCO_FILENAME", "annotations_coco.json")).strip() or "annotations_coco.json"
            (out_path / coco_filename).write_text(json.dumps(coco, indent=2), encoding="utf-8")
            print(f"[min_replicator] Wrote COCO annotations: {out_path / coco_filename}")

        if do_yolo:
            # Ultralytics-style data.yaml
            names = [name for name, _ in sorted(category_name_to_id.items(), key=lambda kv: kv[1])]
            data_yaml = {
                "path": str(yolo_root),
                "train": "images",
                "val": "images",
                "names": names,
            }
            (yolo_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
            print(f"[min_replicator] Wrote YOLO dataset: {yolo_root}")

    # Ensure output dir exists
    os.makedirs(args.output, exist_ok=True)

    kit = SimulationApp(launch_config=cfg.get("CONFIG", {}))

    # Force Replicator's disk backend root directories early (before writer/backend init).
    # Important: Replicator constructs paths like:
    #   <root_dir>/<writer.output_dir>/...
    # So we set root_dir to the *parent* of the requested output, and pass only the
    # folder name as PoseWriter.output_dir.
    try:
        import carb

        settings = carb.settings.get_settings()
        out_abs = os.path.abspath(args.output)
        out_root = os.path.dirname(out_abs)
        out_name = os.path.basename(out_abs)
        settings.set("/omni/replicator/backends/disk/root_dir", out_root)
        settings.set("/omni/replicator/backends/disk/default_root_dir", out_root)
    except Exception:
        # carb is only available inside Kit; safe to ignore for linting / static checks.
        pass

    import omni.replicator.core as rep
    from isaacsim.core.api import World
    try:
        # Available when running inside Isaac Sim / Kit.
        from isaacsim.storage.native import get_assets_root_path
    except Exception:
        get_assets_root_path = None

    def _vec3(value, *, default=None, name=""):
        """Convert a config value into a 3-tuple of floats (or return default)."""
        if value is None:
            return tuple(default) if default is not None else None
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"Expected {name} to be a 3-element list/tuple, got: {value!r}")
        return (float(value[0]), float(value[1]), float(value[2]))

    world = World(physics_dt=1.0 / 60.0)
    world.reset()

    # Isaac/Replicator scenes start empty unless you add something.
    # Add a neutral ground plane (optional). This can show up as a white plane in renders.
    # Disable it via config: USE_GROUND_PLANE: false
    if bool(cfg.get("USE_GROUND_PLANE", False)):
        ground_z = float(cfg.get("GROUND_Z", 0.0))
        ground_scale = cfg.get("GROUND_PLANE_SCALE", [10.0, 10.0, 1.0])
        try:
            rep.create.plane(
                position=(0, 0, ground_z),
                rotation=(0, 0, 0),
                scale=tuple(ground_scale),
            )
        except Exception:
            # Some Kit builds may not expose rep.create.plane; it's optional.
            pass

    # (Background backplate is created later, after camera_pos/look_at exist.)

    # Camera
    width = cfg.get("WIDTH", 640)
    height = cfg.get("HEIGHT", 480)
    focal_length_mm = ((cfg.get("F_X", 600) + cfg.get("F_Y", 600)) * cfg.get("pixel_size", 0.005)) / 2
    horizontal_aperture_mm = cfg.get("pixel_size", 0.005) * width

    # Put the camera in a sensible place and look at the object.
    # Your previous camera was at the origin with 0 rotation, while the object was
    # being randomized anywhere in [-5, 5] for x/y/z, so most frames had no object in view.
    camera_pos = tuple(cfg.get("CAMERA_POSITION", [0.0, -1.2, 0.6]))
    look_at = tuple(cfg.get("LOOK_AT", [0.0, 0.0, 0.35]))

    camera = rep.create.camera(
        position=camera_pos,
        look_at=look_at,
        focal_length=focal_length_mm,
        horizontal_aperture=horizontal_aperture_mm,
        clipping_range=(0.05, 1000),
    )
    render_product = rep.create.render_product(camera, (width, height))

    # Load provided USD object (fallback to a small cube if not specified)
    object_usd = cfg.get("OBJECT_USD")
    class_name = cfg.get("CLASS_NAME", "object")
    using_usd_asset = bool(object_usd and os.path.exists(object_usd))
    if using_usd_asset:
        obj = rep.create.from_usd(
            object_usd,
            semantics=[("class", class_name)],
        )
    else:
        obj = rep.create.cube(
            position=(0, 0, 0.35),
            scale=cfg.get("CUBE_SCALE", [0.08, 0.08, 0.08]),
            semantics=[("class", class_name)],
        )

    # --- Distractors (unlabeled / not annotated) ---
    # We intentionally DO NOT set any semantics on distractors. Writers like PoseWriter only
    # include prims with relevant semantics, so these act as visual noise only.
    #
    # IMPORTANT: Isaac Sim paths like "/Isaac/Props/YCB/Axis_Aligned/" are NOT filesystem paths.
    # To support those, we resolve them via Isaac's assets root (like the official
    # `standalone_examples/replicator/pose_generation/pose_generation.py`).
    distractor_count = int(cfg.get("DISTRACTOR_COUNT", 0))
    distractor_scale_min = tuple(cfg.get("DISTRACTOR_SCALE_MIN", [0.8, 0.8, 0.8]))
    distractor_scale_max = tuple(cfg.get("DISTRACTOR_SCALE_MAX", [1.2, 1.2, 1.2]))
    distractor_rot_min = _vec3(cfg.get("DISTRACTOR_ROT_MIN"), default=(0.0, 0.0, -180.0), name="DISTRACTOR_ROT_MIN")
    distractor_rot_max = _vec3(cfg.get("DISTRACTOR_ROT_MAX"), default=(0.0, 0.0, 180.0), name="DISTRACTOR_ROT_MAX")

    # Config options (supports both local and Isaac assets):
    # 1) DISTRACTOR_USDS: explicit list of USD paths (best for virtual paths)
    # 2) DISTRACTOR_ASSET_PATH + DISTRACTOR_FILENAMES: official-example style
    # 3) DISTRACTOR_ASSET_PATH alone (no filenames): auto-discover all USDs in that folder via omni.client.list()
    # 3) DISTRACTOR_DIR: local folder containing .usd/.usda/.usdc files (legacy)
    distractor_usds = cfg.get("DISTRACTOR_USDS")
    distractor_asset_path = cfg.get("DISTRACTOR_ASSET_PATH")
    distractor_filenames = cfg.get("DISTRACTOR_FILENAMES")
    distractor_dir = cfg.get("DISTRACTOR_DIR", os.path.join(os.path.dirname(__file__), "assets"))

    distractor_assets = []
    if distractor_count > 0:
        # (1) Explicit list always wins.
        if isinstance(distractor_usds, list) and distractor_usds:
            distractor_assets = [str(p) for p in distractor_usds]
        # (2) Isaac assets folder.
        # If filenames are provided, use that curated list.
        # If filenames are missing/empty, auto-discover all USDs in the folder.
        elif distractor_asset_path:
            assets_root_path = None
            if get_assets_root_path is not None:
                try:
                    assets_root_path = get_assets_root_path()
                except Exception:
                    assets_root_path = None
            if assets_root_path is None:
                # Some Isaac builds expose this helper from a different module path.
                try:
                    from omni.isaac.core.utils.nucleus import get_assets_root_path as _get_assets_root_path

                    assets_root_path = _get_assets_root_path()
                except Exception:
                    assets_root_path = None
            if assets_root_path is None:
                print(
                    "[min_replicator] WARNING: Could not resolve Isaac assets root; "
                    "can't expand DISTRACTOR_ASSET_PATH. "
                    "Try running with Isaac assets installed/mounted or use DISTRACTOR_USDS / DISTRACTOR_DIR."
                )
            else:
                # Note: DISTRACTOR_ASSET_PATH is usually like "/Isaac/Props/YCB/Axis_Aligned/".
                # The official example does: assets_root_path + DISTRACTOR_ASSET_PATH
                base = str(assets_root_path) + str(distractor_asset_path)

                # Ensure trailing slash so combine_urls behaves predictably.
                if not base.endswith("/"):
                    base = base + "/"

                if isinstance(distractor_filenames, list) and distractor_filenames:
                    for name in distractor_filenames:
                        # allow either with or without extension in YAML
                        if str(name).lower().endswith((".usd", ".usda", ".usdc")):
                            distractor_assets.append(f"{base}{name}")
                        else:
                            distractor_assets.append(f"{base}{name}.usd")
                else:
                    # Auto-discover everything in the folder.
                    try:
                        import omni

                        result, entries = omni.client.list(base)
                        if result != omni.client.Result.OK:
                            print(
                                "[min_replicator] WARNING: omni.client.list failed for "
                                f"{base!r} (result={result}). Provide DISTRACTOR_FILENAMES or DISTRACTOR_USDS."
                            )
                        else:
                            for entry in entries:
                                rel = str(entry.relative_path)
                                if rel.lower().endswith((".usd", ".usda", ".usdc")):
                                    distractor_assets.append(omni.client.combine_urls(base, rel))
                    except Exception as e:
                        print(
                            "[min_replicator] WARNING: Failed to auto-discover distractors from "
                            f"{base!r}: {e}. Provide DISTRACTOR_FILENAMES or DISTRACTOR_USDS."
                        )
        # (3) Legacy local folder scan (filesystem only)
        else:
            try:
                if os.path.isdir(distractor_dir):
                    for fn in sorted(os.listdir(distractor_dir)):
                        if fn.lower().endswith((".usd", ".usda", ".usdc")):
                            distractor_assets.append(os.path.join(distractor_dir, fn))
            except Exception:
                distractor_assets = []

    distractors = []
    if distractor_count > 0 and distractor_assets:
        # Helpful debug: show what we're about to sample from (virtual paths can be opaque).
        preview_list = ", ".join([str(p) for p in distractor_assets[:3]])
        more = "" if len(distractor_assets) <= 3 else f" (+{len(distractor_assets)-3} more)"
        print(f"[min_replicator] Resolved {len(distractor_assets)} distractor asset(s): {preview_list}{more}")

        # Choose a random asset per distractor instance (allows future multi-asset folders).
        for i in range(distractor_count):
            chosen_usd = random.choice(distractor_assets)
            # No semantics => not annotated.
            d = rep.create.from_usd(chosen_usd)
            distractors.append(d)
    elif distractor_count > 0 and not distractor_assets:
        print(
            "[min_replicator] WARNING: DISTRACTOR_COUNT="
            f"{distractor_count} but no distractor assets were resolved. "
            "Provide one of: DISTRACTOR_USDS, or (DISTRACTOR_ASSET_PATH + DISTRACTOR_FILENAMES), "
            f"or a real local folder in DISTRACTOR_DIR. Got DISTRACTOR_DIR={distractor_dir!r}"
        )

    # If you're spawning the Solo cup USD, keep its authored materials.
    # For the fallback cube, force a strong red PBR material so it's obvious.
    if not using_usd_asset:
        base_mat = rep.create.material_omnipbr(
            metallic=0.0,
            roughness=0.35,
            diffuse=(0.9, 0.05, 0.05),
            emissive_color=(0.0, 0.0, 0.0),
            emissive_intensity=0.0,
            count=1,
        )
        with obj:
            rep.randomizer.materials(base_mat)

    # Lighting: the previous sphere light was extremely bright and tended to wash out the render.
    # Use a couple of softer lights to keep highlights under control.
    # You can tune these with:
    #   LIGHTS_SPHERE_INTENSITY_SCALE (multiplier)
    #   LIGHT_KEY_INTENSITY / LIGHT_FILL_INTENSITY (absolute)
    sphere_scale = float(cfg.get("LIGHTS_SPHERE_INTENSITY_SCALE", 1.0))
    key_intensity = float(cfg.get("LIGHT_KEY_INTENSITY", 15000.0)) * sphere_scale
    fill_intensity = float(cfg.get("LIGHT_FILL_INTENSITY", 8000.0)) * sphere_scale
    rep.create.light(light_type="Sphere", intensity=key_intensity, position=(1.0, -0.5, 1.2), scale=0.3)
    rep.create.light(light_type="Sphere", intensity=fill_intensity, position=(-1.0, -0.8, 0.8), scale=0.3)

    dome_textures_base = cfg.get("DOME_TEXTURES_BASE")
    dome_textures = cfg.get("DOME_TEXTURES", [])
    use_dome = bool(cfg.get("USE_DOME_LIGHT", False))
    dome_intensity = float(cfg.get("DOME_LIGHT_INTENSITY", 1.0))
    dome_paths = []
    if use_dome and dome_textures_base:
        # If DOME_TEXTURES is provided (names without extension), use that curated list.
        if isinstance(dome_textures, list) and dome_textures:
            dome_paths = [os.path.join(dome_textures_base, f"{name}.hdr") for name in dome_textures]
        else:
            # Otherwise, auto-discover textures from the folder.
            # Supports local filesystem folders and (best-effort) Omniverse URLs via omni.client.
            base = str(dome_textures_base)
            try:
                if os.path.isdir(base):
                    for fn in sorted(os.listdir(base)):
                        if fn.lower().endswith((".hdr", ".exr")):
                            dome_paths.append(os.path.join(base, fn))
                else:
                    # Try omni.client for non-filesystem paths.
                    if "://" in base:
                        import omni

                        url = base if base.endswith("/") else base + "/"
                        result, entries = omni.client.list(url)
                        if result == omni.client.Result.OK:
                            for entry in entries:
                                rel = str(entry.relative_path)
                                if rel.lower().endswith((".hdr", ".exr")):
                                    dome_paths.append(omni.client.combine_urls(url, rel))
            except Exception as e:
                print(f"[min_replicator] WARNING: Failed to auto-discover dome textures from {base!r}: {e}")

        if use_dome and not dome_paths:
            print(
                "[min_replicator] WARNING: USE_DOME_LIGHT is enabled but no dome textures were found. "
                "Set DOME_TEXTURES (names) or place .hdr/.exr files under DOME_TEXTURES_BASE."
            )
    # We randomize rotation/texture per-frame below by creating a new Dome light
    # on each frame. Replicator's light "texture" is not exposed as a simple
    # writable USD attribute in all builds, so create-time randomization is the
    # most reliable approach.

    # Use realtime settings
    rep.settings.set_render_rtx_realtime()

    # Minimal writer using PoseWriter in DOPE format
    writer = rep.WriterRegistry.get("PoseWriter")
    # See the note above: output_dir should be a folder name relative to the disk root.
    out_dir_name = os.path.basename(os.path.abspath(args.output))
    writer.initialize(
        output_dir=out_dir_name,
        write_debug_images=False,
        format="dope",  # generates image + JSON
        skip_empty_frames=False,
        use_s3=False,
    )
    writer.attach([render_product])

    # Preview graph to build replicator graph once
    rep.orchestrator.preview()

    # --- Safer randomization defaults ---
    # Keep translations tight and spawn the cup ABOVE the ground.
    # Practical note: many USD assets have their pivot at the *center*, not the bottom.
    # If so, a pivot z==ground_z means "half the mesh is underground".
    # Prefer per-axis bounds (OBJ_POS_MIN/OBJ_POS_MAX). Fall back to legacy XY/Z keys.
    obj_pos_min = cfg.get("OBJ_POS_MIN")
    obj_pos_max = cfg.get("OBJ_POS_MAX")

    xy_min = cfg.get("OBJ_XY_MIN", [-0.10, -0.10])
    xy_max = cfg.get("OBJ_XY_MAX", [0.10, 0.10])
    ground_z = float(cfg.get("GROUND_Z", 0.0))

    # Height estimate to keep the cup bottom above the ground even if the pivot is centered.
    # If your cup USD's origin is already at the bottom, set CUP_HEIGHT_M to 0.0 in YAML.
    # If you're still seeing clipping, either increase CUP_HEIGHT_M or GROUND_CLEARANCE_M.
    cup_height_m = float(cfg.get("CUP_HEIGHT_M", 0.12))
    clearance_m = float(cfg.get("GROUND_CLEARANCE_M", 0.03))
    pivot_lift_m = 0.5 * max(cup_height_m, 0.0) + max(clearance_m, 0.0)

    # Preferred (new) Z randomization is expressed as "bottom of cup" height, then we lift pivot.
    # You can override with OBJ_Z_MIN/OBJ_Z_MAX (pivot height) if desired.
    obj_z_min = cfg.get("OBJ_Z_MIN")
    obj_z_max = cfg.get("OBJ_Z_MAX")
    if obj_z_min is not None or obj_z_max is not None:
        # Treat as pivot z range.
        z_min = float(obj_z_min) if obj_z_min is not None else float(ground_z + pivot_lift_m)
        z_max = float(obj_z_max) if obj_z_max is not None else float(z_min + 0.03)
    else:
        bottom_z_min = float(cfg.get("OBJ_BOTTOM_Z_MIN", ground_z + 0.0))
        bottom_z_max = float(cfg.get("OBJ_BOTTOM_Z_MAX", ground_z + 0.02))
        z_min = bottom_z_min + pivot_lift_m
        z_max = bottom_z_max + pivot_lift_m
    if z_max < z_min:
        z_max = z_min + 1e-3

    # Allow legacy MIN_POSITION/MAX_POSITION to override, but clamp Z to be safe.
    # (In your previous config these were ±5 meters, so the object was off-camera a lot.)
    min_pos = cfg.get("MIN_POSITION")
    max_pos = cfg.get("MAX_POSITION")

    if obj_pos_min is not None and obj_pos_max is not None:
        min_pos = list(_vec3(obj_pos_min, name="OBJ_POS_MIN"))
        max_pos = list(_vec3(obj_pos_max, name="OBJ_POS_MAX"))
        # Keep Z safe by clamping to the computed safe lift.
        min_pos[2] = max(float(min_pos[2]), z_min)
        max_pos[2] = max(float(max_pos[2]), min_pos[2] + 1e-3)
    elif min_pos is None or max_pos is None:
        min_pos = [xy_min[0], xy_min[1], z_min]
        max_pos = [xy_max[0], xy_max[1], z_max]
    else:
        min_pos = [min_pos[0], min_pos[1], max(float(min_pos[2]), z_min)]
        max_pos = [max_pos[0], max_pos[1], max(float(max_pos[2]), min_pos[2] + 1e-3)]

    # Object rotation bounds (degrees)
    obj_rot_min = _vec3(cfg.get("OBJ_ROT_MIN"), default=(0.0, -10.0, -25.0), name="OBJ_ROT_MIN")
    obj_rot_max = _vec3(cfg.get("OBJ_ROT_MAX"), default=(0.0, 10.0, 25.0), name="OBJ_ROT_MAX")

    # Camera pose bounds.
    # Prefer per-axis bounds (CAM_POS_MIN/CAM_POS_MAX and CAM_ROT_MIN/CAM_ROT_MAX).
    # If CAM_POSITIONS is provided, we keep supporting it.
    cam_positions = cfg.get("CAM_POSITIONS")
    cam_pos_min = cfg.get("CAM_POS_MIN")
    cam_pos_max = cfg.get("CAM_POS_MAX")
    cam_rot_min = cfg.get("CAM_ROT_MIN")
    cam_rot_max = cfg.get("CAM_ROT_MAX")

    # Defaults roughly match the previous curated list bounds.
    cam_pos_min_v = _vec3(cam_pos_min, default=(-0.90, -1.60, 0.55), name="CAM_POS_MIN")
    cam_pos_max_v = _vec3(cam_pos_max, default=(0.90, -0.70, 0.95), name="CAM_POS_MAX")

    # If you use look_at (below), camera rotation is optional. But sometimes you may
    # want to disable look_at and randomize Euler rotations directly.
    cam_rot_min_v = _vec3(cam_rot_min, default=(0.0, 0.0, 0.0), name="CAM_ROT_MIN")
    cam_rot_max_v = _vec3(cam_rot_max, default=(0.0, 0.0, 0.0), name="CAM_ROT_MAX")

    cam_use_look_at = bool(cfg.get("CAM_USE_LOOK_AT", True))
    cam_look_at_offset_min = _vec3(cfg.get("CAM_LOOK_AT_OFFSET_MIN"), default=(0.0, 0.0, 0.0), name="CAM_LOOK_AT_OFFSET_MIN")
    cam_look_at_offset_max = _vec3(cfg.get("CAM_LOOK_AT_OFFSET_MAX"), default=(0.0, 0.0, 0.0), name="CAM_LOOK_AT_OFFSET_MAX")

    with rep.trigger.on_frame():
        # Randomize HDR dome background per frame (optional).
        if use_dome and dome_paths:
            rep.create.light(
                light_type="Dome",
                rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
                texture=rep.distribution.choice(dome_paths),
                intensity=dome_intensity,
            )

        # Sample object pose.
        pos_dist = rep.distribution.uniform(tuple(min_pos), tuple(max_pos))
        with obj:
            rep.modify.pose(
                position=pos_dist,
                rotation=rep.distribution.uniform(obj_rot_min, obj_rot_max),
            )

            # Avoid overriding the Solo cup USD material (it can look gray if the random
            # material has high roughness + no good lighting).
            if not using_usd_asset:
                mats = rep.create.material_omnipbr(
                    metallic=0.0,
                    roughness=rep.distribution.uniform(0.25, 0.45),
                    diffuse=rep.distribution.uniform((0.75, 0.02, 0.02), (0.95, 0.15, 0.15)),
                    count=1,
                )
                rep.randomizer.materials(mats)

        # Randomize distractor poses (visual noise only; no semantics attached).
        # Default behavior: keep them near the object so they appear in-frame.
        if distractors:
            d_pos_min = _vec3(cfg.get("DISTRACTOR_POS_MIN"), default=(float(min_pos[0]) - 0.5, float(min_pos[1]) - 0.5, float(min_pos[2]) - 0.05), name="DISTRACTOR_POS_MIN")
            d_pos_max = _vec3(cfg.get("DISTRACTOR_POS_MAX"), default=(float(max_pos[0]) + 0.5, float(max_pos[1]) + 0.5, float(max_pos[2]) + 0.35), name="DISTRACTOR_POS_MAX")
            for d in distractors:
                with d:
                    rep.modify.pose(
                        # Important: create a *separate* distribution node per distractor.
                        # If you reuse the same distribution object for all prims, Replicator
                        # may evaluate it once per frame, causing all distractors to share the
                        # same sampled position.
                        position=rep.distribution.uniform(d_pos_min, d_pos_max),
                        rotation=rep.distribution.uniform(distractor_rot_min, distractor_rot_max),
                        scale=rep.distribution.uniform(distractor_scale_min, distractor_scale_max),
                    )

        # Move the camera.
        # Default behavior is to keep using look_at so the object stays centered.
        # If you set CAM_USE_LOOK_AT: false, then CAM_ROT_MIN/MAX will be used instead.
        with camera:
            if cam_positions is not None:
                cam_position_dist = rep.distribution.choice(cam_positions)
            else:
                cam_position_dist = rep.distribution.uniform(cam_pos_min_v, cam_pos_max_v)

            if cam_use_look_at:
                # ReplicatorItems don't support python-side arithmetic, so express
                # the "look_at jitter" as bounds around the target directly.
                look_at_target = rep.distribution.uniform(
                    (
                        float(min_pos[0]) + cam_look_at_offset_min[0],
                        float(min_pos[1]) + cam_look_at_offset_min[1],
                        float(min_pos[2]) + cam_look_at_offset_min[2],
                    ),
                    (
                        float(max_pos[0]) + cam_look_at_offset_max[0],
                        float(max_pos[1]) + cam_look_at_offset_max[1],
                        float(max_pos[2]) + cam_look_at_offset_max[2],
                    ),
                )
                # Better default: keep looking at the sampled object position.
                # If you want fully decoupled look targets, set CAM_USE_OBJECT_POS_AS_LOOK_AT: false.
                use_obj_look_at = bool(cfg.get("CAM_USE_OBJECT_POS_AS_LOOK_AT", True))
                rep.modify.pose(
                    position=cam_position_dist,
                    look_at=pos_dist if use_obj_look_at else look_at_target,
                )
            else:
                rep.modify.pose(
                    position=cam_position_dist,
                    rotation=rep.distribution.uniform(cam_rot_min_v, cam_rot_max_v),
                )

    # The very first frame after `preview()` often renders with incomplete state
    # (e.g., white background while materials / dome textures / RTX settle).
    # Warm up once and discard it, then write exactly `--frames` outputs.
    #
    # Important: `rep.orchestrator.step()` triggers writers, so we temporarily
    # detach the writer during warm-up to avoid an extra image on disk.
    writer.detach()
    rep.orchestrator.step(rt_subframes=2)
    writer.attach([render_product])
    print("[min_replicator] Discarded warm-up frame 0 (not written)")

    # Step for N frames (written)
    for idx in range(args.frames):
        rep.orchestrator.step(rt_subframes=2)
        print(f"[min_replicator] Wrote frame {idx+1}/{args.frames} to {args.output}")

    # Optional post-processing export for training datasets.
    # Converts PoseWriter's per-frame DOPE JSON into COCO (Detectron2-friendly) and/or YOLO.
    try:
        _export_coco_and_or_yolo(args.output, int(width), int(height), expected_frames=int(args.frames))
    except Exception as e:
        print(f"[min_replicator] WARNING: Dataset export failed: {e}")

    kit.close()


if __name__ == "__main__":
    main()
