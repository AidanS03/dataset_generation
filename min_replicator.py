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

from isaacsim import SimulationApp


def main():
    parser = argparse.ArgumentParser("Minimal Replicator")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config/min_config.yaml"))
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "output"))
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

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
    distractor_dir = cfg.get(
        "DISTRACTOR_DIR",
        os.path.join(os.path.dirname(__file__), "assets"),
    )
    distractor_count = int(cfg.get("DISTRACTOR_COUNT", 0))
    distractor_scale_min = tuple(cfg.get("DISTRACTOR_SCALE_MIN", [0.8, 0.8, 0.8]))
    distractor_scale_max = tuple(cfg.get("DISTRACTOR_SCALE_MAX", [1.2, 1.2, 1.2]))
    distractor_rot_min = _vec3(cfg.get("DISTRACTOR_ROT_MIN"), default=(0.0, 0.0, -180.0), name="DISTRACTOR_ROT_MIN")
    distractor_rot_max = _vec3(cfg.get("DISTRACTOR_ROT_MAX"), default=(0.0, 0.0, 180.0), name="DISTRACTOR_ROT_MAX")

    distractor_assets = []
    if distractor_count > 0:
        try:
            if os.path.isdir(distractor_dir):
                for fn in sorted(os.listdir(distractor_dir)):
                    if fn.lower().endswith((".usd", ".usda", ".usdc")):
                        distractor_assets.append(os.path.join(distractor_dir, fn))
        except Exception:
            distractor_assets = []

    distractors = []
    if distractor_count > 0 and distractor_assets:
        # Choose a random asset per distractor instance (allows future multi-asset folders).
        for i in range(distractor_count):
            chosen_usd = random.choice(distractor_assets)
            # No semantics => not annotated.
            d = rep.create.from_usd(chosen_usd)
            distractors.append(d)
    elif distractor_count > 0 and not distractor_assets:
        print(f"[min_replicator] WARNING: DISTRACTOR_COUNT={distractor_count} but no .usd/.usda/.usdc found in {distractor_dir}")

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
    if use_dome and dome_textures_base and dome_textures:
        # Build HDR texture paths (expects names without extension)
        dome_paths = [os.path.join(dome_textures_base, f"{name}.hdr") for name in dome_textures]
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
            d_pos_dist = rep.distribution.uniform(d_pos_min, d_pos_max)
            d_scale_dist = rep.distribution.uniform(distractor_scale_min, distractor_scale_max)

            for d in distractors:
                with d:
                    rep.modify.pose(
                        position=d_pos_dist,
                        rotation=rep.distribution.uniform(distractor_rot_min, distractor_rot_max),
                        scale=d_scale_dist,
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

    kit.close()


if __name__ == "__main__":
    main()
