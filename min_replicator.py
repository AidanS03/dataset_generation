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
    if min_pos is None or max_pos is None:
        min_pos = [xy_min[0], xy_min[1], z_min]
        max_pos = [xy_max[0], xy_max[1], z_max]
    else:
        min_pos = [min_pos[0], min_pos[1], max(float(min_pos[2]), z_min)]
        max_pos = [max_pos[0], max_pos[1], max(float(max_pos[2]), min_pos[2] + 1e-3)]

    # Camera viewpoints.
    # Replicator distributions don't support arbitrary arithmetic/trig in Python, so we use
    # a curated set of positions (overrideable via YAML).
    cam_positions = cfg.get("CAM_POSITIONS")
    if cam_positions is None:
        cam_positions = [
            (0.00, -1.60, 0.80),
            (0.00, -1.20, 0.55),
            (0.60, -1.10, 0.65),
            (-0.60, -1.10, 0.65),
            (0.90, -0.70, 0.75),
            (-0.90, -0.70, 0.75),
            (0.85, -1.40, 0.95),
            (-0.85, -1.40, 0.95),
        ]

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
                rotation=rep.distribution.uniform((0, -10, -25), (0, 10, 25)),
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

        # Move the camera but always look at the sampled object position.
        with camera:
            rep.modify.pose(
                position=rep.distribution.choice(cam_positions),
                look_at=pos_dist,
            )

    # Step for N frames
    for idx in range(args.frames):
        rep.orchestrator.step(rt_subframes=2)
        print(f"[min_replicator] Wrote frame {idx+1}/{args.frames} to {args.output}")

    kit.close()


if __name__ == "__main__":
    main()
