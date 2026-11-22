"""Render test: can we get camera images out of the scene (needed to record demonstrations for the VLA)?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\render_test.py

Resets the scene with three bars (red t1, blue t2, green t3), then saves 3 images from free (virtual) cameras,
224 x 224 pixels, the usual VLA input size, plus times how long one render takes:
  render_front.png   a third-person view from the robot's front, looking down at the table
  render_top.png     a view from above the table
  render_side.png    a view from the side
Also moves the arm to the pre-grasp pose above red and saves render_grasp.png, so you can see the hand and the bar.
Needs scripts\\world.py, place_validation.py, stack_unstack_validation.py. About 1 minute. CPU only (Intel graphics is fine).
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import world as W

SIZE = 224
VIEWS = {  # name: (lookat x, y, z, distance, azimuth deg, elevation deg)
    "front": (0.40, -0.15, 0.80, 0.90, 180.0, -35.0),
    "top": (0.40, -0.15, 0.78, 0.80, 180.0, -89.0),
    "side": (0.40, -0.15, 0.80, 0.90, 90.0, -25.0),
}


def shoot(renderer, data, spec, path):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = spec[:3]
    cam.distance, cam.azimuth, cam.elevation = spec[3], spec[4], spec[5]
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    import imageio.v2 as imageio
    imageio.imwrite(path, img)
    return img


def main():
    w = W.World()
    w.reset({"red": ("slot", "t1"), "blue": ("slot", "t2"), "green": ("slot", "t3")}, np.random.default_rng(1))
    model, data = W.pv.model, W.pv.data
    try:
        renderer = mujoco.Renderer(model, height=SIZE, width=SIZE)
    except Exception as e:
        sys.exit(f"Could not create an offscreen renderer: {e!r}\nTell Claude this message.")
    for name, spec in VIEWS.items():
        t0 = time.time()
        img = shoot(renderer, data, spec, W.ROOT / f"render_{name}.png")
        print(f"{name}: image {img.shape}, mean brightness {img.mean():.0f}, {1000 * (time.time() - t0):.0f} ms")
    # arm at the pre-grasp pose over red: run the first part of a pick
    try:
        W.su.pick(w.body["red"], None, 0.02)
    except Exception as e:
        print("pick for the grasp image failed:", repr(e))
    shoot(renderer, data, VIEWS["front"], W.ROOT / "render_grasp.png")
    t0 = time.time()
    for _ in range(20):
        renderer.update_scene(data)
        renderer.render()
    print(f"average render time: {1000 * (time.time() - t0) / 20:.0f} ms per frame")
    print(f"Saved render_front.png, render_top.png, render_side.png, render_grasp.png in {W.ROOT}")


if __name__ == "__main__":
    main()
