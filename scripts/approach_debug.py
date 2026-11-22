"""Approach debug: WHICH step of the pick knocks the bar on the bin row?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\approach_debug.py

The approach-height test showed that raising the approach from 5 to 15 cm cut the damage to the bin-row bar from
12-19 cm to 2-7 cm, but never to zero, and at 20 cm even picks without any neighbour failed. So the approach
height is not the whole story. This replays 4 picks (t1 with a bar at bin1; t2 with bin1; t2 with bin2; t3 with bin2)
and prints, after EVERY step of the pick (the two raise-first waypoints, pre-grasp, half way, grasp point, closing,
lift), how far the bin-row bar has moved. It also saves a picture sheet per case (approach_debug_<n>.png,
two views per step). Needs scripts\\world.py. Expect about 1 minute. CPU only.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import world as W

pv, su = W.pv, W.su
CASES = [("t1", "bin1"), ("t2", "bin1"), ("t2", "bin2"), ("t3", "bin2")]
_renderer = None


def snap(frames, label):
    global _renderer
    if _renderer is None:
        _renderer = mujoco.Renderer(pv.model, height=360, width=480)
    imgs = []
    for az, el in ((90, -20), (180, -20)):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.38, -0.20, 0.85]
        cam.distance, cam.azimuth, cam.elevation = 0.8, az, el
        _renderer.update_scene(pv.data, camera=cam)
        imgs.append(Image.fromarray(_renderer.render()).resize((360, 270)))
    row = Image.new("RGB", (720, 270))
    row.paste(imgs[0], (0, 0))
    row.paste(imgs[1], (360, 0))
    ImageDraw.Draw(row).text((6, 6), label, fill=(255, 255, 255))
    frames.append(row)


def main():
    w = W.World()
    for n, (p, q) in enumerate(CASES, 1):
        print(f"\n=== pick at {p}, other bar at {q} ===")
        w.reset({"red": ("slot", p), "green": ("slot", q)}, np.random.default_rng(4000))
        g0 = w.xy("green")
        frames = []
        d = pv.data
        body = w.body["red"]
        a = pv.hand_yaw("bar", su.bar_yaw(body))
        r_des = pv.rot_z(a)
        pre = r_des @ pv.PRE
        offset = np.array([pv.DX, pv.OY, 0.0])
        grasp = d.xpos[body].copy() + np.array([0.0, 0.0, pv.DZ])

        def log(label):
            mv = float(np.linalg.norm(w.xy("green") - g0)) * 100
            print(f"   {label:28s} bin-row bar moved {mv:5.1f} cm", flush=True)
            snap(frames, f"{p}+{q}: {label} ({mv:.1f} cm)")

        log("start")
        for label, pt, sec in (("raise 1 (0.12,-0.25,0.80)", [0.12, -0.25, 0.80], 1.5),
                               ("raise 2 (0.30,-0.25,0.92)", [0.30, -0.25, 0.92], 1.5),
                               ("pre-grasp high", grasp + pre + np.array([0.0, 0.0, 0.03]), 1.5),
                               ("pre-grasp", grasp + pre, 1.0), ("half way", grasp + 0.5 * pre, 1.0), ("grasp point", grasp, 1.0)):
            pv.go_to(pt, sec, offset, r_des)
            log(label)
        target = d.ctrl.copy()
        for nm, v in pv.FINGERS.items():
            target[pv.act(f"right_hand_{nm}_joint")] = v
        for j, v in enumerate(pv.THUMB):
            target[pv.act(f"right_hand_thumb_{j}_joint")] = v
        pv.move(target, 1.5)
        pv.step(0.5)
        log("fingers closed")
        pv.go_to(grasp + np.array([0.0, 0.0, 0.12]), 1.5, offset, r_des)
        pv.step(1.0)
        log("lifted 12 cm")
        sheet = Image.new("RGB", (720 * 2, 270 * ((len(frames) + 1) // 2)))
        for i, f in enumerate(frames):
            sheet.paste(f, ((i % 2) * 720, (i // 2) * 270))
        out = W.ROOT / f"approach_debug_{n}.png"
        sheet.save(out)
        print(f"   saved {out.name}")


if __name__ == "__main__":
    main()
