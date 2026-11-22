"""Stack debug: WHEN does the base bar move, and what does each failure look like?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\stack_debug.py

Replays 4 trials from stack_unstack_104158.csv (a stack that slid off, a bar that could not be lifted off the
stack, an unstack that knocked the base, an unstack whose carry could not reach). At every step of the stack
and unstack it prints how far the base bar has moved from where it started and where the top bar is, and saves
a picture per step (stack_debug_<trial>.png, 2 camera views per step). Needs scripts\\place_validation.py and
scripts\\stack_unstack_validation.py. Expect about 2 minutes. CPU only.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import place_validation as pv
import stack_unstack_validation as su

TRIALS = {  # name: (src_xy, src_yaw_deg, base_xy, base_yaw_deg)
    "A_stack_slid_off_bin2_t1": ((0.398, -0.251), 55.9, (0.317, -0.155), 24.2),
    "B_unstack_pick_failed_bin2_t2": ((0.392, -0.240), 38.3, (0.406, -0.148), 84.0),
    "C_unstack_knocked_base_bin1_t2": ((0.315, -0.254), 55.7, (0.404, -0.157), 44.9),
    "D_unstack_unreachable_bin1_t1": ((0.320, -0.253), 10.1, (0.322, -0.145), 9.8),
}
_renderer = None
frames = []
base0 = None


def snap(label):
    global _renderer
    if _renderer is None:
        _renderer = mujoco.Renderer(pv.model, height=360, width=480)
    imgs = []
    for az, el in ((90, -15), (180, -15)):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.38, -0.15, 0.88]
        cam.distance, cam.azimuth, cam.elevation = 0.8, az, el
        _renderer.update_scene(pv.data, camera=cam)
        imgs.append(Image.fromarray(_renderer.render()).resize((360, 270)))
    row = Image.new("RGB", (720, 270))
    row.paste(imgs[0], (0, 0))
    row.paste(imgs[1], (360, 0))
    ImageDraw.Draw(row).text((6, 6), label, fill=(255, 255, 255))
    frames.append(row)


def log(label):
    d = pv.data
    shift = np.linalg.norm(d.xpos[su.base_body][:2] - base0[:2]) * 100
    t = d.xpos[pv.cube_body]
    bt = su.tilt_of(su.base_body)
    print(f"   {label:34s} base moved {shift:5.1f} cm (tilt {bt:4.1f})   top bar at "
          f"({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})", flush=True)
    snap(label)


def pick(body, tag):
    d = pv.data
    a = pv.hand_yaw("bar", su.bar_yaw(body))
    r_des = pv.rot_z(a)
    pre = r_des @ pv.PRE
    offset = np.array([pv.DX, pv.OY, 0.0])
    c0 = d.xpos[body].copy()
    grasp = c0 + np.array([0.0, 0.0, pv.DZ])
    pv.go_to([0.12, -0.25, 0.80], 1.5, offset, r_des)
    pv.go_to([0.30, -0.25, 0.92], 1.5, offset, r_des)
    log(f"{tag}: above start")
    pv.go_to(grasp + pre + np.array([0.0, 0.0, 0.03]), 1.5, offset, r_des)
    pv.go_to(grasp + pre, 1.0, offset, r_des)
    log(f"{tag}: at pre-grasp")
    pv.go_to(grasp + 0.5 * pre, 1.0, offset, r_des)
    pv.go_to(grasp, 1.0, offset, r_des)
    log(f"{tag}: at grasp point")
    target = d.ctrl.copy()
    for n, v in pv.FINGERS.items():
        target[pv.act(f"right_hand_{n}_joint")] = v
    for j, v in enumerate(pv.THUMB):
        target[pv.act(f"right_hand_thumb_{j}_joint")] = v
    pv.move(target, 1.5)
    pv.step(0.5)
    log(f"{tag}: fingers closed")
    pv.go_to(grasp + np.array([0.0, 0.0, 0.10]), 1.5, offset, r_des)
    pv.step(1.0)
    log(f"{tag}: lifted 10 cm")
    return dict(r_des=r_des, pre=pre, offset=offset, lift=float(d.xpos[body][2] - c0[2]))


def carry(body, ctx, dst3, clearance, tag):
    d, m = pv.data, pv.model
    r_des, pre, offset = ctx["r_des"], ctx["pre"], ctx["offset"]
    gp = lambda: d.xpos[pv.wrist] + d.xmat[pv.wrist].reshape(3, 3) @ offset
    rel = d.xpos[body] - gp()
    aim = dst3 - r_des @ np.array([pv.CALIB[0], pv.CALIB[1], 0.0])
    ik0, _, _ = pv.go_to(gp() + np.array([0.0, 0.0, 0.06]), 1.0, offset, r_des)
    log(f"{tag}: extra lift (IK err {ik0 * 100:.1f} cm)")
    ik, _, _ = pv.go_to(aim + np.array([0.0, 0.0, clearance]) - rel, 1.8, offset, r_des)
    pv.step(0.5)
    log(f"{tag}: over target (IK err {ik * 100:.1f} cm)")
    if ik > 0.03:
        print(f"   -> carry would be reported as 'arm could not reach' (IK err {ik * 100:.1f} cm)")
    pv.go_to(aim + np.array([0.0, 0.0, 0.004]) - rel, 1.5, offset, r_des)
    pv.step(0.3)
    log(f"{tag}: lowered")
    q, _ = pv.solve_ik(d.qpos[pv.qadr].copy(), gp() + r_des @ np.array([-pv.BACKUP, 0.0, 0.0]), offset, r_des)
    target = d.ctrl.copy()
    target[pv.arm_act] = q
    target[np.array(pv.HAND_ACT)] = m.key_ctrl[pv.key][pv.HAND_ACT]
    pv.move(target, 1.0)
    pv.step(0.5)
    log(f"{tag}: released, backed up")
    pv.go_to(gp() + np.array([0.0, 0.0, 0.06]), 1.0, offset, r_des)
    pv.go_to(gp() + np.array([pre[0], pre[1], 0.0]), 1.2, offset, r_des)
    pv.step(1.0)
    log(f"{tag}: retreated")


def main():
    global base0, frames
    su.build_scene()
    for name, (sxy, syaw, bxy, byaw) in TRIALS.items():
        print(f"\n=== {name} ===")
        frames = []
        su.reset_trial(sxy, np.radians(syaw), bxy, np.radians(byaw))
        base0 = pv.data.xpos[su.base_body].copy()
        log("start")
        ctx = pick(pv.cube_body, "stack/pick")
        if ctx["lift"] >= 0.06:
            carry(pv.cube_body, ctx, np.array([base0[0], base0[1], base0[2] + su.BAR_H]), su.STACK_CLEAR, "stack/carry")
            if name[0] in "BCD":
                ctx = pick(pv.cube_body, "unstack/pick")
                print(f"   lift of top bar: {ctx['lift'] * 100:.1f} cm")
                if ctx["lift"] >= 0.06:
                    carry(pv.cube_body, ctx, np.array([sxy[0], sxy[1], pv.table_z]), su.TABLE_CLEAR, "unstack/carry")
        sheet = Image.new("RGB", (720 * 2, 270 * ((len(frames) + 1) // 2)))
        for i, f in enumerate(frames):
            sheet.paste(f, ((i % 2) * 720, (i // 2) * 270))
        out = pv.ROOT / f"stack_debug_{name[0]}.png"
        sheet.save(out)
        print(f"   saved {out.name}")


if __name__ == "__main__":
    main()
