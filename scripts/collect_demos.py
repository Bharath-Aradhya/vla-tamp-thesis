"""Demonstration recorder: run the scripted skills through the executor and save what a VLA would need to learn them.

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\collect_demos.py --tasks p01 --reps 1               # quick test: one episode, check time and size
    python scripts\\collect_demos.py --tasks p01 p02 p08 p09 p10 --reps 20 --out demos

Every episode is one run of the executor on one task and scene (see executor.py). While the skills move the arm, every
1/hz seconds of simulated time (default 10 Hz) one frame is recorded:
  top, side   224 x 224 RGB images from the two fixed cameras (render_test.py)
  state       measured arm joint angles (7) + measured finger joint angles (one per hand actuator) <- what the robot "feels"
  action      the position target the skill sent to the same actuators             <- what the VLA must output
  bars        true xyz of the red/blue/green bars (for debugging and checks, NOT for the VLA)
  seg         index of the skill segment the frame belongs to
and per segment a language instruction such as "pick up the red bar from table spot 1".
Saved to <out>\\ep_<task>_<seed>.npz (compressed) with success, replans and the segment list as metadata.
Episodes that needed a replan or failed are saved too (flag in the file name: _ok / _bad); train on the _ok ones.
Existing files are skipped, so the script can be stopped (Ctrl+C) and restarted. CPU only; rendering costs about 0.2 s per
image, so expect a few minutes per episode. Needs scripts\\executor.py, world.py, place_validation.py, stack_unstack_validation.py.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import executor as E  # noqa: E402

SIZE = 224
VIEWS = {"top": (0.40, -0.15, 0.78, 0.80, 180.0, -89.0), "side": (0.40, -0.15, 0.80, 0.90, 90.0, -25.0)}
SLOT_WORDS = {"t1": "table spot 1", "t2": "table spot 2", "t3": "table spot 3", "bin1": "bin 1", "bin2": "bin 2"}


def instruction(kind, item, target):
    if kind == "pick":
        return f"pick up the {item} bar"
    if kind == "place":
        return f"place the {item} bar at {SLOT_WORDS[target]}"
    return f"stack the {item} bar on the {target} bar"


class Recorder:
    def __init__(self, hz, views):
        self.pv = E.sys.modules["world"].pv
        self.hz, self.views = hz, views
        self.every = max(1, round(1.0 / (hz * self.pv.model.opt.timestep)))
        self.renderer = mujoco.Renderer(self.pv.model, height=SIZE, width=SIZE)
        self.cams = {}
        for name in views:
            spec = VIEWS[name]
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = spec[:3]
            cam.distance, cam.azimuth, cam.elevation = spec[3], spec[4], spec[5]
            self.cams[name] = cam
        self.idx = np.array(list(self.pv.arm_act) + list(self.pv.HAND_ACT))
        m = self.pv.model
        self.hand_qadr = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in self.pv.HAND_ACT])   # measured finger angles
        self.active, self.count, self.seg = False, 0, -1
        self.segments, self.frames = [], {k: [] for k in ("state", "action", "bars", "seg", *views)}
        self.bar_bodies = None
        self._orig = mujoco.mj_step
        mujoco.mj_step = self._step                  # every motion in place_validation.py goes through mj_step

    def _step(self, model, data, *a, **k):
        self._orig(model, data, *a, **k)
        if self.active:
            self.count += 1
            if self.count % self.every == 0:
                self.capture(model, data)

    def capture(self, model, data):
        for name, cam in self.cams.items():
            self.renderer.update_scene(data, camera=cam)
            self.frames[name].append(self.renderer.render().copy())
        pv = self.pv
        q = np.concatenate([data.qpos[pv.qadr], data.qpos[self.hand_qadr]])   # measured arm and finger angles
        self.frames["state"].append(q.astype(np.float32))
        self.frames["action"].append(data.ctrl[self.idx].astype(np.float32))
        self.frames["bars"].append(np.array([data.xpos[b] for b in self.bar_bodies], np.float32))
        self.frames["seg"].append(self.seg)

    def start(self, bar_bodies):
        self.bar_bodies = bar_bodies
        self.active, self.count, self.seg, self.segments = True, 0, -1, []
        self.frames = {k: [] for k in self.frames}

    def new_segment(self, text):
        self.seg += 1
        self.segments.append(text)

    def stop(self):
        self.active = False


class RecBackend(E.MujocoBackend):
    def __init__(self, seed, rec_args):
        super().__init__(seed)
        self.rec = Recorder(*rec_args)

    def reset(self, placement, items, seed):
        super().reset(placement, items, seed)
        self.rec.start([self.body_of(i) for i in ("red", "blue", "green")])

    def body_of(self, i):
        return self.w.body[i]

    def pick(self, item):
        self.rec.new_segment(instruction("pick", item, None))
        return super().pick(item)

    def place(self, item, slot):
        self.rec.new_segment(instruction("place", item, slot))
        return super().place(item, slot)

    def stack(self, item, base):
        self.rec.new_segment(instruction("stack", item, base))
        return super().stack(item, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--views", nargs="+", default=["top", "side"], choices=list(VIEWS))
    ap.add_argument("--out", default="demos")
    ap.add_argument("--domain", default=str(E.ROOT / "pddl" / "domain_skills.pddl"))
    args = ap.parse_args()
    out = E.ROOT / args.out
    out.mkdir(exist_ok=True)
    planner, domain = E.make_planner(args.domain, False)
    backend = RecBackend(0, (args.hz, args.views))
    print(f"recording every {backend.rec.every} simulation steps = {args.hz} Hz, views {args.views}", flush=True)
    t0, n_ok, n = time.time(), 0, 0
    for task in args.tasks:
        f = sorted((E.ROOT / "pddl" / "problems").glob(f"{task}*.pddl"))[0]
        problem = E.parse_problem(f)
        for rep in range(args.reps):
            seed = args.seed0 + rep
            if any(out.glob(f"ep_{task}_{seed}_*.npz")):
                print(f"{task} seed {seed}: exists, skipped")
                continue
            t1 = time.time()
            ok, replans, skills, first, why, dev = E.run_task(backend, planner, domain, problem, seed, verbose=False)
            rec = backend.rec
            rec.stop()
            good = ok and replans == 0
            n, n_ok = n + 1, n_ok + good
            T = len(rec.frames["seg"])
            path = out / f"ep_{task}_{seed}_{'ok' if good else 'bad'}.npz"
            data = {k: np.array(v) for k, v in rec.frames.items() if v}
            np.savez_compressed(path, segments=json.dumps(rec.segments), meta=json.dumps(
                {"task": task, "seed": seed, "success": bool(ok), "replans": int(max(replans, 0)), "why": why, "hz": args.hz}), **data)
            print(f"{task} seed {seed}: {'ok' if good else 'bad (' + why + ')'}, {T} frames, {len(rec.segments)} skills, "
                  f"{path.stat().st_size / 1e6:.1f} MB, {time.time() - t1:.0f}s   (total {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"\n{n_ok}/{n} clean episodes saved in {out}")


if __name__ == "__main__":
    main()
