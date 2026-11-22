"""World: the validated scene with three movable bars (red, blue, green), plus the scripted skills.

Not run on its own: other scripts import it (scripts\\dense_pick_test.py, later the plan executor).
Needs scripts\\place_validation.py and scripts\\stack_unstack_validation.py next to it.
Slots (the PDDL locations): t1 (0.32,-0.15)  t2 (0.40,-0.15)  t3 (0.48,-0.15)  bin1 (0.32,-0.25)  bin2 (0.40,-0.25).
Bars: 5x5x10 cm, mass 0.1 kg, friction 0.6 (see the unstack friction sweep). Random start rotations avoid
38-54 degrees, where the scripted grasp is known to be unreliable (grasp validation).
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import place_validation as pv
import stack_unstack_validation as su

ROOT = pv.ROOT
WORLD_SRC = ROOT / "models" / "_world_src.xml"
ITEMS = ["red", "blue", "green"]
SLOTS = {"t1": (0.32, -0.15), "t2": (0.40, -0.15), "t3": (0.48, -0.15), "bin1": (0.32, -0.25), "bin2": (0.40, -0.25)}
FRICTION = 0.6
YAW_BAD = (38.0, 54.0)
BAR_H = 0.10
PARK = (0.40, 0.35)            # where an unused bar is put (off the table, it falls to the floor)
LIFT_BY = 0.12                 # lift after grasping: carried bars clear the 10 cm bars standing next to them
START_JITTER = 0.005


def sample_yaw(rng):
    while True:
        y = float(rng.uniform(0.0, 90.0))
        if not (YAW_BAD[0] <= y <= YAW_BAD[1]):
            return y


class World:
    def __init__(self):
        text = (ROOT / "models" / "scene_pick.xml").read_text(encoding="utf-8")
        assert text.count("</worldbody>") == 1, "scene_pick.xml has more than one </worldbody>; tell Claude"
        bodies = ""
        for k, (name, rgba) in enumerate((("blue", "0.15 0.35 0.85 1"), ("green", "0.2 0.7 0.25 1"))):
            bodies += (f'<body name="bar_{name}" pos="0.30 {-0.40 - 0.05 * k} 0.80">\n'
                       f'      <freejoint name="{name}_joint"/>\n'
                       f'      <geom name="{name}_geom" type="box" size="0.025 0.025 0.05" mass="0.1" rgba="{rgba}"/>\n'
                       f'    </body>\n')
        WORLD_SRC.write_text(text.replace("</worldbody>", bodies + "</worldbody>"), encoding="utf-8")
        pv.SCENE_SRC = WORLD_SRC
        pv.load_scene(pv.SHAPE)
        m = pv.model
        self.body = {"red": pv.cube_body, "blue": m.body("bar_blue").id, "green": m.body("bar_green").id}
        self.qadr = {"red": pv.cube_qadr, "blue": m.joint("blue_joint").qposadr[0], "green": m.joint("green_joint").qposadr[0]}
        for g in ("cube_geom", "blue_geom", "green_geom"):
            m.geom_friction[m.geom(g).id, 0] = FRICTION
        self.table_z = pv.table_z

    # ---- scene ---------------------------------------------------------------------------
    def reset(self, placement, rng):
        """placement: {item: ('slot', name) | ('on', base_item)}. Items not listed are parked off the table."""
        d, m = pv.data, pv.model
        mujoco.mj_resetDataKeyframe(m, d, pv.key)
        d.ctrl[:] = m.key_ctrl[pv.key]
        d.qpos[pv.waist_qadr] = list(pv.STANCE)
        d.ctrl[pv.waist_act] = list(pv.STANCE)
        yaws = {}
        for item in ITEMS:                                    # always draw in the same order: same seed, same scene
            yaws[item] = sample_yaw(rng)
        jit = {item: (rng.uniform(-START_JITTER, START_JITTER), rng.uniform(-START_JITTER, START_JITTER)) for item in ITEMS}
        pos = {}
        for item in ITEMS:
            kind = placement.get(item)
            if kind is None:
                su.set_bar(self.qadr[item], PARK, np.radians(yaws[item]))
                pos[item] = (PARK, yaws[item])
            elif kind[0] == "slot":
                xy = (SLOTS[kind[1]][0] + jit[item][0], SLOTS[kind[1]][1] + jit[item][1])
                su.set_bar(self.qadr[item], xy, np.radians(yaws[item]))
                pos[item] = (xy, yaws[item])
        for item in ITEMS:
            kind = placement.get(item)
            if kind is not None and kind[0] == "on":
                (bxy, byaw) = pos[kind[1]]
                yaw = byaw + float(rng.uniform(-8.0, 8.0))
                su.set_bar(self.qadr[item], bxy, np.radians(yaw))
                d.qpos[self.qadr[item] + 2] = self.table_z + BAR_H + 0.001
                pos[item] = (bxy, yaw)
        mujoco.mj_forward(m, d)
        pv.step(1.0)

    def xy(self, item):
        return pv.data.xpos[self.body[item]][:2].copy()

    def z(self, item):
        return float(pv.data.xpos[self.body[item]][2])

    def tilt(self, item):
        return su.tilt_of(self.body[item])

    # ---- skills --------------------------------------------------------------------------
    def pick(self, item, lift_by=LIFT_BY):
        """Grasp the bar (from the table or from a stack) and lift it. Returns a grasp context or None if not lifted.
        From a stack use lift_by=su.UNSTACK_LIFT (0.04): lifting higher near the body was out of reach at t1."""
        ctx = su.pick(self.body[item], None, lift_by)
        return ctx if ctx["lift"] >= 0.6 * lift_by else None

    def place_table(self, item, slot, ctx):
        """Carry the held bar to a table or bin slot and put it down. Returns None or a failure reason."""
        dst = np.array([SLOTS[slot][0], SLOTS[slot][1], self.table_z])
        return su.carry_and_release(self.body[item], ctx, dst, su.TABLE_CLEAR, 0.0)

    def place_on(self, item, base, ctx):
        """Carry the held bar over another bar and put it on top. Returns None or a failure reason."""
        b = pv.data.xpos[self.body[base]]
        dst = np.array([b[0], b[1], b[2] + BAR_H])
        return su.carry_and_release(self.body[item], ctx, dst, su.STACK_CLEAR, 0.06)
