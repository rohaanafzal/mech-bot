"""Mech Royale bot (protocol mechfight/1). Standard library only.

Layers:
  Nav      - precomputed walk grid (3 m) + surface height grid (2 m): A*, walk-line and line-of-sight tests
  Bot.act  - zone safety > retreat to cover > fight > loot > roam; aim/fire control is independent of legs
"""
import sys
import os
import json
import math
import time
import random
import heapq
import array
import threading
import queue
import traceback

PI = math.pi
TAU = 2 * math.pi
HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    try:
        print(*a, file=sys.stderr, flush=True)
    except Exception:
        pass


def angdiff(a, b):
    """a - b wrapped to [-pi, pi]."""
    d = (a - b) % TAU
    if d > PI:
        d -= TAU
    return d


def norm2(x, y):
    d = math.hypot(x, y)
    if d < 1e-9:
        return 0.0, 0.0
    return x / d, y / d


# ----------------------------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------------------------
class Nav:
    def __init__(self, path):
        self.ok = False
        self.nm_cache = {}
        try:
            with open(path, "rb") as f:
                hdr = json.loads(f.readline().decode())
                self.minX = hdr["minX"]
                self.minY = hdr["minY"]
                self.snx = hdr["snx"]
                self.sny = hdr["sny"]
                self.sc = hdr["sc"]
                self.nnx = hdr["nnx"]
                self.nny = hdr["nny"]
                self.nc = hdr["nc"]
                s = array.array("h")
                s.frombytes(f.read(self.snx * self.sny * 2))
                if sys.byteorder != "little":
                    s.byteswap()
                self.surf = s
                self.nav = f.read(self.nnx * self.nny)
            self.ok = len(self.surf) == self.snx * self.sny and len(self.nav) == self.nnx * self.nny
        except Exception:
            log("nav load failed", traceback.format_exc())
            self.ok = False

    # --- grids
    def surf_at(self, x, y):
        i = int((x - self.minX) / self.sc)
        j = int((y - self.minY) / self.sc)
        if i < 0:
            i = 0
        elif i >= self.snx:
            i = self.snx - 1
        if j < 0:
            j = 0
        elif j >= self.sny:
            j = self.sny - 1
        return self.surf[j * self.snx + i] * 0.1

    def cell(self, x, y):
        i = int((x - self.minX) / self.nc)
        j = int((y - self.minY) / self.nc)
        if i < 0:
            i = 0
        elif i >= self.nnx:
            i = self.nnx - 1
        if j < 0:
            j = 0
        elif j >= self.nny:
            j = self.nny - 1
        return j * self.nnx + i

    def val(self, x, y):
        i = int((x - self.minX) / self.nc)
        j = int((y - self.minY) / self.nc)
        if i < 0 or j < 0 or i >= self.nnx or j >= self.nny:
            return 0
        return self.nav[j * self.nnx + i]

    def center(self, c):
        return (self.minX + (c % self.nnx + 0.5) * self.nc, self.minY + (c // self.nnx + 0.5) * self.nc)

    def nearest_main(self, x, y, maxr=45):
        nnx, nny, nav = self.nnx, self.nny, self.nav
        ci = int((x - self.minX) / self.nc)
        cj = int((y - self.minY) / self.nc)
        ci = min(max(ci, 0), nnx - 1)
        cj = min(max(cj, 0), nny - 1)
        if nav[cj * nnx + ci] == 1:
            return cj * nnx + ci
        key = (cj * nnx + ci) * 128 + maxr
        cache = self.nm_cache
        if key in cache:
            return cache[key]
        r = self._nearest_main(ci, cj, maxr)
        if len(cache) > 20000:
            cache.clear()
        cache[key] = r
        return r

    def _nearest_main(self, ci, cj, maxr):
        nnx, nny, nav = self.nnx, self.nny, self.nav
        best = None
        bd = 1e18
        for r in range(1, maxr + 1):
            if best is not None and (r - 1) * (r - 1) > bd:
                break
            for a in range(-r, r + 1):
                for (i, j) in ((ci + a, cj - r), (ci + a, cj + r), (ci - r, cj + a), (ci + r, cj + a)):
                    if 0 <= i < nnx and 0 <= j < nny and nav[j * nnx + i] == 1:
                        d = (i - ci) * (i - ci) + (j - cj) * (j - cj)
                        if d < bd:
                            bd = d
                            best = j * nnx + i
        return best

    def los(self, ax, ay, az, bx, by, bz, step=2.0):
        """3D line of sight over the surface grid (buildings + terrain). Trees ignored."""
        dx = bx - ax
        dy = by - ay
        d = math.hypot(dx, dy)
        n = int(d / step)
        if n < 3:
            return d < 12.0 or step <= 2.0 or self.los(ax, ay, az, bx, by, bz, 2.0)
        dz = bz - az
        surf = self.surf
        snx, sny, minX, minY = self.snx, self.sny, self.minX, self.minY
        inv = 1.0 / n
        for k in range(2 if step <= 2.0 else 1, n - 1 if step <= 2.0 else n):
            t = k * inv
            i = int((ax + dx * t - minX) * 0.5)
            j = int((ay + dy * t - minY) * 0.5)
            if i < 0 or j < 0 or i >= snx or j >= sny:
                continue
            if surf[j * snx + i] * 0.1 > az + dz * t + 0.5:
                return False
        return True

    def walk_clear(self, ax, ay, bx, by):
        dx = bx - ax
        dy = by - ay
        d = math.hypot(dx, dy)
        n = int(d / 1.5) + 1
        nav = self.nav
        nnx, nny, minX, minY = self.nnx, self.nny, self.minX, self.minY
        inv = 1.0 / n
        k0 = 1
        s = 1.0 / self.nc
        for k in range(k0, n + 1):
            t = k * inv
            i = int((ax + dx * t - minX) * s)
            j = int((ay + dy * t - minY) * s)
            if i < 0 or j < 0 or i >= nnx or j >= nny or nav[j * nnx + i] != 1:
                return False
        return True


class Search:
    """Time-sliced weighted A* on the nav grid."""
    W = 1.6
    MAXEXP = 50000

    def __init__(self, nav, start, goal):
        self.nav = nav
        self.start = start
        self.goal = goal
        self.g = {start: 0.0}
        self.parent = {start: -1}
        self.open = [(0.0, start)]
        self.closed = set()
        self.done = False
        self.path = None
        self.exp = 0
        self.best = start
        self.besth = 1e18

    def run(self, deadline):
        nav = self.nav
        navb = nav.nav
        nnx, nny = nav.nnx, nav.nny
        goal = self.goal
        gi, gj = goal % nnx, goal // nnx
        g = self.g
        parent = self.parent
        op = self.open
        closed = self.closed
        W = self.W
        pop = heapq.heappop
        push = heapq.heappush
        cnt = 0
        while op:
            cnt += 1
            if (cnt & 63) == 0 and time.perf_counter() > deadline:
                self.exp += cnt
                if self.exp > self.MAXEXP:
                    self._finish(self.best)
                return
            f, c = pop(op)
            if c in closed:
                continue
            closed.add(c)
            if c == goal:
                self._finish(c)
                return
            ci, cj = c % nnx, c // nnx
            gc = g[c]
            for di, dj, cost in NEI:
                i = ci + di
                j = cj + dj
                if i < 0 or j < 0 or i >= nnx or j >= nny:
                    continue
                d = j * nnx + i
                if navb[d] != 1 or d in closed:
                    continue
                if di and dj and (navb[cj * nnx + i] != 1 or navb[j * nnx + ci] != 1):
                    continue
                ng = gc + cost
                og = g.get(d)
                if og is None or ng < og:
                    g[d] = ng
                    parent[d] = c
                    ax = i - gi if i > gi else gi - i
                    ay = j - gj if j > gj else gj - j
                    h = (ax + ay) - 0.586 * (ax if ax < ay else ay)
                    if h < self.besth:
                        self.besth = h
                        self.best = d
                    push(op, (ng + W * h, d))
        self._finish(self.best)

    def _finish(self, c):
        p = []
        n = 0
        while c != -1 and n < 5000:
            p.append(c)
            c = self.parent.get(c, -1)
            n += 1
        p.reverse()
        self.path = [self.nav.center(c) for c in p]
        self.done = True


NEI = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))

UPGRADE_PRIO = [
    "plating", "armor", "nanites", "lightning_efficiency", "laser_damage", "servos", "laser_charge",
    "lightning_range", "lightning_chain", "gun_damage", "gun_rate", "reactor", "capacitor", "gyros",
    "gun_mag", "gun_velocity", "jets", "laser_pierce", "sensors", "mortar_reload", "mortar_splash", "mortar_velocity",
]
UPGRADE_DMG = [
    "laser_damage", "lightning_efficiency", "gun_damage", "gun_rate", "laser_charge", "lightning_chain", "lightning_range",
    "plating", "armor", "nanites", "servos", "gun_mag", "gun_velocity", "reactor", "capacitor", "gyros", "laser_pierce",
    "jets", "sensors", "mortar_reload", "mortar_splash", "mortar_velocity",
]
WRANK_GZ = {"lightning": 4, "gun": 3, "laser": 2, "mortar": 1}
WRANK_GL = {"laser": 4, "gun": 3, "lightning": 2, "mortar": 1}
WRANK = {"lightning": 4, "laser": 3, "gun": 2, "mortar": 1}


# ----------------------------------------------------------------------------------------------
# Bot
# ----------------------------------------------------------------------------------------------
class Bot:
    name = "kestrel"

    def __init__(self, hello):
        self.cfg = hello.get("config", {}) or {}
        self.slot = hello.get("slot", 0)
        self.rng = random.Random(hello.get("seed", 1))
        self.nav = Nav(os.path.join(HERE, "nav.bin"))
        loot = self.cfg.get("loot", {})
        self.take_r = float(loot.get("pickup_radius", 4)) - 0.5
        self.wcfg = self.cfg.get("weapons", {})
        self.mortar_g = float(self.wcfg.get("mortar", {}).get("gravity", 22))
        # state
        self.items = {}          # id -> {type, kind, pos, seen}
        self.black = {}          # id -> until t
        self.path = None
        self.path_goal = None
        self.path_i = 0
        self.path_fail = 0
        self.search = None
        self.hist = []           # (t, x, y, wanted_move)
        self.unstuck_until = -1.0
        self.unstuck_dir = (0.0, 0.0)
        self.unstuck_mode = 0
        self.unstuck_n = 0
        self.strafe = 1.0
        self.strafe_until = 0.0
        self.retreat = False
        self.cover = None
        self.cover_t = -9.0
        self.loot_goal = None
        self.loot_since = 0.0
        self.roam = None
        self.roam_t = -99.0
        self.last_dmg_t = -99.0
        self.last_dmg_bearing = None
        self.last_dmg_by = None
        self.last_seen_enemy_t = -99.0
        self.scan_yaw = 0.0
        self.zone_mode = False
        self.unstuck_t0 = -9.0
        self.nav_arrived = False
        self.F = {"roof": 0, "retreat": 0, "r1": 0.45, "r2": 0.7, "r3": 0.85, "rexit": 0.85, "selective": 0, "sel_near": 45.0, "zaprespect": 0, "kite": 0, "peek": 0, "avoid": 0, "farm_until": 0, "chase": 1, "focus": 1, "dmgup": 1, "cubegrab": 30, "gl": 1, "rearpref": 0, "drops": 1, "econ_t": 0, "near_r": 100, "roofprio": 0, "kiter": 0, "bystander_first": 0, "econ_cubes": 10, "econ_cap": 170, "econ_nokill": 0, "precharge": 0, "boostfight": 0, "passive_t": 0, "kill_cap": 0, "passive_len": 0}
        self.F.update({"mode": "auto", "open_s": 40.0, "third": 0.0})
        self.env_flags = {}
        try:
            self.env_flags = json.loads(os.environ.get("MECH_FLAGS", "{}"))
        except Exception:
            self.env_flags = {}
        self.base_F = dict(self.F)
        self.mode = "squad"
        self.base_sel = self.F.get("selective", 0)
        self.base_chase = self.F.get("chase", 0)
        self.kills = 0
        self.alive0 = None
        self.ramped = False
        self.probe_passive = False
        self.probe_used = False
        self.probe_t0 = 0.0
        self.econ = False
        self.econ_done = False
        self.npc_mem = {}
        self.hp_prev = {}
        self.hp_prev_tick = -99
        self.myhits_prev = set()
        self.foreign_hits = 0
        self.F.update(self.env_flags)
        if self.F.get("mode") == "auto":
            self.set_mode("open")
        elif self.F.get("mode") == "ffa":
            self.set_mode("ffa")
        self.pk_cands = None
        self.pk_i = 0
        self.pk_best = None
        self.pk_bs = -1e18
        self.pk_t = -99.0
        self.pk_spot = None
        self.pk_score = 0.0
        self.den = None
        self.den_t = -99.0
        self.den_cands = None
        self.den_i = 0
        self.den_best = None
        self.den_bs = -1e18
        self.cur_target = None
        self.hostile = {}
        self.dist_hist = {}
        self.bystanders = []
        self.roofs = []
        try:
            with open(os.path.join(HERE, "roofs.json")) as f:
                self.roofs = json.load(f)
        except Exception:
            self.roofs = []
        self.roof = None          # index of the home roof
        self.roof_t = -99.0
        self.roof_bad = {}
        self.climb = None
        self.climb_fails = 0
        self.onroof = False
        self.rp_cands = None
        self.rp_i = 0
        self.rp_best = None
        self.rp_bs = -1e18
        self.rp_t = -99.0
        self.rp_target = None
        self.mem_threats = []
        self.cov_cands = None
        self.cov_i = 0
        self.cov_best = None
        self.cov_bs = -1e18
        self.bullet_flip = False
        self.flip_ok_t = 0.0
        self.mem_threats_t = -99.0
        self._shoot = None
        self.deadline = 0.0

    # ------------------------------------------------------------------ modes
    PROFILES = {
        "squad": {},
        # cautious opening used until we know which kind of match this is
        "open": {"retreat": 1, "selective": 1, "peek": 1, "zaprespect": 1, "chase": 0},
        # free-for-all: outlive the others
        "ffa": {"retreat": 1, "r1": 0.6, "r2": 0.8, "r3": 0.9, "rexit": 0.9, "selective": 1, "peek": 1, "zaprespect": 2,
                "chase": 0, "avoid": 2, "dmgup": 0, "third": 0.35, "roof": 1, "gl": 0, "drops": 0, "econ_cubes": 0, "near_r": 180},
    }

    def set_mode(self, name):
        f = dict(self.base_F)
        prof = name
        if name == "open":
            prof = self.env_flags.get("open_prof", self.base_F.get("open_prof", "open"))
        f.update(self.PROFILES.get(prof, {}))
        f.update(self.env_flags)
        self.F = f
        self.mode = name
        self.base_sel = f.get("selective", 0)
        self.base_chase = f.get("chase", 0)

    # ------------------------------------------------------------------ helpers
    def w_slot(self, me, kind):
        ws = me.get("weapons") or []
        for i, w in enumerate(ws):
            if w and w.get("kind") == kind:
                return i, w
        return None, None

    def want_pickup(self, me, kind):
        """Return (want, drop_slot or None)."""
        ws = (me.get("weapons") or [None, None]) + [None, None]
        ws = ws[:2]
        kinds = [w.get("kind") if w else None for w in ws]
        if kind in kinds:
            return False, None
        for i in (1, 0):
            if kinds[i] is None:
                return True, None
        WR = WRANK_GZ if self.F.get("gz", 0) else (WRANK_GL if self.F["gl"] else WRANK)
        r = WR.get(kind, 0)
        lo = 0 if WR.get(kinds[0], 0) <= WR.get(kinds[1], 0) else 1
        if r > WR.get(kinds[lo], 0):
            w = ws[lo]
            if w.get("kind") == "laser" and (w.get("charge") or 0) > 0:
                return False, None
            return True, lo
        return False, None

    # ------------------------------------------------------------------ navigation
    def nav_move(self, x, y, gx, gy):
        """Unit direction to walk toward (gx, gy) around obstacles."""
        nav = self.nav
        if not nav.ok:
            return norm2(gx - x, gy - y)
        if nav.val(x, y) != 1:
            c = nav.nearest_main(x, y, 36)
            if c is None:
                return norm2(gx - x, gy - y)
            cx, cy = nav.center(c)
            # blend with goal direction a little so we leave on the right side
            ux, uy = norm2(cx - x, cy - y)
            return ux, uy
        gx = min(max(gx, nav.minX + 6.0), nav.minX + nav.nnx * nav.nc - 6.0)
        gy = min(max(gy, nav.minY + 6.0), nav.minY + nav.nny * nav.nc - 6.0)
        gc = nav.nearest_main(gx, gy, 36)
        if gc is None:
            return norm2(gx - x, gy - y)
        if nav.val(gx, gy) != 1:
            gx, gy = nav.center(gc)
            if math.hypot(gx - x, gy - y) < 2.0:
                self.nav_arrived = True
                return 0.0, 0.0
        if nav.walk_clear(x, y, gx, gy):
            self.path = None
            self.search = None
            return norm2(gx - x, gy - y)
        nnx = nav.nnx
        if self.path is not None and self.path_goal is not None:
            pg = self.path_goal
            if abs(pg % nnx - gc % nnx) > 3 or abs(pg // nnx - gc // nnx) > 3:
                # goal moved: plan again but keep following the old path meanwhile
                if self.search is None or abs(self.search.goal % nnx - gc % nnx) > 3 or abs(self.search.goal // nnx - gc // nnx) > 3:
                    self.search = Search(nav, nav.cell(x, y), gc)
            elif self.search is not None:
                self.search = None
        elif self.search is None or abs(self.search.goal % nnx - gc % nnx) > 3 or abs(self.search.goal // nnx - gc // nnx) > 3:
            self.search = Search(nav, nav.cell(x, y), gc)
        if self.search is not None:
            self.search.run(min(self.deadline, time.perf_counter() + 0.003))
            if self.search.done:
                self.path = self.search.path
                self.path_goal = self.search.goal
                self.path_i = 0
                self.path_fail = 0
                self.search = None
        p = self.path
        if p:
            n = len(p)
            hi = min(n - 1, self.path_i + 10)
            found = -1
            for j in range(hi, self.path_i - 1, -1):
                px, py = p[j]
                if nav.walk_clear(x, y, px, py):
                    found = j
                    break
            if found < 0:
                self.path_fail += 1
                if self.path_fail > 12:
                    self.path = None
                    return norm2(gx - x, gy - y)
                px, py = p[min(self.path_i, n - 1)]
                return norm2(px - x, py - y)
            self.path_fail = 0
            self.path_i = found
            px, py = p[found]
            if found == n - 1 and math.hypot(px - x, py - y) < 2.5:
                self.path = None
                return norm2(gx - x, gy - y)
            return norm2(px - x, py - y)
        return norm2(gx - x, gy - y)

    def step_ok(self, x, y, dx, dy, dist=6.0):
        if not self.nav.ok:
            return True
        if self.onroof:
            return self.nav.val(x + dx * dist, y + dy * dist) == 2 and self.nav.val(x + dx * dist * 0.5, y + dy * dist * 0.5) == 2
        return self.nav.val(x + dx * dist, y + dy * dist) == 1 and self.nav.val(x + dx * dist * 0.5, y + dy * dist * 0.5) != 0

    # ------------------------------------------------------------------ aiming
    def aim_at(self, me, tx, ty, tz):
        x, y, z = me["pos"]
        mz = z + float(me.get("stats", {}).get("muzzle_height", 7))
        dx, dy, dz = tx - x, ty - y, tz - mz
        d2 = math.hypot(dx, dy)
        return math.atan2(dy, dx), math.atan2(dz, max(d2, 0.01)), math.sqrt(d2 * d2 + dz * dz)

    def lead(self, me, e, speed, extra=0.08):
        """Predicted centre of e when a projectile of `speed` arrives."""
        x, y, z = me["pos"]
        px, py, pz = e["pos"]
        vx, vy, vz = e.get("vel", [0, 0, 0])
        h = 1.75 if e.get("kind") == "npc" else 4.5
        t = extra
        for _ in range(3):
            ex = px + vx * t
            ey = py + vy * t
            d = math.sqrt((ex - x) ** 2 + (ey - y) ** 2 + (pz + h - z - 7) ** 2)
            t = extra + d / speed
        return px + vx * t, py + vy * t, pz + h + (vz * t * 0.5 if abs(vz) > 0.5 else 0.0)

    def mortar_solution(self, me, tx, ty, tz, speed):
        x, y, z = me["pos"]
        h0 = z + float(me.get("stats", {}).get("height", 9)) + 0.5
        d = math.hypot(tx - x, ty - y)
        h = tz - h0
        g = self.mortar_g
        v2 = speed * speed
        disc = v2 * v2 - g * (g * d * d + 2 * h * v2)
        if disc < 0 or d < 1:
            return None
        th = math.atan2(v2 - math.sqrt(disc), g * d)
        tf = d / max(1e-3, speed * math.cos(th))
        return math.atan2(ty - y, tx - x), th, tf

    # ------------------------------------------------------------------ main
    def act(self, obs):
        t0 = time.perf_counter()
        try:
            alive_now = obs.get("alive", 16)
            if self.alive0 is None:
                self.alive0 = alive_now
            for ev in obs.get("events") or []:
                if ev.get("type") == "kill":
                    self.kills += 1
            # mechs losing a weapon-sized chunk of HP that we did not cause = others are fighting each other
            if self.F.get("fh", 0) and self.mode != "ffa":
                tk = obs.get("tick", 0)
                myhits = set()
                for ev in obs.get("events") or []:
                    if ev.get("type") == "hit":
                        myhits.add(ev.get("target"))
                prev = self.hp_prev
                cur = {}
                for m_ in obs.get("mechs") or []:
                    if m_.get("kind") != "mech" or "hp" not in m_:
                        continue
                    mid_ = m_.get("id")
                    cur[mid_] = m_["hp"]
                    if mid_ in myhits or mid_ in self.myhits_prev or "mortar" in (m_.get("weapons") or []):
                        continue
                    ph_ = prev.get(mid_)
                    if ph_ is not None and tk - self.hp_prev_tick <= 2 and ph_ - m_["hp"] >= 44.0:
                        self.foreign_hits += 1
                self.hp_prev = cur
                self.hp_prev_tick = tk
                self.myhits_prev = myhits
            if self.F.get("mode") == "auto":
                tt = obs.get("t", 0.0)
                unexplained = (self.alive0 - alive_now) - self.kills
                if self.mode != "ffa" and self.foreign_hits >= self.F.get("fh_n", 3):
                    self.set_mode("ffa")
                if self.mode != "ffa" and (unexplained >= 2 or (unexplained >= 1 and tt < 60.0)):
                    self.set_mode("ffa")   # mechs are dying that we did not kill: others fight each other
                elif self.mode == "open" and tt >= self.F.get("open_s", 40.0):
                    self.set_mode("squad")
            self.econ = False
            if (self.F.get("econ_t", 0) or self.F.get("econ_cubes", 0)) and self.mode != "ffa":
                # economy phase: only fight what attacks us and farm drones/crates/cubes; ends at a cube count or a time cap
                tt_ = obs.get("t", 0.0)
                if self.F.get("econ_cubes", 0):
                    early = (obs["self"].get("cubes", 0) < self.F["econ_cubes"]) and tt_ < self.F.get("econ_cap", 170.0) and not self.econ_done
                    if not early:
                        self.econ_done = True
                else:
                    early = tt_ < self.F["econ_t"]
                self.econ = bool(early)
                self.F["selective"] = 1 if early else self.base_sel
                self.F["chase"] = 0 if early else self.base_chase
                if self.F.get("econ_cubes", 0):
                    self.F["drops"] = 1 if early else 2
            if self.mode == "ffa" and self.F.get("ffa_econ", 0) and not self.ramped:
                # free-for-all economy: value drones/crates highly whenever it is safe to loot at all
                self.econ = obs["self"].get("cubes", 0) < self.F["ffa_econ"]
            # measurement probes: go fully passive (never shoot, hide) after a time or a kill count
            if self.mode != "ffa" and not self.probe_passive:
                pt_ = self.F.get("passive_t", 0)
                kc_ = self.F.get("kill_cap", 0)
                if not self.probe_used and ((pt_ and obs.get("t", 0.0) >= pt_) or (kc_ and self.kills >= kc_)):
                    self.probe_passive = True
                    self.probe_used = True
                    self.probe_t0 = obs.get("t", 0.0)
            elif self.probe_passive and self.F.get("passive_len", 0) and obs.get("t", 0.0) - self.probe_t0 >= self.F["passive_len"]:
                # pause over: go back to the normal squad play
                self.probe_passive = False
                self.set_mode(self.mode)
                self.F["econ_cubes"] = 0
            if self.probe_passive and self.mode != "ffa":
                self.econ = False
                self.F["avoid"] = 1
                self.F["selective"] = 1
                self.F["chase"] = 0
                self.F["retreat"] = 1
                self.F["drops"] = 0
            if self.mode == "ffa" and not self.ramped and self.F.get("ramp_t", 0):
                # few deaths this late = a lobby of hiders: hiding no longer gains places, so start hunting
                kr = self.F.get("kill_ramp", 0)
                if (kr and self.kills >= kr) or (not kr and obs.get("t", 0.0) >= self.F["ramp_t"] and alive_now >= self.F.get("ramp_alive", 9)):
                    self.ramped = True
                    self.F.update({"selective": 0, "avoid": 0, "chase": 1, "roof": 0, "gl": 1, "dmgup": 1, "peek": 0,
                                   "zaprespect": 0, "r1": 0.45, "r2": 0.7, "r3": 0.85, "rexit": 0.85, "third": 0.0})
                    self.F.update(self.F.get("ramp_prof", {}) or {})
            if self.mode == "ffa" and self.F.get("mode") != "squad" and not self.ramped:
                end = alive_now <= self.F.get("end_alive", 3)
                self.F["avoid"] = 0 if end else self.PROFILES["ffa"]["avoid"] if "avoid" not in self.env_flags else self.env_flags["avoid"]
                self.F["chase"] = 1 if end else 0
        except Exception:
            pass
        F = self.F
        self.deadline = t0 + 0.006
        nav = self.nav
        me = obs["self"]
        t = obs.get("t", 0.0)
        x, y, z = me["pos"]
        vx, vy, vz = me.get("vel", [0, 0, 0])
        st = me.get("stats", {}) or {}
        hp = me.get("hp", 1)
        mhp = max(1.0, me.get("max_hp", 1400))
        hpf = hp / mhp
        energy = me.get("energy", 0)
        walk = float(st.get("walk_speed", 11.7))
        eye = float(st.get("eye_height", 8))
        aim_yaw, aim_pitch = me.get("aim", [0, 0])
        act = {"tick": obs["tick"]}
        move = (0.0, 0.0)
        boost = False
        jump = False
        fire = [False, False]

        # ---- upgrades
        offer = obs.get("offer")
        if offer and offer.get("choices"):
            best_i, best_r = 0, 999
            for i, ch in enumerate(offer["choices"]):
                cid = ch.get("id")
                prio = UPGRADE_DMG if F["dmgup"] else UPGRADE_PRIO
                r = prio.index(cid) if cid in prio else 500
                if r < best_r:
                    best_i, best_r = i, r
            act["pick"] = best_i

        # ---- events
        for ev in obs.get("events") or []:
            ty = ev.get("type")
            if ty == "damaged" and ev.get("by") is not None and ev.get("by") < 100:
                self.hostile[ev.get("by")] = t + 20.0
            if ty == "damaged" and ev.get("by") is not None:
                self.last_dmg_t = t
                self.last_dmg_bearing = ev.get("bearing")
                self.last_dmg_by = ev.get("by")
            elif ty in ("took", "take_failed"):
                self.items.pop(ev.get("id"), None)
                if ty == "take_failed" and ev.get("reason") in ("have", "full", "busy"):
                    self.black[ev.get("id")] = t + 20

        # ---- loot memory
        items = self.items
        seen = set()
        for typ, key in (("pickup", "pickups"), ("cube", "cubes"), ("crate", "crates")):
            for it in obs.get(key) or []:
                iid = it["id"]
                seen.add(iid)
                items[iid] = {"type": typ, "kind": it.get("kind"), "pos": it["pos"], "seen": t, "id": iid}
        if items:
            dead = []
            for iid, it in items.items():
                if iid in seen:
                    continue
                p = it["pos"]
                d = math.hypot(p[0] - x, p[1] - y)
                if d < 30 and (not nav.ok or nav.los(x, y, z + eye, p[0], p[1], p[2] + 1.0)):
                    dead.append(iid)
                elif t - it["seen"] > 150:
                    dead.append(iid)
            for iid in dead:
                items.pop(iid, None)

        # ---- zone
        zn = obs.get("zone") or {}
        zc = zn.get("center", [0, 0])
        zr = float(zn.get("radius", 1000))
        ncx, ncy = zn.get("next_center", zc)
        nr = float(zn.get("next_radius", zr))
        ss = float(zn.get("shrink_start", 1e9))
        se = float(zn.get("shrink_end", 1e9))
        dzc = math.hypot(x - zc[0], y - zc[1])
        dnc = math.hypot(x - ncx, y - ncy)
        dur = max(1.0, se - max(ss, t)) if t < se else 1.0
        edge_speed = ((zr - nr) + math.hypot(zc[0] - ncx, zc[1] - ncy)) / dur if zr > nr else 0.0
        look = 14.0
        if t < ss - look:
            margin = 22.0
        else:
            margin = 22.0 + min(look, max(0.0, t + look - ss)) * edge_speed
        margin = min(margin, zr * 0.9)
        edge_gap = zr - dzc  # >0 inside
        in_zone = me.get("in_zone", True)
        if (not in_zone) or edge_gap < margin:
            self.zone_mode = True
        elif edge_gap > margin + 18 or dnc < nr * 0.7:
            self.zone_mode = False
        if nr <= 1.0 and t > ss - 20:
            self.zone_mode = True
        zone_urgent = (not in_zone) or edge_gap < 6 + 4 * edge_speed

        def zone_ok(px, py, extra=0.0):
            return math.hypot(px - zc[0], py - zc[1]) < zr - margin - 10 - extra

        # ---- enemies
        mechs = []
        npcs = []
        for m in obs.get("mechs") or []:
            if m.get("kind") == "npc":
                npcs.append(m)
            else:
                mechs.append(m)
        threats = []
        for m in mechs:
            p = m["pos"]
            d = math.sqrt((p[0] - x) ** 2 + (p[1] - y) ** 2 + (p[2] - z) ** 2)
            threats.append((d, m, False))
        threats.sort(key=lambda a: a[0])
        for k in range(len(threats)):
            d, m, _ = threats[k]
            p = m["pos"]
            if k < 5 and d < 260:
                threats[k] = (d, m, (not nav.ok) or nav.los(x, y, z + eye, p[0], p[1], p[2] + 6.0, 2.0 if d < 120 else 3.0))
        if threats:
            self.last_seen_enemy_t = t
        recently_hit = t - self.last_dmg_t < 3.0
        all_threats = threats
        if F["selective"]:
            # only mechs that act hostile (shot us, or close in while aiming at us) are fought
            hs = self.hostile
            dh = self.dist_hist
            keep = []
            for d, m, los in threats:
                mid = m.get("id")
                prev = dh.get(mid)
                if prev is None or t - prev[0] >= 1.0:
                    if prev is not None and t - prev[0] < 2.5 and los:
                        closing = (prev[1] - d) / max(0.1, t - prev[0])
                        p = m["pos"]
                        aim_at_me = abs(angdiff(math.atan2(y - p[1], x - p[0]), (m.get("aim") or [0, 0])[0])) < 0.3
                        if closing > 4.0 and aim_at_me and d < 170:
                            hs[mid] = t + 15.0
                    dh[mid] = (t, d)
                if d < F["sel_near"] and los:
                    hs[mid] = max(hs.get(mid, 0), t + 5.0)
                if m.get("charging") and los:
                    p = m["pos"]
                    if abs(angdiff(math.atan2(y - p[1], x - p[0]), (m.get("aim") or [0, 0])[0])) < 0.15:
                        hs[mid] = t + 15.0
                th3 = F["third"] if t < F.get("third_t", 1e9) else F.get("third_late", F["third"])
                if th3 and los and d < 110 and hpf > 0.6 and m.get("hp", 9999) < th3 * max(1.0, m.get("max_hp", 1400)) and m.get("hp", 9999) < hp:
                    hs[mid] = max(hs.get(mid, 0), t + 6.0)   # weakened mech in reach: finish it for the cubes
                if hs.get(mid, 0) > t:
                    keep.append((d, m, los))
            self.bystanders = [a_ for a_ in threats if hs.get(a_[1].get("id"), 0) <= t]
            threats = keep

        _, gun = self.w_slot(me, "gun")
        gi, _ = self.w_slot(me, "gun")
        li, laser = self.w_slot(me, "laser")
        zi, zap = self.w_slot(me, "lightning")
        mi, mortar = self.w_slot(me, "mortar")
        zap_range = float(zap["stats"].get("range", 45)) if zap else 0.0

        # target: prefer visible, close, weak
        target = None
        tgt_d = 1e9
        tgt_los = False
        bests = 1e18
        for d, m, los in threats:
            if F["focus"]:
                s = 0.5 * d + 0.09 * m.get("hp", 1000) + (0 if los else 60)
                if m.get("id") == self.cur_target:
                    s -= 15
                if F.get("bystander_first", 0) and los and self.hostile.get(m.get("id"), 0) <= t and d < 160:
                    s -= 60   # not hunting us (so probably a stronger tier, still passive): take it first
                if F["rearpref"] and los:
                    p_ = m["pos"]
                    if abs(angdiff(math.atan2(y - p_[1], x - p_[0]), (m.get("aim") or [0, 0])[0])) > 1.7:
                        s -= 30  # its back is to us: x1.5 damage
            else:
                s = d + 40.0 * (m.get("hp", 1000) / max(1.0, m.get("max_hp", 1400))) + (0 if los else 80)
            if m.get("id") == self.last_dmg_by and recently_hit:
                s -= 25
            if s < bests:
                bests = s
                target, tgt_d, tgt_los = m, d, los
        self.cur_target = target.get("id") if target is not None else None
        n_close = sum(1 for d, m, los in threats if los and d < 150)
        self.n_close = n_close

        # ---- retreat decision (earlier when outnumbered; hold cover until healed)
        if threats:
            self.mem_threats = [(d, m, False) for d, m, los in threats[:4]]
            self.mem_threats_t = t
        if target is not None:
            ehp = target.get("hp", 1400)
            if hpf < F["r1"] and not (n_close <= 1 and ehp < hp * 0.6):
                self.retreat = True
            if n_close >= 2 and hpf < F["r2"]:
                self.retreat = True
            if n_close >= 3 and hpf < F["r3"]:
                self.retreat = True
        elif recently_hit and hpf < F["r1"]:
            self.retreat = True
        if not F["retreat"]:
            self.retreat = False
        if self.retreat:
            if hpf > F["rexit"]:
                self.retreat = False
            elif target is not None and n_close <= 1 and target.get("hp", 1400) < hp * 0.45:
                self.retreat = False
            elif not threats and t - self.last_dmg_t > 25 and t - self.mem_threats_t > 25:
                self.retreat = False
        if self.retreat and not threats and t - self.mem_threats_t < 15:
            cover_th = self.mem_threats
        else:
            cover_th = threats

        # ---- engage decision
        engage_r = 150.0 if laser else 100.0
        fight = target is not None and (tgt_d < engage_r or (recently_hit and tgt_d < 220))

        hold_fire = False
        if F["avoid"] == 2:
            # free-for-all caution: fight only what is hostile and close (or weak); otherwise keep out of everyone's sight
            cornered = target is not None and tgt_los and tgt_d < F.get("corner", 75)
            th3_ = F["third"] if t < F.get("third_t", 1e9) else F.get("third_late", F["third"])
            weak = target is not None and target.get("hp", 9999) < max(0.35, th3_) * max(1.0, target.get("max_hp", 1400)) and hpf > 0.5
            arming = False
            if F.get("armfirst", 0) and not (zap or laser or mortar):
                # unarmed: a weapon within reach comes before hiding
                for iid, it in items.items():
                    if it["type"] == "pickup" and self.black.get(iid, 0) <= t:
                        p_ = it["pos"]
                        if math.hypot(p_[0] - x, p_[1] - y) < F["armfirst"] and abs(p_[2] - nav.surf_at(p_[0], p_[1])) < 4.0 and nav.val(p_[0], p_[1]) == 1:
                            arming = True
                            break
            if arming:
                fight = False
                self.retreat = False
            elif not (cornered or weak):
                near = [a_ for a_ in all_threats if a_[0] < F.get("shy", 150) and (a_[2] or not F.get("shylos", 0) or a_[0] < 50)]
                if near:
                    fight = False
                    self.retreat = True
                    cover_th = near[:4]
                    self.mem_threats = [(d_, m_, False) for d_, m_, l_ in near[:4]]
                    self.mem_threats_t = t
                elif hpf > F["rexit"] or (hpf > 0.5 and not recently_hit and t - self.mem_threats_t > 6):
                    self.retreat = False
        elif F["avoid"]:
            # pure evasion: never fight, hide from every mech we know about
            fight = False
            hold_fire = True
            if all_threats:
                self.retreat = True
                cover_th = all_threats[:4]
                self.mem_threats = [(d_, m_, False) for d_, m_, l_ in all_threats[:4]]
                self.mem_threats_t = t
            elif t - self.mem_threats_t > 6 and not recently_hit:
                self.retreat = False
        elif self.econ and F.get("econ_nokill", 0) and hpf > 0.5:
            # farming with one hunter on us: do not thin the squad yet. Shoot back only while it is hurting us,
            # and stop shooting when it is nearly dead (a dead hunter brings more hunters).
            fight = False
            self.retreat = False
            thp = (target.get("hp", 1400) / max(1.0, target.get("max_hp", 1400))) if target is not None else 1.0
            hold_fire = (not recently_hit) or thp < 0.3
        elif F["farm_until"] and t < F["farm_until"] and hpf > 0.5:
            # power up first: do not fight (or kill) the early hunter, keep looting
            fight = False
            self.retreat = False
            hold_fire = True
        # ---- strafing clock
        if t >= self.strafe_until:
            self.strafe = -self.strafe if self.rng.random() < 0.75 else self.strafe
            self.strafe_until = t + 0.45 + self.rng.random() * 0.9

        # inbound bullets aimed at where we are heading -> change direction
        self.bullet_flip = False
        for pr in obs.get("projectiles") or []:
            if pr.get("kind") != "gun":
                continue
            p = pr["pos"]
            v = pr["vel"]
            sp2 = v[0] * v[0] + v[1] * v[1]
            if sp2 < 1.0:
                continue
            tc = ((x - p[0]) * v[0] + (y - p[1]) * v[1]) / sp2
            if tc <= 0.0 or tc > 0.45:
                continue
            fx, fy = x + vx * tc, y + vy * tc
            ox, oy = fx - (p[0] + v[0] * tc), fy - (p[1] + v[1] * tc)
            if ox * ox + oy * oy < 12.0 and abs(p[2] + v[2] * tc - (z + 4.5)) < 7.0:
                self.bullet_flip = True
                break

        label = "idle"
        arrived_prev = self.nav_arrived
        self.nav_arrived = False
        # ================= movement =================
        # ---- home roof
        if not F["roof"] or t < F.get("roof_t", 0):
            self.roof = None
        elif t - self.roof_t > 1.0:
            self.roof_t = t
            self.pick_roof(t, x, y, zc, zr, ncx, ncy, nr, margin)
        rf = self.roofs[self.roof] if self.roof is not None else None
        self.onroof = bool(rf is not None and me.get("grounded", True) and z > rf["z"] - 3.5 and abs(z - nav.surf_at(x, y)) < 1.6
                           and math.hypot(x - rf["camp"][0], y - rf["camp"][1]) < rf["rad"] + 8)
        if rf is None:
            self.climb = None
        no_unstuck = self.onroof or self.climb is not None
        same_level = target is not None and self.onroof and abs(target["pos"][2] - z) < 5.0 and tgt_d < 70
        contact = bool(threats) or recently_hit or t - self.last_seen_enemy_t < 8.0
        zone_need = self.zone_mode and (dnc > nr * 0.55 or nr <= 1.0 or not in_zone)
        # free-for-all opening: put distance between us and every mech that can see us
        flee_mv = None
        if self.mode == "ffa" and t < F.get("flee_t", 0) and not same_level:
            ax = ay = 0.0
            for d_, m_, l_ in all_threats:
                if l_ and d_ < 220:
                    p_ = m_["pos"]
                    w_ = 1.0 / max(20.0, d_)
                    ux_, uy_ = norm2(x - p_[0], y - p_[1])
                    ax += ux_ * w_
                    ay += uy_ * w_
            if ax or ay:
                ax, ay = norm2(ax, ay)
                bsc = -1e9
                for k in range(16):
                    a_ = k * TAU / 16
                    cx_, cy_ = math.cos(a_), math.sin(a_)
                    sc_ = cx_ * ax + cy_ * ay
                    if not self.step_ok(x, y, cx_, cy_, 7.0):
                        sc_ -= 1.5
                    elif nav.ok and nav.val(x + cx_ * 16.0, y + cy_ * 16.0) != 1:
                        sc_ -= 0.6
                    if not zone_ok(x + cx_ * 25.0, y + cy_ * 25.0, -10):
                        sc_ -= 3.0
                    if sc_ > bsc:
                        bsc = sc_
                        flee_mv = (cx_, cy_)
        if nr <= 1.0 and in_zone and dnc < max(3.0, zr * 0.3) and (fight or self.retreat) and edge_gap > 5:
            zone_need = False
        fmv = None
        if self.onroof and not same_level:
            fight = False
        if fight and not (zone_need and zone_urgent):
            fmv = self.fight_move(t, me, target, tgt_d, tgt_los, zap, laser, energy, zone_ok)
            if F["peek"] and len(threats) >= 2 and fmv is not None:
                pr = 24.0 if (zap and "lightning" not in (target.get("weapons") or [])) else (70.0 if laser else 40.0)
                pm = self.peek_move(t, me, threats, target, pr, zone_ok, energy)
                if pm is not None:
                    fmv = (pm, False)
            if fmv is None:
                fight = False
        if zone_need and (zone_urgent or not fight or edge_gap < margin * 0.6):
            label = "zone"
            if nr <= 1.0:
                gx, gy = ncx, ncy
            else:
                ux, uy = norm2(x - ncx, y - ncy)
                r = min(dnc, nr * 0.55)
                gx, gy = ncx + ux * r, ncy + uy * r
            move = self.nav_move(x, y, gx, gy)
            boost = zone_urgent and energy > 5
            if nr <= 1.0 and dnc < 2.0:
                move = (0.0, 0.0)
            if zone_urgent:
                self.climb = None
        elif flee_mv is not None:
            label = "flee"
            move = flee_mv
            boost = energy > 10
            self.climb = None
        elif rf is not None and self.climb is not None:
            label = "climb"
            move, boost, jump = self.roof_move(t, me, cover_th, target, True, energy)
        elif rf is not None and self.onroof and contact and not same_level:
            label = "roof"
            hide = self.retreat or hpf < 0.55 or n_close >= 3
            move, boost, jump = self.roof_move(t, me, cover_th if cover_th else threats, target, hide, energy)
        elif F.get("roofprio", 0) and rf is not None and not self.onroof and self.mode == "squad" and (gun or laser) \
                and not (target is not None and tgt_los and tgt_d < 45):
            # roof sniper: get on the home roof first, fight from up there
            label = "toroof"
            move, boost, jump = self.roof_move(t, me, cover_th if cover_th else threats, target, True, energy)
            no_unstuck = no_unstuck or self.climb is not None
        elif self.retreat and rf is not None and math.hypot(x - rf["camp"][0], y - rf["camp"][1]) < 220:
            label = "toroof"
            move, boost, jump = self.roof_move(t, me, cover_th, target, True, energy)
            no_unstuck = no_unstuck or self.climb is not None
            if self.climb is None and not self.onroof and energy > 80 and any(a_[2] for a_ in threats):
                boost = True
        elif self.retreat:
            label = "retreat"
            move, boost = self.retreat_move(t, x, y, z, cover_th, zone_ok, energy)
        elif fight:
            label = "fight"
            move, boost = fmv
            # under-armed: grab a close weapon first, shooting on the way
            if not (zap or laser) or not (zap and laser):
                bw = None
                bwd = 70.0 if not (zap or laser) else 35.0
                for iid, it in items.items():
                    if it["type"] != "pickup" or self.black.get(iid, 0) > t:
                        continue
                    if it["kind"] == "mortar" and (zap or laser or mortar):
                        continue
                    p = it["pos"]
                    dd = math.hypot(p[0] - x, p[1] - y)
                    if dd < bwd and self.want_pickup(me, it["kind"])[0] and zone_ok(p[0], p[1], -10):
                        if nav.ok and (nav.nearest_main(p[0], p[1], 1) is None or abs(p[2] - nav.surf_at(p[0], p[1])) > 4.0):
                            continue
                        bwd = dd
                        bw = p
                if bw is not None:
                    label = "arm"
                    move = self.nav_move(x, y, bw[0], bw[1]) if bwd > 1.0 else (0.0, 0.0)
                    boost = energy > 40 and bwd > 15
            if F.get("drops", 0) == 1 and label in ("fight", "arm"):
                # a landed (or landing) supply drop is a full heal plus 3 cubes: go and stand in its ring
                bdp = None
                bdd = 220.0
                for dr in obs.get("drops") or []:
                    try:
                        p = dr["pos"]
                        dd = math.hypot(p[0] - x, p[1] - y)
                        if dd < bdd and float(dr.get("lands_in", 0) or 0) < dd / 10.0 + 6.0 and zone_ok(p[0], p[1], -8) and (hpf < 0.9 or me.get("cubes", 0) < 10):
                            bdd = dd
                            bdp = p
                    except Exception:
                        continue
                if bdp is not None:
                    label = "drop"
                    if bdd > 3.5:
                        move = self.nav_move(x, y, bdp[0], bdp[1])
                        boost = energy > 50 and bdd > 40
                    else:
                        move = (0.0, 0.0)
                        boost = False
            if F["cubegrab"] and label == "fight":
                # scoop power cubes mid-fight: each is +4% damage and HP on the spot
                bc = None
                bcd = float(F["cubegrab"])
                for iid, it in items.items():
                    if it["type"] != "cube" or self.black.get(iid, 0) > t:
                        continue
                    p = it["pos"]
                    dd = math.hypot(p[0] - x, p[1] - y)
                    if dd < bcd and abs(p[2] - z) < 6.0 and zone_ok(p[0], p[1], -10) and (not nav.ok or nav.nearest_main(p[0], p[1], 1) is not None):
                        bcd = dd
                        bc = p
                if bc is not None:
                    label = "cube"
                    move = self.nav_move(x, y, bc[0], bc[1])
                    boost = False
        else:
            mv = None
            if not (self.onroof and contact) and not F.get("noloot", 0):
                mv = self.loot_move(t, me, obs, npcs, hpf, zone_ok, threats)
            if mv is not None:
                label = "loot"
                move = mv
            elif rf is not None and F["roof"] != 2:
                label = "nest"
                move, boost, jump = self.roof_move(t, me, cover_th, target, True, energy)
                no_unstuck = no_unstuck or self.climb is not None
            else:
                label = "roam"
                if F.get("den", 0) and nav.ok:
                    label = "den"
                    dm = self.den_move(t, x, y, z, ncx, ncy, nr, zone_ok)
                    if dm is not None:
                        move = dm
                        self.roam = None
                if label == "den" and self.roam is None:
                    pass
                elif self.roam is None or arrived_prev or t - self.roam_t > 20 or math.hypot(self.roam[0] - x, self.roam[1] - y) < 8 or not zone_ok(self.roam[0], self.roam[1]):
                    self.roam = (ncx, ncy)
                    for _ in range(8):
                        a = self.rng.random() * TAU
                        r = max(nr, 10.0) * (0.15 + 0.5 * self.rng.random())
                        rx, ry = ncx + math.cos(a) * r, ncy + math.sin(a) * r
                        if (not nav.ok) or nav.val(rx, ry) == 1:
                            self.roam = (rx, ry)
                            break
                    self.roam_t = t
                if self.roam is not None:
                    move = self.nav_move(x, y, self.roam[0], self.roam[1])

        # ---- dodge mortar shells
        for pr in obs.get("projectiles") or []:
            if pr.get("kind") != "mortar":
                continue
            p = pr["pos"]
            v = pr["vel"]
            g = self.mortar_g
            dz0 = p[2] - z
            disc = v[2] * v[2] + 2 * g * max(dz0, 0.0)
            tl = (v[2] + math.sqrt(disc)) / g
            if tl > 2.5:
                continue
            lx, ly = p[0] + v[0] * tl, p[1] + v[1] * tl
            dd = math.hypot(lx - x, ly - y)
            if dd < 15:
                if dd > 1.0:
                    ax, ay = (x - lx) / dd, (y - ly) / dd
                else:
                    ax, ay = norm2(-v[1], v[0])
                if not self.step_ok(x, y, ax, ay):
                    ax, ay = norm2(-v[1], v[0])
                move = (ax, ay)
                boost = energy > 10
                label = "dodge"
                break

        # ================= aim & fire =================
        aim = None
        aim_target = target
        loot_tgt = getattr(self, "_shoot", None)  # (pos3, kind) from loot_move
        self._shoot = None
        if hold_fire and aim_target is not None and loot_tgt is not None and (tgt_d > 45 or not tgt_los):
            aim_target = None
        if self.retreat:
            loot_tgt = None
        if aim_target is None and loot_tgt is None and npcs and hpf > 0.3 and not self.retreat:
            bd = 75.0
            for m in npcs:
                p = m["pos"]
                d = math.hypot(p[0] - x, p[1] - y)
                if d < bd and ((not nav.ok) or nav.los(x, y, z + 7, p[0], p[1], p[2] + 1.75)):
                    bd = d
                    loot_tgt = (p, 1.75)
        if aim_target is not None:
            # default aim: gun lead (or plain centre)
            gspeed = float(gun["stats"].get("speed", 252)) if gun else 400.0
            lx, ly, lz = self.lead(me, aim_target, gspeed)
            yaw, pitch, dist = self.aim_at(me, lx, ly, lz)
            cx_, cy_, cz_ = self.lead(me, aim_target, 1e9, 0.1)
            hyaw, hpitch, hdist = self.aim_at(me, cx_, cy_, cz_)
            aim = [yaw, pitch]
            charge = (laser.get("charge") or 0.0) if laser else 0.0
            if laser and (charge > 0 or not gun):
                aim = [hyaw, hpitch]
            shot_los = tgt_los and ((not nav.ok) or nav.los(x, y, z + 7, aim_target["pos"][0], aim_target["pos"][1], aim_target["pos"][2] + 4.5))
            # gun
            if gun and shot_los and gun.get("ammo", 1) > 0 and (gun.get("reload") or 0) <= 0:
                if dist < min(float(gun["stats"].get("range", 230)) * 0.97, 140.0):
                    if abs(angdiff(aim_yaw, yaw)) < max(0.012, 2.4 / dist) and abs(aim_pitch - pitch) < max(0.015, 3.5 / dist):
                        fire[gi] = True
            # lightning
            if zap and shot_los and energy >= float(zap["stats"].get("energy", 3)) + 1:
                cone = math.radians(float(zap["stats"].get("cone_deg", 30))) * 0.42
                if tgt_d < zap_range * 0.96 and abs(angdiff(aim_yaw, hyaw)) < cone:
                    fire[zi] = True
            # laser
            if laser:
                ls = laser["stats"]
                full = float(ls.get("charge", 0.75))
                lrange = float(ls.get("range", 280)) * 0.97
                cd = laser.get("cooldown") or 0.0
                if charge > 0:
                    err = abs(angdiff(aim_yaw, hyaw))
                    perr = abs(aim_pitch - hpitch)
                    good = shot_los and hdist < lrange and err < max(0.008, 1.9 / hdist) and perr < max(0.01, 3.2 / hdist)
                    if charge >= full - 0.06 and good:
                        fire[li] = False  # release
                    elif charge >= full + 1.2 and not shot_los:
                        fire[li] = False
                    else:
                        fire[li] = True
                elif cd <= 0.11 and shot_los and hdist < lrange and not (self.retreat and hpf < 0.2):
                    fire[li] = True
                elif F.get("precharge", 0) and cd <= 0.11 and not shot_los and tgt_d < 90:
                    fire[li] = True   # target is just out of sight: have the beam ready for when it reappears
            # mortar takes over the aim briefly when ready
            if mortar and (mortar.get("cooldown") or 0) <= 0.2 and not (laser and charge > 0):
                d2 = math.hypot(aim_target["pos"][0] - x, aim_target["pos"][1] - y)
                mspeed = float(mortar["stats"].get("speed", 72))
                if 22 < d2 < mspeed * mspeed / self.mortar_g * 0.9 and tgt_los:
                    px, py, pz = aim_target["pos"]
                    ev = aim_target.get("vel", [0, 0, 0])
                    sol = self.mortar_solution(me, px, py, pz + 1.0, mspeed)
                    if sol:
                        tf = sol[2] + 0.1
                        sol = self.mortar_solution(me, px + ev[0] * tf, py + ev[1] * tf, pz + 1.0, mspeed)
                    if sol:
                        aim = [sol[0], sol[1]]
                        if abs(angdiff(aim_yaw, sol[0])) < 0.03 and abs(aim_pitch - sol[1]) < 0.02:
                            fire[mi] = True
        elif loot_tgt is not None:
            p, h = loot_tgt
            yaw, pitch, dist = self.aim_at(me, p[0], p[1], p[2] + h)
            aim = [yaw, pitch]
            clear = (not nav.ok) or nav.los(x, y, z + 7, p[0], p[1], p[2] + h)
            if clear and abs(angdiff(aim_yaw, yaw)) < max(0.01, 1.0 / dist) and abs(aim_pitch - pitch) < max(0.01, 1.0 / dist):
                if gun and gun.get("ammo", 1) > 0 and dist < 200:
                    fire[gi] = True
                if laser and dist < 250:
                    ch = laser.get("charge") or 0.0
                    fire[li] = ch < float(laser["stats"].get("charge", 0.75)) - 0.06 and (ch > 0 or (laser.get("cooldown") or 0) <= 0.11)
                if mortar and (mortar.get("cooldown") or 0) <= 0 and not gun and not laser and 25 < dist < 150:
                    sol = self.mortar_solution(me, p[0], p[1], p[2], float(mortar["stats"].get("speed", 72)))
                    if sol:
                        aim = [sol[0], sol[1]]
                        if abs(angdiff(aim_yaw, sol[0])) < 0.03 and abs(aim_pitch - sol[1]) < 0.02:
                            fire[mi] = True
            if zap and clear and dist < zap_range * 0.9 and abs(angdiff(aim_yaw, yaw)) < 0.2 and energy > 30:
                fire[zi] = True
        else:
            # look toward damage, else sweep
            if recently_hit and self.last_dmg_bearing is not None:
                aim = [self.last_dmg_bearing, 0.0]
            else:
                heard = None
                for s in obs.get("sounds") or []:
                    if s.get("dist") == "near" and s.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "boost", "step", "jet"):
                        heard = s.get("bearing")
                        break
                if heard is not None and self.rng.random() < 0.5:
                    self.scan_yaw = heard
                    aim = [heard, 0.0]
                else:
                    aim = [aim_yaw + 1.2, 0.0]
            # release a held laser charge harmlessly only when it would auto fire anyway
            if laser and (laser.get("charge") or 0) > 0:
                fire[li] = False
        # reload when quiet
        if gun and (gun.get("reload") or 0) <= 0:
            ammo = gun.get("ammo", 99)
            mag = float(gun["stats"].get("mag", 12))
            if ammo <= 0 or (ammo < mag * 0.75 and not (target is not None and tgt_los and tgt_d < 150)):
                act["reload"] = True

        # ---- weapon pickups in reach
        best_take = None
        for iid, it in items.items():
            if it["type"] != "pickup" or iid not in seen:
                continue
            p = it["pos"]
            if math.hypot(p[0] - x, p[1] - y) <= self.take_r and abs(p[2] - z) < 3.6 and self.black.get(iid, 0) < t:
                want, drop = self.want_pickup(me, it["kind"])
                if want and (best_take is None or WRANK.get(it["kind"], 0) > best_take[2]):
                    best_take = (iid, drop, WRANK.get(it["kind"], 0))
        if best_take:
            act["take"] = best_take[0]
            if best_take[1] is not None:
                act["drop"] = best_take[1]
                fire[best_take[1]] = False

        # ---- stuck handling
        mvmag = math.hypot(move[0], move[1])
        self.hist.append((t, x, y, mvmag))
        if len(self.hist) > 14:
            self.hist.pop(0)
        if t < self.unstuck_until:
            if self.unstuck_mode == 0:
                move = self.unstuck_dir if mvmag < 0.1 else move
                boost = (not me.get("grounded", True)) and energy > 3
                if me.get("grounded", True) and t - self.unstuck_t0 < 0.25:
                    jump = True
                elif me.get("grounded", True) and t - self.unstuck_t0 > 0.6:
                    self.unstuck_until = t  # landed
            else:
                move = self.unstuck_dir
        elif no_unstuck:
            self.hist = []
        elif len(self.hist) >= 12 and mvmag > 0.5:
            t1, x1, y1, m1 = self.hist[0]
            trav = 0.0
            for k in range(1, len(self.hist)):
                trav += math.hypot(self.hist[k][1] - self.hist[k - 1][1], self.hist[k][2] - self.hist[k - 1][2])
            if all(h[3] > 0.5 for h in self.hist) and trav < 2.5 and t - t1 >= 1.0:
                self.unstuck_n += 1
                self.unstuck_mode = 0 if (self.unstuck_n % 2 == 1 and energy > 25) else 1
                self.unstuck_t0 = t
                if self.unstuck_mode == 0:
                    self.unstuck_until = t + 3.0
                    self.unstuck_dir = move
                    jump = True
                else:
                    self.unstuck_until = t + 0.9
                    a = math.atan2(move[1], move[0]) + self.rng.choice((-1, 1)) * (1.2 + self.rng.random())
                    self.unstuck_dir = (math.cos(a), math.sin(a))
                    move = self.unstuck_dir
                self.path = None
                self.search = None
                self.hist = []
                self.roam = None
                self.cover = None
                if self.loot_goal is not None and self.unstuck_n % 3 == 0:
                    self.black[self.loot_goal] = t + 40
                    self.loot_goal = None

        act["move"] = [round(move[0], 3), round(move[1], 3)]
        act["boost"] = bool(boost)
        if jump:
            act["jump"] = True
        if aim is not None:
            act["aim"] = [round(aim[0], 4), round(max(-0.7, min(1.3, aim[1])), 4)]
        if hold_fire and aim_target is not None:
            fire = [False, False]
        act["fire"] = [bool(fire[0]), bool(fire[1])]
        act["debug"] = {"label": "%s hp%d" % (label, hp)}
        return act

    # ------------------------------------------------------------------ roofs
    def pick_roof(self, t, x, y, zc, zr, ncx, ncy, nr, margin):
        if not self.roofs or not self.nav.ok:
            self.roof = None
            return
        def valid(i):
            r = self.roofs[i]
            cx, cy = r["camp"]
            if self.roof_bad.get(i, 0) > t:
                return False
            return math.hypot(cx - ncx, cy - ncy) < nr - r["rad"] * 0.5 - 8 and math.hypot(cx - zc[0], cy - zc[1]) < zr - 25
        if self.roof is not None and valid(self.roof):
            return
        if self.climb is not None:
            return
        self.roof = None
        best = None
        bs = -1e18
        for i, r in enumerate(self.roofs):
            if not valid(i):
                continue
            cx, cy = r["camp"]
            d = math.hypot(cx - x, cy - y)
            if d > self.F.get("roofnear", 260):
                continue
            H = r["z"] - r["acc"][0][4]
            sc = -d + 5.0 * min(r["inset"] * 3.0, 24.0) + (25.0 if 12 <= H <= 30 else 0.0) - (40.0 if H > 30 else 0.0)
            if sc > bs:
                bs = sc
                best = i
        self.roof = best
        self.rp_cands = None
        self.rp_target = None

    def roof_move(self, t, me, threats, target, hide, energy):
        """Go to / climb / hold the home roof. Returns (move, boost, jump)."""
        nav = self.nav
        r = self.roofs[self.roof]
        x, y, z = me["pos"]
        vx, vy, vz = me.get("vel", [0, 0, 0])
        grounded = me.get("grounded", True)
        rz = r["z"]
        if self.onroof:
            self.climb = None
            return self.roof_hold(t, me, r, threats, target, hide), False, False
        cl = self.climb
        if cl is None:
            # nearest usable access point
            best = None
            bd = 1e18
            for a in r["acc"]:
                d = math.hypot(a[0] - x, a[1] - y)
                if d < bd:
                    bd = d
                    best = a
            if bd > 2.5:
                mv = self.nav_move(x, y, best[0], best[1])
                if self.nav_arrived:
                    self.roof_bad[self.roof] = t + 60
                return mv, False, False
            cl = self.climb = {"a": best, "ph": "wait", "t0": t}
        a = cl["a"]
        H = rz - a[4]
        dx, dy = norm2(a[2] - x, a[3] - y)
        if cl["ph"] == "wait":
            need = min(95.0, H * 1.8 + 14.0)
            if energy < need and t - cl["t0"] < 6.0:
                if math.hypot(a[0] - x, a[1] - y) > 3.0:
                    return norm2(a[0] - x, a[1] - y), False, False
                return (0.0, 0.0), False, False
            cl["ph"] = "push"
            cl["t0"] = t
        if cl["ph"] == "push":
            if grounded and (t - cl["t0"] > 1.6 or (t - cl["t0"] > 0.4 and math.hypot(vx, vy) < 2.0)):
                cl["ph"] = "jet"
                cl["t0"] = t
                return (dx, dy), False, True
            return (dx, dy), False, False
        # jet phase
        el = t - cl["t0"]
        if grounded and el > 0.4:
            if z > rz - 2.0:
                self.climb = None
                self.climb_fails = 0
                return (dx, dy), False, False
            self.climb_fails += 1
            self.climb = None
            if self.climb_fails >= 2:
                self.roof_bad[self.roof] = t + 90
                self.climb_fails = 0
                self.roof = None
            return (dx, dy), False, False
        if el > 8.0:
            self.climb = None
            self.roof_bad[self.roof] = t + 90
            self.roof = None
            return (dx, dy), False, False
        boost = energy > 0.5 and (z < rz + 1.2 or (vz < -1.0 and z < rz + 3.0))
        return (dx, dy), boost, (grounded and el <= 0.4)

    def roof_hold(self, t, me, r, threats, target, hide):
        nav = self.nav
        x, y, z = me["pos"]
        th = [a for a in threats[:4] if a[0] < 280][:3]
        if self.rp_cands is None and t - self.rp_t > 0.7:
            self.rp_t = t
            self.rp_cands = r["cells"]
            self.rp_i = 0
            self.rp_best = None
            self.rp_bs = -1e18
        if self.rp_cands is not None:
            cands = self.rp_cands
            n = len(cands)
            i = self.rp_i
            stop = min(self.deadline, time.perf_counter() + 0.002)
            tid = target.get("id") if target is not None else None
            cur = self.rp_target
            while i < n:
                cx, cy, ins = cands[i]
                i += 1
                d = math.hypot(cx - x, cy - y)
                # straight walk must stay on the roof
                okline = True
                k = int(d / 3.0)
                for q in range(1, k + 1):
                    f = q / (k + 1.0)
                    if nav.val(x + (cx - x) * f, y + (cy - y) * f) != 2:
                        okline = False
                        break
                if not okline:
                    continue
                cz = nav.surf_at(cx, cy)
                nvis = 0
                tvis = False
                for dd, m, los in th:
                    p = m["pos"]
                    if nav.los(p[0], p[1], p[2] + 7.0, cx, cy, cz + 4.5, 2.0):
                        nvis += 1
                        if m.get("id") == tid:
                            tvis = True
                if hide or tid is None:
                    sc = -100.0 * nvis + 3.0 * min(ins, 8) - 0.3 * d
                else:
                    sc = (80.0 if tvis else 0.0) - 70.0 * (nvis - (1 if tvis else 0)) - 0.3 * d + 1.0 * min(ins, 8)
                if cur is not None and abs(cx - cur[0]) < 0.1 and abs(cy - cur[1]) < 0.1:
                    sc += 12.0
                if sc > self.rp_bs:
                    self.rp_bs = sc
                    self.rp_best = (cx, cy)
                if time.perf_counter() > stop:
                    break
            self.rp_i = i
            if i >= n:
                self.rp_cands = None
                if self.rp_best is not None:
                    self.rp_target = self.rp_best
        tg = self.rp_target
        if tg is None:
            tg = r["camp"]
            if nav.val(x, y) == 2:
                # camp may not be on a straight roof line; stay put until the search finishes
                return (0.0, 0.0)
        d = math.hypot(tg[0] - x, tg[1] - y)
        if d < 1.5:
            return (0.0, 0.0)
        ux, uy = norm2(tg[0] - x, tg[1] - y)
        # never step off the edge
        if nav.val(x + ux * 4.0, y + uy * 4.0) != 2 and nav.val(x, y) == 2:
            return (0.0, 0.0)
        return (ux, uy)

    # ------------------------------------------------------------------ den (enclosed hiding spot)
    def den_move(self, t, x, y, z, ncx, ncy, nr, zone_ok):
        """Stand still in the most enclosed nearby spot inside the next circle. Returns move or None."""
        nav = self.nav
        den = self.den
        if den is not None and not (math.hypot(den[0] - ncx, den[1] - ncy) < nr - 12 and zone_ok(den[0], den[1], -5)):
            den = self.den = None
        if self.den_cands is None and (den is None or t - self.den_t > 12.0):
            self.den_t = t
            cands = []
            if den is not None:
                cands.append(den)
            for _ in range(28):
                a = self.rng.random() * TAU
                r = 12.0 + 110.0 * self.rng.random()
                cx, cy = x + math.cos(a) * r, y + math.sin(a) * r
                if nav.val(cx, cy) == 1 and math.hypot(cx - ncx, cy - ncy) < nr - 15 and zone_ok(cx, cy, -5):
                    cands.append((cx, cy))
            if not cands and nr > 20:
                for _ in range(10):
                    a = self.rng.random() * TAU
                    r = nr * 0.6 * self.rng.random()
                    cx, cy = ncx + math.cos(a) * r, ncy + math.sin(a) * r
                    if nav.val(cx, cy) == 1:
                        cands.append((cx, cy))
            self.den_cands = cands
            self.den_i = 0
            self.den_best = None
            self.den_bs = -1e18
        if self.den_cands is not None:
            cands = self.den_cands
            i = self.den_i
            stop = min(self.deadline, time.perf_counter() + 0.002)
            while i < len(cands):
                cx, cy = cands[i]
                i += 1
                cz = nav.surf_at(cx, cy)
                blocked = 0
                for k in range(12):
                    a = k * TAU / 12
                    if not nav.los(cx, cy, cz + 8.0, cx + math.cos(a) * 60.0, cy + math.sin(a) * 60.0, cz + 6.0, 3.0):
                        blocked += 1
                sc = blocked * 10.0 - 0.12 * math.hypot(cx - x, cy - y)
                if den is not None and abs(cx - den[0]) < 0.1 and abs(cy - den[1]) < 0.1:
                    sc += 8.0
                if sc > self.den_bs:
                    self.den_bs = sc
                    self.den_best = (cx, cy)
                if time.perf_counter() > stop:
                    break
            self.den_i = i
            if i >= len(cands):
                self.den_cands = None
                if self.den_best is not None:
                    self.den = self.den_best
        den = self.den
        if den is None:
            return None
        if math.hypot(den[0] - x, den[1] - y) < 2.5:
            return (0.0, 0.0)
        return self.nav_move(x, y, den[0], den[1])

    # ------------------------------------------------------------------ fight
    def fight_move(self, t, me, e, d, los, zap, laser, energy, zone_ok):
        x, y, z = me["pos"]
        ex, ey, ez = e["pos"]
        ew = e.get("weapons") or []
        dx, dy = norm2(ex - x, ey - y)
        boost = False
        if self.F.get("kiter", 0):
            R = 115.0 if laser else 85.0
            rad = 0.0
            if d < R - 10:
                rad = -1.0
            elif d > R + 25:
                rad = 0.6
            if not los:
                return None
            sx, sy = -dy * self.strafe, dx * self.strafe
            for sgn in (1.0, -1.0):
                mx, my = norm2(dx * rad + sx * sgn * (0.55 if rad < 0 else 1.0), dy * rad + sy * sgn * (0.55 if rad < 0 else 1.0))
                if self.step_ok(x, y, mx, my) and zone_ok(x + mx * 10, y + my * 10, -12):
                    if sgn < 0:
                        self.strafe = -self.strafe
                        self.strafe_until = t + 0.7
                    return (mx, my), (rad < 0 and d < 60 and energy > 25)
            # boxed in: break line of sight instead
            cm, cb = self.retreat_move(t, x, y, z, [(d, e, True)], zone_ok, energy)
            return cm, cb
        if "lightning" in ew and self.F["zaprespect"] and (laser or not zap or self.F["zaprespect"] > 1):
            R = 75.0 if laser else 58.0
        elif zap and energy > 12:
            R = 24.0
        elif "lightning" in ew:
            R = 75.0 if laser else 58.0
        elif laser:
            R = 45.0 if (self.F["gl"] and self.w_slot(me, "gun")[1]) else 70.0
        else:
            R = 32.0
        nh = getattr(self, "n_close", 1)
        if self.F["kite"] and nh >= 2:
            R = min(140.0, max(R, 45.0) + self.F["kite"] * (nh - 1))
        finish = e.get("hp", 1400) < 320 and me.get("hp", 0) > 0.4 * max(1.0, me.get("max_hp", 1400)) and "lightning" not in ew
        if finish:
            R = min(R, 20.0)
            if not los:
                return self.nav_move(x, y, ex, ey), energy > 50
        if not los:
            # go find it (or let it come to us around the corner when we out-range it at close quarters)
            if d < 60 and not zap:
                # hold the angle and let it come round the corner; keep moving if something is hitting us
                if t - self.last_dmg_t < 2.0:
                    sx, sy = -dy * self.strafe, dx * self.strafe
                    if self.step_ok(x, y, sx, sy):
                        return (sx, sy), False
                return (0.0, 0.0), False
            if (zap and d < 90) or self.F["chase"]:
                return self.nav_move(x, y, ex, ey), False
            return None
        rad = 0.0
        if d > R + 8:
            rad = 1.0
        elif d < R - 8:
            rad = -1.0
        elif d > R:
            rad = 0.3
        tw = 1.0 if abs(rad) < 0.9 else 0.75
        if zap and d > 40:
            tw = 0.45
            boost = energy > 45 and d < 110
        if e.get("charging"):
            tw = 1.0
            rad *= 0.3
            boost = boost or energy > 35
        if finish and d > R + 8:
            boost = boost or energy > 40
        if self.bullet_flip and t >= self.flip_ok_t:
            self.strafe = -self.strafe
            self.strafe_until = t + 0.4 + self.rng.random() * 0.5
            self.flip_ok_t = t + 0.35
        sx, sy = -dy * self.strafe, dx * self.strafe
        mx, my = norm2(dx * rad + sx * tw, dy * rad + sy * tw)
        if not (self.step_ok(x, y, mx, my) and zone_ok(x + mx * 8, y + my * 8, -15)):
            self.strafe = -self.strafe
            self.strafe_until = t + 0.6
            sx, sy = -sx, -sy
            mx, my = norm2(dx * rad + sx * tw, dy * rad + sy * tw)
            if not self.step_ok(x, y, mx, my):
                if rad > 0:
                    return self.nav_move(x, y, ex, ey), boost
                mx, my = norm2(sx, sy)
        if self.F.get("boostfight", 0) and energy > 30 and not (laser and (laser.get("charge") or 0) > 0):
            boost = True   # spare energy (no lightning to feed): strafe and close at boost speed
        return (mx, my), boost

    # ------------------------------------------------------------------ peek (fight one at a time)
    def peek_move(self, t, me, threats, target, R, zone_ok, energy):
        """With several hostiles around, stand where only the target can see us. Returns move or None."""
        nav = self.nav
        if not nav.ok or target is None:
            return None
        x, y, z = me["pos"]
        th = threats[:4]
        tid = target.get("id")
        if self.pk_cands is None and t - self.pk_t > 0.6:
            self.pk_t = t
            cands = [(x, y, 0.0)]
            if self.pk_spot is not None:
                cands.append((self.pk_spot[0], self.pk_spot[1], math.hypot(self.pk_spot[0] - x, self.pk_spot[1] - y)))
            base = self.rng.random() * TAU
            for r in (9.0, 18.0, 30.0, 45.0):
                for k in range(10):
                    a = base + k * TAU / 10
                    cands.append((x + math.cos(a) * r, y + math.sin(a) * r, r))
            self.pk_cands = cands
            self.pk_i = 0
            self.pk_best = None
            self.pk_bs = -1e18
        if self.pk_cands is not None:
            cands = self.pk_cands
            stop = min(self.deadline, time.perf_counter() + 0.002)
            i = self.pk_i
            n = len(cands)
            tp = target["pos"]
            while i < n:
                cx, cy, r = cands[i]
                i += 1
                if r > 0 and nav.val(cx, cy) != 1:
                    continue
                if not zone_ok(cx, cy, -12):
                    continue
                cz = nav.surf_at(cx, cy) if r > 0 else z
                others = 0
                tvis = False
                for d, m, los in th:
                    p = m["pos"]
                    if nav.los(p[0], p[1], p[2] + 7.0, cx, cy, cz + 4.5, 3.0):
                        if m.get("id") == tid:
                            tvis = True
                        else:
                            others += 1
                dt_ = math.hypot(tp[0] - cx, tp[1] - cy)
                sc = (60.0 if tvis else 0.0) - 70.0 * others - 0.35 * abs(dt_ - R) - 0.5 * r
                if self.pk_spot is not None and abs(cx - self.pk_spot[0]) < 0.1 and abs(cy - self.pk_spot[1]) < 0.1:
                    sc += 10
                if sc > self.pk_bs:
                    self.pk_bs = sc
                    self.pk_best = (cx, cy)
                if time.perf_counter() > stop:
                    break
            self.pk_i = i
            if i >= n:
                self.pk_cands = None
                self.pk_spot = self.pk_best
                self.pk_score = self.pk_bs
        if self.pk_spot is None:
            return None
        cx, cy = self.pk_spot
        d = math.hypot(cx - x, cy - y)
        if d < 2.5:
            # jiggle sideways relative to the target so we are not a standing target
            tp = target["pos"]
            ux, uy = norm2(tp[0] - x, tp[1] - y)
            sx, sy = -uy * self.strafe, ux * self.strafe
            if self.step_ok(x, y, sx, sy, 4.0):
                return (sx * 0.7, sy * 0.7)
            return (0.0, 0.0)
        return self.nav_move(x, y, cx, cy)

    # ------------------------------------------------------------------ retreat
    def retreat_move(self, t, x, y, z, threats, zone_ok, energy):
        nav = self.nav
        th = threats[:4]
        if not nav.ok:
            if th:
                e = th[0][1]["pos"]
                return norm2(x - e[0], y - e[1]), energy > 10
            return (0.0, 0.0), False
        exposed = any(los for d, m, los in th)
        th = th[:3]
        if self.cov_cands is None and (t - self.cover_t > 0.7 or self.cover is None):
            self.cover_t = t
            cands = [(x, y, 0.0)]
            if self.cover is not None:
                cands.append((self.cover[0], self.cover[1], math.hypot(self.cover[0] - x, self.cover[1] - y)))
            base = self.rng.random() * TAU
            for r in (14.0, 28.0, 48.0):
                for k in range(12):
                    a = base + k * TAU / 12
                    cands.append((x + math.cos(a) * r, y + math.sin(a) * r, r))
            self.cov_cands = cands
            self.cov_i = 0
            self.cov_best = None
            self.cov_bs = -1e18
        if self.cov_cands is not None:
            cands = self.cov_cands
            stop = min(self.deadline, time.perf_counter() + 0.002)
            i = self.cov_i
            n = len(cands)
            while i < n:
                cx, cy, r = cands[i]
                i += 1
                if r > 0 and nav.val(cx, cy) != 1:
                    continue
                if not zone_ok(cx, cy, -12):
                    continue
                cz = nav.surf_at(cx, cy) if r > 0 else z
                hidden = True
                mind = 1e9
                for d, m, los in th:
                    p = m["pos"]
                    dd = math.hypot(p[0] - cx, p[1] - cy)
                    if dd < mind:
                        mind = dd
                    if hidden and nav.los(p[0], p[1], p[2] + 8.0, cx, cy, cz + 6.0, 4.0):
                        hidden = False
                sc = (120.0 if hidden else 0.0) - 0.7 * r + 0.3 * min(mind, 150.0)
                if self.cover is not None and abs(cx - self.cover[0]) < 0.1 and abs(cy - self.cover[1]) < 0.1:
                    sc += 15
                if sc > self.cov_bs:
                    self.cov_bs = sc
                    self.cov_best = (cx, cy)
                if time.perf_counter() > stop:
                    break
            self.cov_i = i
            if i >= n:
                self.cov_cands = None
                if self.cov_best is not None:
                    self.cover = self.cov_best
        if self.cover is None:
            if th:
                e = th[0][1]["pos"]
                return norm2(x - e[0], y - e[1]), energy > 10
            return (0.0, 0.0), False
        cx, cy = self.cover
        if math.hypot(cx - x, cy - y) < 2.0:
            return (0.0, 0.0), False
        return self.nav_move(x, y, cx, cy), (exposed and energy > 8)

    # ------------------------------------------------------------------ loot
    def loot_move(self, t, me, obs, npcs, hpf, zone_ok, threats):
        nav = self.nav
        x, y, z = me["pos"]
        best = None
        bs = 0.0
        enemy_near = threats and threats[0][0] < self.F.get("near_r", 180)
        if self.F["farm_until"] and t < self.F["farm_until"]:
            enemy_near = False
        econ = self.econ
        if econ:
            enemy_near = False

        def reachable(p):
            if not nav.ok:
                return True
            if nav.val(p[0], p[1]) != 1:
                c = nav.nearest_main(p[0], p[1], 1)
                if c is None:
                    return False
            return p[2] < nav.surf_at(p[0], p[1]) + 3.0 and p[2] > nav.surf_at(p[0], p[1]) - 6.0

        for iid, it in self.items.items():
            if self.black.get(iid, 0) > t:
                continue
            p = it["pos"]
            typ = it["type"]
            if typ == "pickup":
                want, drop = self.want_pickup(me, it["kind"])
                if not want:
                    continue
                val = {"lightning": 110, "laser": 95, "mortar": 45}.get(it["kind"], 20)
                if drop is not None:
                    val *= 0.6
            elif typ == "cube":
                val = 75
            else:
                val = 38
                if econ:
                    val = 50
                elif enemy_near or hpf < 0.7:
                    continue
            if not zone_ok(p[0], p[1]) or not reachable(p):
                continue
            d = math.hypot(p[0] - x, p[1] - y)
            s = val / (d + 25.0)
            if self.loot_goal == iid:
                s *= 1.35
            if s > bs:
                bs = s
                best = (iid, typ, p, 1.2)
        for dr in obs.get("drops") or []:
            p = dr["pos"]
            d = math.hypot(p[0] - x, p[1] - y)
            dg = self.F.get("drops", 0)
            if d / 10.0 + 3 < dr.get("lands_in", 0) + 40 and zone_ok(p[0], p[1], -5) and d < (320 if dg else 260) and (dg or not enemy_near):
                s = (260 if dg else 130) / (d + 25.0)
                if s > bs:
                    bs = s
                    best = (dr["id"], "drop", p, 0)
        # drones: remember where they live so we can come back for them
        nm = self.npc_mem
        for m in npcs:
            nm[m["id"]] = (m["pos"], t, m)
        if nm:
            for nid in [k for k, v in nm.items() if t - v[1] > 150 or (t - v[1] > 1.0 and math.hypot(v[0][0] - x, v[0][1] - y) < 25)]:
                nm.pop(nid, None)
        if (hpf > 0.6 and not enemy_near) or (econ and hpf > 0.35):
            cand = list(npcs)
            if econ:
                seen_ids = set(m["id"] for m in npcs)
                cand += [v[2] for k, v in nm.items() if k not in seen_ids]
            for m in cand:
                p = m["pos"]
                if not zone_ok(p[0], p[1]) or self.black.get(m["id"], 0) > t:
                    continue
                d = math.hypot(p[0] - x, p[1] - y)
                s = (95 if econ else 34) / (d + 25.0)
                if self.loot_goal == m["id"]:
                    s *= 1.35
                if s > bs:
                    bs = s
                    best = (m["id"], "npc", p, 1.75)
        if best is None:
            self.loot_goal = None
            return None
        iid, typ, p, h = best
        if self.loot_goal != iid:
            self.loot_goal = iid
            self.loot_since = t
        elif (t - self.loot_since > 30 or (self.nav_arrived and typ in ("pickup", "cube") and math.hypot(p[0] - x, p[1] - y) > 4.5)) and typ != "drop":
            self.black[iid] = t + 45
            self.loot_goal = None
            return None
        d = math.hypot(p[0] - x, p[1] - y)
        if typ == "npc" and not any(m_.get("id") == iid for m_ in npcs):
            # remembered drone, not in view: walk to where it lives
            if d < 12.0:
                self.npc_mem.pop(iid, None)
            return self.nav_move(x, y, p[0], p[1])
        if typ in ("crate", "npc"):
            clear = (not nav.ok) or nav.los(x, y, z + 7, p[0], p[1], p[2] + h)
            rng_ = 32.0 if typ == "crate" else 50.0
            if clear and d < 190:
                self._shoot = (p, h)
            if clear and d < rng_:
                if typ == "npc":
                    # strafe slowly sideways to throw its aim off
                    ux, uy = norm2(p[0] - x, p[1] - y)
                    sx, sy = -uy * self.strafe, ux * self.strafe
                    if self.step_ok(x, y, sx, sy):
                        return (sx, sy)
                return (0.0, 0.0)
            return self.nav_move(x, y, p[0], p[1])
        if typ == "drop":
            if d < 3.0:
                return (0.0, 0.0)
            return self.nav_move(x, y, p[0], p[1])
        if d < 1.0:
            return (0.0, 0.0)
        return self.nav_move(x, y, p[0], p[1])


# ----------------------------------------------------------------------------------------------
# IO loop
# ----------------------------------------------------------------------------------------------
def _send(msg):
    sys.stdout.write(json.dumps(msg, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _reader(q):
    try:
        for line in sys.stdin:
            if line.strip():
                q.put(line)
    except Exception:
        pass
    q.put(None)


def fallback(obs, last):
    a = {"tick": obs.get("tick", 0)}
    try:
        me = obs["self"]
        zn = obs["zone"]
        dx = zn["next_center"][0] - me["pos"][0]
        dy = zn["next_center"][1] - me["pos"][1]
        d = math.hypot(dx, dy)
        if d > 5:
            a["move"] = [dx / d, dy / d]
        if obs.get("offer"):
            a["pick"] = 0
    except Exception:
        a.update(last)
    return a


def main():
    try:
        import gc
        gc.disable()
    except Exception:
        pass
    q = queue.Queue()
    threading.Thread(target=_reader, args=(q,), daemon=True).start()
    bot = None
    last = {}
    errors = 0
    while True:
        batch = [q.get()]
        while True:
            try:
                batch.append(q.get_nowait())
            except queue.Empty:
                break
        latest = None
        for raw in batch:
            if raw is None:
                return
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            kind = msg.get("type")
            if kind == "hello":
                try:
                    bot = Bot(msg)
                except Exception:
                    log(traceback.format_exc())
                    bot = None
                _send({"type": "ready", "name": Bot.name})
            elif kind == "obs":
                latest = msg
            elif kind == "end":
                return
        if latest is None:
            continue
        tick = 0
        try:
            tick = latest.get("tick", 0)
            action = None
            if bot is not None and isinstance(latest.get("self"), dict):
                try:
                    action = bot.act(latest)
                    last = {k: action[k] for k in ("move", "aim", "boost") if k in action}
                except Exception:
                    errors += 1
                    if errors <= 5 or errors % 200 == 0:
                        log("act failed at tick %s (%d):\n%s" % (tick, errors, traceback.format_exc()))
                    action = None
            if action is None:
                action = fallback(latest, last)
            for k in ("move", "aim"):
                v = action.get(k)
                if v is not None and not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(c, (int, float)) and c == c and abs(c) < 1e9 for c in v)):
                    action.pop(k, None)
            action["tick"] = tick
            try:
                _send(action)
            except (TypeError, ValueError):
                _send({"tick": tick})
        except (BrokenPipeError, KeyboardInterrupt):
            return
        except Exception:
            # never let one bad message kill the process: a dead bot is a sitting target
            try:
                log("loop error:\n%s" % traceback.format_exc())
                _send({"tick": tick})
            except Exception:
                pass


if __name__ == "__main__":
    main()
