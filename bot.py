import math
import random
import time
import mechfight as mf
from world import World, Nav
WEAPON_RANK = {'laser': 3, 'lightning': 2, 'mortar': 1, 'gun': 0}
UPGRADE_PREF = ['plating', 'armor', 'nanites', 'gun_damage', 'gun_rate', 'laser_damage', 'laser_charge', 'gun_mag', 'lightning_damage', 'servos', 'gyros', 'gun_velocity', 'capacitor', 'reactor', 'jets', 'sensors', 'mortar_reload', 'mortar_splash', 'mortar_velocity']

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

class Bot:
    name = 'rohaan'

    def __init__(self, hello):
        self.cfg = hello['config']
        self.w = World(hello['map']['dir'])
        self.nav = Nav(self.w)
        self.rng = random.Random(hello.get('seed', 0))
        self.reach = self.cfg['loot']['pickup_radius']
        self.enemies = {}
        self.dmg_log = []
        self.path = None
        self.path_goal = None
        self.path_t = -99.0
        self.strafe = 1
        self.strafe_until = 0.0
        self.last_pos = None
        self.stuck = 0.0
        self.detour = None
        self.detour_until = -1.0
        self.cover = None
        self.cover_t = -99.0
        self.hold = None
        self.hold_t = -99.0
        self.scan = 0.0
        self.mode = 'start'
        self.last_sound = None
        self.farm_id = None
        self.last_fight_t = -99.0
        self.heal_below = 0.9
        self.last_seen_mech_t = -99.0
        self.ffa = False
        self.ffa_evidence = 0
        self.prev_hp = {}
        self.my_hits = set()
        self.practice_aggro_hp = 0.6
        self.practice_aggro_range = 110
        self.practice_confirm_t = 15.0
        self.my_kills = 0
        self.retreating = False
        self.farm_until = 200.0
        self.r1, self.r2, self.r3, self.rexit = (0.4, 0.9, 0.97, 0.88)
        self.hp_hist = {}
        self.third_hp = 0.6
        self.energy_reserve = 30
        self.drop_range = 220
        import os, json as _json
        self.debug = bool(os.environ.get('BOT_DEBUG'))
        self.dodge_on = True
        self._overrides = _json.loads(os.environ.get('BOT_PARAMS', '{}') or '{}')
        for k, v in self._overrides.items():
            setattr(self, k, v)

    def los(self, a, b):
        return bool(self.w.los_many([a], [b])[0])

    def clear_line(self, a, b):
        nav = self.nav
        L = d2(a, b)
        n = max(2, int(L / 2.0))
        ok = nav.ok
        for k in range(1, n):
            f = k / (n - 1)
            x = a[0] + (b[0] - a[0]) * f
            y = a[1] + (b[1] - a[1]) * f
            i, j = nav.ij(x, y)
            if not ok[j, i]:
                return False
        return True

    def steer(self, t, pos, goal):
        dx, dy = (goal[0] - pos[0], goal[1] - pos[1])
        d = math.hypot(dx, dy)
        if d < 0.7:
            return [0.0, 0.0]
        if d < 10 or self.clear_line(pos, goal):
            self.path = None
            return [dx / d, dy / d]
        if self.path is None or self.path_goal is None or d2(self.path_goal, goal) > 15 or (t - self.path_t > 3.0):
            self.path = self.nav.path(pos, goal, max_nodes=1800)
            self.path_goal = list(goal)
            self.path_t = t
        while self.path and len(self.path) > 1 and (d2(self.path[0], pos) < 5):
            self.path.pop(0)
        if self.path:
            k = 0
            for i in range(min(len(self.path), 6) - 1, 0, -1):
                if self.clear_line(pos, self.path[i]):
                    k = i
                    break
            wp = self.path[k]
            ex, ey = (wp[0] - pos[0], wp[1] - pos[1])
            e = math.hypot(ex, ey) or 1.0
            return [ex / e, ey / e]
        return [dx / d, dy / d]

    def unstick(self, t, pos, mv, dt=0.1):
        if self.last_pos is not None and abs(mv[0]) + abs(mv[1]) > 0.3:
            if d2(self.last_pos, pos) < 0.25:
                self.stuck += dt
            else:
                self.stuck = max(0.0, self.stuck - dt)
        self.last_pos = list(pos)
        jump = False
        if self.stuck > 0.5 and t > self.detour_until:
            a = math.atan2(mv[1], mv[0]) + self.rng.choice([-1, 1]) * self.rng.uniform(1.0, 1.9)
            self.detour = [math.cos(a), math.sin(a)]
            self.detour_until = t + self.rng.uniform(0.6, 1.4)
            self.stuck = 0.0
            self.path = None
            jump = True
        if self.detour is not None and t < self.detour_until:
            return (self.detour, jump)
        return (mv, jump)

    def safe_circle(self, obs):
        z = obs['zone']
        t = obs['t']
        if t > z['shrink_start'] - 30 or z['next_radius'] < 1:
            c, r = (z['next_center'], z['next_radius'])
        else:
            c, r = (z['center'], z['radius'])
        if r < 1:
            c, r = (z['next_center'], 1.0)
        return (c, r)

    def point_in_safe(self, obs, p, margin=0.75):
        c, r = self.safe_circle(obs)
        return d2(p, c) <= r * margin

    def find_cover(self, obs, me, threats):
        pos = me['pos']
        c, r = self.safe_circle(obs)
        cands = []
        for k in range(20):
            a = k * (2 * math.pi / 20) + self.rng.uniform(-0.1, 0.1)
            for rad in (14, 26, 40):
                x, y = (pos[0] + rad * math.cos(a), pos[1] + rad * math.sin(a))
                i, j = self.nav.ij(x, y)
                if not self.nav.ok[j, i]:
                    continue
                if d2((x, y), c) > max(r * 0.85, 8):
                    continue
                cands.append((x, y, float(self.nav.gz[j, i])))
        if not cands or not threats:
            return None
        A, B = ([], [])
        for x, y, z in cands:
            for e in threats:
                A.append([e['pos'][0], e['pos'][1], e['pos'][2] + 8])
                B.append([x, y, z + 6])
        ok = self.w.los_many(A, B)
        n = len(threats)
        best, bs = (None, -1000000000.0)
        for k, (x, y, z) in enumerate(cands):
            hidden = sum((1 for q in range(n) if not ok[k * n + q]))
            if hidden == 0:
                continue
            dist_cost = d2((x, y), pos)
            away = min((d2((x, y), e['pos']) for e in threats))
            s = hidden * 100 - dist_cost * 1.5 + min(away, 80) * 0.5 - d2((x, y), c) / max(r, 1) * 10
            if s > bs:
                best, bs = ((x, y), s)
        return best

    def find_hold(self, obs, me):
        c, r = self.safe_circle(obs)
        nav = self.nav
        best, bs = (None, -1000000000.0)
        rr = max(6.0, r * 0.55)
        for _ in range(60):
            a = self.rng.uniform(0, 2 * math.pi)
            dd = rr * math.sqrt(self.rng.random())
            x, y = (c[0] + dd * math.cos(a), c[1] + dd * math.sin(a))
            i, j = nav.ij(x, y)
            if not nav.ok[j, i]:
                continue
            walls = 0
            for di, dj in ((2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (-1, 1), (1, -1), (-1, -1)):
                a2, b2 = (min(max(i + di, 0), nav.nx - 1), min(max(j + dj, 0), nav.ny - 1))
                if not nav.ok[b2, a2]:
                    walls += 1
            s = walls * 10 - d2((x, y), c) / max(r, 1) * 15 - d2((x, y), me['pos']) * 0.05
            if s > bs:
                best, bs = ((x, y), s)
        return best

    def aim_at(self, me, tgt, speed=None):
        st = me['stats']
        mz = [me['pos'][0], me['pos'][1], me['pos'][2] + st.get('muzzle_height', 7)]
        h = 9.0 * (1 + 0.05 * tgt.get('cubes', 0)) if tgt.get('kind') == 'mech' else 3.5
        p = [tgt['pos'][0], tgt['pos'][1], tgt['pos'][2] + h * 0.5]
        v = tgt.get('vel', [0, 0, 0])
        if speed:
            tt = 0.0
            for _ in range(3):
                q = [p[0] + v[0] * tt, p[1] + v[1] * tt, p[2] + v[2] * tt * 0.3]
                tt = math.dist(q, mz) / speed + 0.05
            p = [p[0] + v[0] * tt, p[1] + v[1] * tt, p[2] + v[2] * tt * 0.3]
        yaw = math.atan2(p[1] - mz[1], p[0] - mz[0])
        hd = d2(p, mz)
        pitch = math.atan2(p[2] - mz[2], hd)
        return ([yaw, pitch], hd, p)

    def mortar_pitch(self, me, p, ws):
        v, g = (ws['speed'], ws['gravity'])
        o = [me['pos'][0], me['pos'][1], me['pos'][2] + me['stats'].get('height', 9) + 0.5]
        d = d2(p, o)
        dz = p[2] - 2.0 - o[2]
        disc = v ** 4 - g * (g * d * d + 2 * dz * v * v)
        if disc < 0:
            return None
        return math.atan((v * v - math.sqrt(disc)) / (g * d))

    def act(self, obs):
        t0 = time.perf_counter()
        t = obs['t']
        me = obs['self']
        pos = me['pos']
        act = {'tick': obs['tick']}
        hpf = me['hp'] / me['max_hp']
        if obs.get('offer'):
            ch = [c['id'] for c in obs['offer']['choices']]
            act['pick'] = min(range(len(ch)), key=lambda i: UPGRADE_PREF.index(ch[i]) if ch[i] in UPGRADE_PREF else 99)
        for e in obs['events']:
            if e['type'] == 'damaged' and e.get('weapon') not in ('zone', None):
                self.dmg_log.append((t, e.get('bearing'), e.get('by')))
        self.dmg_log = [x for x in self.dmg_log if t - x[0] < 4.0]
        for s in obs['sounds']:
            if s['kind'] in ('gun', 'laser', 'laser_charge', 'lightning', 'mortar', 'boost', 'step', 'jet') and s['dist'] != 'far':
                self.last_sound = (t, s['bearing'], s['dist'])
        mechs = [m for m in obs['mechs'] if m['kind'] == 'mech']
        my_hit_ids = {e['target'] for e in obs['events'] if e['type'] == 'hit'}
        for m in mechs:
            hist = self.hp_hist.setdefault(m['id'], [])
            hist.append((t, m['hp'], m['id'] in my_hit_ids))
            while hist and t - hist[0][0] > 2.5:
                hist.pop(0)
        npcs = [m for m in obs['mechs'] if m['kind'] == 'npc']
        for m in mechs:
            self.enemies[m['id']] = dict(m, t=t)
        mine = {m['id'] for m in mechs}
        recent_attackers = {by for _, _, by in self.dmg_log if by is not None and by < 100}
        threats = [m for m in mechs if d2(m['pos'], pos) < 230]

        def hostile(m):
            if m['id'] in recent_attackers:
                return True
            dd = d2(m['pos'], pos)
            if dd < 55:
                return True
            to_me = math.atan2(pos[1] - m['pos'][1], pos[0] - m['pos'][0])
            aiming = abs(wrap(m.get('aim', [0, 0])[0] - to_me)) < 0.3
            v = m.get('vel', [0, 0, 0])
            closing = v[0] * math.cos(to_me) + v[1] * math.sin(to_me) > 4
            return dd < 140 and aiming and closing

        def busy(m):
            hist = self.hp_hist.get(m['id'], [])
            if len(hist) < 3 or any((h[2] for h in hist)):
                return False
            return hist[0][1] - hist[-1][1] > 120

        def weak(m):
            return m['hp'] < m['max_hp'] * 0.35 and d2(m['pos'], pos) < 140 and (hpf > 0.5) and (m['hp'] < me['hp'] * 0.5)
        seen_any = [m for m in threats]
        if seen_any:
            self.last_seen_mech_t = t
        self.detect_ffa(obs, mechs, npcs)
        if self.ffa or t < self.practice_confirm_t:
            fight = [m for m in threats if hostile(m) or weak(m) or (self.ffa and busy(m) and (m['hp'] < m['max_hp'] * self.third_hp) and (hpf > 0.55) and (d2(m['pos'], pos) < 160))]
        else:
            fight = [m for m in threats if hostile(m) or weak(m) or (hpf > self.practice_aggro_hp and d2(m['pos'], pos) < self.practice_aggro_range)]
        watchers = [m for m in threats if m not in fight and d2(m['pos'], pos) < 170]
        threats = fight
        self.handle_loot(obs, me, act)
        target = None
        if threats:

            def tscore(m):
                s = d2(m['pos'], pos)
                if m['id'] in recent_attackers:
                    s -= 60
                s -= (1 - m['hp'] / max(m['max_hp'], 1)) * 60
                return s
            target = min(threats, key=tscore)
        c, r = self.safe_circle(obs)
        in_safe = d2(pos, c) <= r * 0.9
        in_zone = me['in_zone']
        mv = [0.0, 0.0]
        boost = False
        aim = None
        fire = [False, False]
        st = me['stats']
        want_w = self.wanted_pickup(obs, me) if me['weapons'][1] is None else None
        if want_w is not None and d2(want_w['pos'], pos) > 120:
            want_w = None
        hostile_close = target is not None and (target['id'] in recent_attackers or d2(target['pos'], pos) < 80)
        loot_cubes = [q for q in obs['cubes'] if d2(q['pos'], pos) < 45 and abs(q['pos'][2] - pos[2]) < 4]
        close_enemy = target is not None and d2(target['pos'], pos) < 50
        n_close = sum((1 for m in fight if d2(m['pos'], pos) < 120))
        if target is not None:
            if hpf < self.r1 and (not (n_close <= 1 and target['hp'] < me['hp'] * 0.6)):
                self.retreating = True
            if n_close >= 2 and hpf < self.r2:
                self.retreating = True
            if n_close >= 3 and hpf < self.r3:
                self.retreating = True
        if self.retreating:
            if hpf > self.rexit:
                self.retreating = False
            elif target is not None and n_close <= 1 and (target['hp'] < me['hp'] * 0.45):
                self.retreating = False
            elif target is None and (not self.dmg_log) and (t - self.last_fight_t > 25):
                self.retreating = False
        if want_w is not None and in_zone and (not hostile_close) and (not self.retreating):
            self.mode = 'arm'
            mv = self.steer(t, pos, want_w['pos'])
            boost = me['energy'] > 30 and d2(want_w['pos'], pos) > 25
        elif loot_cubes and in_zone and (not close_enemy) and (target is None or hpf > 0.4):
            q = min(loot_cubes, key=lambda q: d2(q['pos'], pos))
            self.mode = 'cubes'
            mv = self.steer(t, pos, q['pos'])
            boost = me['energy'] > 40
        elif not in_zone or (not in_safe and (t > obs['zone']['shrink_start'] - 20 or not target)):
            self.mode = 'zone'
            goal = [c[0] + (pos[0] - c[0]) * 0.3, c[1] + (pos[1] - c[1]) * 0.3]
            mv = self.steer(t, pos, goal)
            boost = not in_zone or me['energy'] > 50
        elif self.drop_goal(obs, me, target) is not None:
            dg = self.drop_goal(obs, me, target)
            self.mode = 'drop'
            mv = self.steer(t, pos, dg['pos']) if d2(dg['pos'], pos) > 3 else [0.0, 0.0]
            boost = d2(dg['pos'], pos) > 40 and me['energy'] > self.energy_reserve
        elif target is not None and self.retreating:
            self.mode = 'retreat'
            if t - self.cover_t > 1.0 or self.cover is None:
                self.cover = self.find_cover(obs, me, threats)
                self.cover_t = t
            if self.cover:
                mv = self.steer(t, pos, self.cover)
            else:
                ax, ay = (pos[0] - target['pos'][0], pos[1] - target['pos'][1])
                L = math.hypot(ax, ay) or 1
                mv = [ax / L, ay / L]
            boost = me['energy'] > 15
        elif target is not None:
            self.mode = 'fight'
            mv = self.fight_move(obs, me, target)
            boost = me['energy'] > 60 and d2(target['pos'], pos) > 120
        elif watchers and target is None:
            self.mode = 'evade'
            if t - self.cover_t > 1.5 or self.cover is None:
                self.cover = self.find_cover(obs, me, watchers)
                self.cover_t = t
            mv = self.steer(t, pos, self.cover) if self.cover and d2(self.cover, pos) > 2.5 else [0.0, 0.0]
        elif hpf < self.heal_below and t - self.last_fight_t < 40:
            self.mode = 'heal'
            if self.hold is None or t - self.hold_t > 15 or (not self.point_in_safe(obs, self.hold, 0.7)):
                self.hold = self.find_hold(obs, me)
                self.hold_t = t
            mv = self.steer(t, pos, self.hold) if self.hold and d2(self.hold, pos) > 3 else [0.0, 0.0]
        else:
            self.mode = 'roam'
            mv, boost = self.roam(obs, me, npcs)
        if target is not None:
            self.last_fight_t = t
        shoot = target
        if shoot is None and self.mode == 'roam' and (hpf > 0.6) and (t - self.last_seen_mech_t > 8) and (t < self.farm_until):
            shoot = self.farm_target(obs, me, npcs)
        if shoot is not None:
            aim, fire = self.weapons(obs, me, shoot)
        else:
            if self.dmg_log and self.dmg_log[-1][1] is not None:
                aim = [self.dmg_log[-1][1], 0.0]
            elif watchers:
                wm = min(watchers, key=lambda m: d2(m['pos'], pos))
                aim = [math.atan2(wm['pos'][1] - pos[1], wm['pos'][0] - pos[0]), 0.0]
            elif self.last_sound and t - self.last_sound[0] < 2.0:
                aim = [self.last_sound[1], 0.0]
            elif self.mode == 'hold':
                self.scan += 0.25
                aim = [self.scan, 0.0]
            elif abs(mv[0]) + abs(mv[1]) > 0.2:
                aim = [math.atan2(mv[1], mv[0]), 0.0]
            else:
                self.scan += 0.25
                aim = [self.scan, 0.0]
            for s, w in enumerate(me['weapons']):
                if w and w['kind'] == 'laser' and (w.get('charge', 0) > 0):
                    fire[s] = True
        if boost and me['energy'] < self.energy_reserve and (self.mode not in ('retreat', 'zone')) and in_zone:
            boost = False
        if self.dodge_on:
            dv, dboost = self.dodge(obs, me, mechs)
            if dv is not None:
                mv = [mv[0] * 0.35 + dv[0], mv[1] * 0.35 + dv[1]]
                boost = boost or (dboost and me['energy'] > 15)
        mv, jump = self.unstick(t, pos, mv)
        act['move'] = [round(mv[0], 3), round(mv[1], 3)]
        act['boost'] = bool(boost)
        act['jump'] = bool(jump)
        act['aim'] = [round(aim[0], 4), round(aim[1], 4)]
        act['fire'] = fire
        g = me['weapons'][0] if me['weapons'][0] and me['weapons'][0]['kind'] == 'gun' else None
        if g and g['ammo'] < 5 and (g['reload'] <= 0) and (not fire[0]) and (target is None):
            act['reload'] = True
        if self.debug and obs['tick'] % 10 == 0:
            vis = sorted(((round(d2(m['pos'], pos)), m['id'], int(m['hp']), ''.join((k[0] for k in m.get('weapons', [])))) for m in mechs))
            mf.log(f"t={t:.0f} {self.mode} hp={me['hp']:.0f}/{me['max_hp']:.0f} cubes={me['cubes']} w={[w['kind'] if w else None for w in me['weapons']]} tgt={(target['id'] if target else None)} vis={vis[:4]} dmg={len(self.dmg_log)}")
        mf.draw.label(f"{self.mode} hp{me['hp']:.0f} {(time.perf_counter() - t0) * 1000:.0f}ms")
        return act

    def drop_goal(self, obs, me, target):
        pos = me['pos']
        if target is not None and d2(target['pos'], pos) < 90:
            return None
        best = None
        for dr in obs.get('drops', []):
            if dr['holder'] not in (None, me['id']) and dr.get('progress', 0) > 1.5:
                continue
            dd = d2(dr['pos'], pos)
            eta = dd / 15.0
            if dd < self.drop_range and dr['lands_in'] < eta + 12 and self.point_in_safe(obs, dr['pos'], 1.0):
                if best is None or dd < d2(best['pos'], pos):
                    best = dr
        return best

    def dodge(self, obs, me, mechs):
        pos = me['pos']
        h = me['stats'].get('height', 9)
        c = [pos[0], pos[1], pos[2] + h / 2]
        r = me['stats'].get('radius', 2.5)
        best = None
        for p in obs['projectiles']:
            v = p['vel']
            if p['kind'] == 'mortar':
                g = self.cfg['weapons']['mortar']['gravity']
                a, b, cc = (-0.5 * g, v[2], p['pos'][2] - pos[2])
                disc = b * b - 4 * a * cc
                if disc < 0:
                    continue
                tl = (-b - math.sqrt(disc)) / (2 * a)
                if not 0 < tl < 3.5:
                    continue
                ix, iy = (p['pos'][0] + v[0] * tl, p['pos'][1] + v[1] * tl)
                d = math.hypot(ix - pos[0], iy - pos[1])
                if d < 11:
                    ax, ay = (pos[0] - ix, pos[1] - iy)
                    L = math.hypot(ax, ay) or 1.0
                    w = (11 - d) / 11 + 0.5
                    if best is None or w > best[2]:
                        best = (ax / L, ay / L, w)
                continue
            rx, ry, rz = (c[0] - p['pos'][0], c[1] - p['pos'][1], c[2] - p['pos'][2])
            vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
            if vv < 1:
                continue
            ts = (rx * v[0] + ry * v[1] + rz * v[2]) / vv
            if not 0.03 < ts < 0.9:
                continue
            mx, my = (rx - v[0] * ts, ry - v[1] * ts)
            dmiss = math.hypot(mx, my)
            if dmiss < r + 2.0:
                vh = math.hypot(v[0], v[1]) or 1.0
                px, py = (-v[1] / vh, v[0] / vh)
                if px * mx + py * my < 0:
                    px, py = (-px, py * -1)
                w = 1.0 - dmiss / (r + 2.0) + 0.3
                if best is None or w > best[2]:
                    best = (px, py, w)
        for m in mechs:
            if not m.get('charging'):
                continue
            dd = d2(m['pos'], pos)
            if dd > 290:
                continue
            to_me = math.atan2(pos[1] - m['pos'][1], pos[0] - m['pos'][0])
            if abs(wrap(m.get('aim', [9, 0])[0] - to_me)) < math.atan2(r + 3, max(dd, 1)) + 0.05:
                px, py = (-math.sin(to_me), math.cos(to_me))
                if self.strafe < 0:
                    px, py = (-px, -py)
                best = (px, py, 2.0)
        if best is None:
            return (None, False)
        return ([best[0], best[1]], best[2] > 0.8)

    def detect_ffa(self, obs, mechs, npcs):
        if self.ffa:
            return
        t = obs['t']
        self.my_kills += sum((1 for e in obs['events'] if e['type'] == 'kill'))
        unexplained = 16 - obs['alive'] - self.my_kills
        if unexplained >= 2:
            self.ffa = True
            mf.log(f'FFA detected (deaths) at t={t}')
            return
        mine = {e['target'] for e in obs['events'] if e['type'] == 'hit'}
        z = obs['zone']
        for m in mechs:
            prev = self.prev_hp.get(m['id'])
            if prev is not None and m['hp'] < prev[0] - 30 and (t - prev[1] < 0.35) and (m['id'] not in mine):
                inzone = d2(m['pos'], z['center']) < z['radius'] - 5
                drone = any((d2(n['pos'], m['pos']) < 110 for n in npcs))
                shooter = False
                for o in mechs:
                    if o['id'] == m['id']:
                        continue
                    dd = d2(o['pos'], m['pos'])
                    if dd > 230:
                        continue
                    bearing = math.atan2(m['pos'][1] - o['pos'][1], m['pos'][0] - o['pos'][0])
                    if abs(wrap(o.get('aim', [9, 0])[0] - bearing)) < math.atan2(4.0, max(dd, 1)) + 0.03:
                        shooter = True
                if inzone and (not drone) and shooter:
                    self.ffa_evidence += 1
                    if self.ffa_evidence >= 3:
                        self.ffa = True
                        mf.log(f'FFA detected at t={t}')
            self.prev_hp[m['id']] = (m['hp'], t)

    def fight_move(self, obs, me, tgt):
        t = obs['t']
        pos = me['pos']
        kinds = [w['kind'] for w in me['weapons'] if w]
        dist = d2(tgt['pos'], pos)
        if 'lightning' in kinds and me['energy'] > 20:
            want = 28
        elif 'laser' in kinds:
            want = 90
        else:
            want = 70
        if 'lightning' in tgt.get('weapons', []) and 'lightning' not in kinds:
            want = max(want, 65)
        ux, uy = ((tgt['pos'][0] - pos[0]) / max(dist, 1e-06), (tgt['pos'][1] - pos[1]) / max(dist, 1e-06))
        radial = 0.0
        if dist > want + 15:
            radial = 1.0
        elif dist < want - 15:
            radial = -0.8
        their_yaw = tgt.get('aim', [0, 0])[0]
        to_me = math.atan2(pos[1] - tgt['pos'][1], pos[0] - tgt['pos'][0])
        side = 1 if wrap(to_me - their_yaw) > 0 else -1
        if t > self.strafe_until:
            self.strafe = side if self.rng.random() < 0.7 else -side
            self.strafe_until = t + self.rng.uniform(0.7, 1.6)
        sx, sy = (-uy * self.strafe, ux * self.strafe)
        mv = [ux * radial + sx * 0.9, uy * radial + sy * 0.9]
        c, r = self.safe_circle(obs)
        nxt = [pos[0] + mv[0] * 10, pos[1] + mv[1] * 10]
        if d2(nxt, c) > r * 0.92:
            cx, cy = (c[0] - pos[0], c[1] - pos[1])
            L = math.hypot(cx, cy) or 1
            mv = [mv[0] * 0.3 + cx / L, mv[1] * 0.3 + cy / L]
        if radial > 0 and (not self.clear_line(pos, tgt['pos'])):
            g = self.steer(t, pos, tgt['pos'])
            mv = [g[0] + sx * 0.4, g[1] + sy * 0.4]
        return mv

    def roam(self, obs, me, npcs):
        t = obs['t']
        pos = me['pos']
        c, r = self.safe_circle(obs)
        cubes = [q for q in obs['cubes'] if abs(q['pos'][2] - pos[2]) < 4 and self.point_in_safe(obs, q['pos'], 0.95)]
        if cubes:
            q = min(cubes, key=lambda q: d2(q['pos'], pos))
            if d2(q['pos'], pos) < 60:
                return (self.steer(t, pos, q['pos']), False)
        for dr in obs.get('drops', []):
            if dr['lands_in'] < 8 and d2(dr['pos'], pos) < 120 and (dr['holder'] in (None, me['id'])):
                return (self.steer(t, pos, dr['pos']), d2(dr['pos'], pos) > 30)
        want = self.wanted_pickup(obs, me)
        if want and d2(want['pos'], pos) < 90:
            return (self.steer(t, pos, want['pos']), False)
        if t < self.farm_until and t - self.last_seen_mech_t > 8:
            tgt = self.farm_target(obs, me, npcs)
            if tgt is not None:
                dd = d2(tgt['pos'], pos)
                if dd > 70:
                    return (self.steer(t, pos, tgt['pos']), False)
                return ([0.0, 0.0], False)
        if self.hold is None or t - self.hold_t > 15 or (not self.point_in_safe(obs, self.hold, 0.7)):
            self.hold = self.find_hold(obs, me)
            self.hold_t = t
        if self.hold:
            if d2(self.hold, pos) > 3:
                return (self.steer(t, pos, self.hold), False)
            self.mode = 'hold'
            return ([0.0, 0.0], False)
        return (self.steer(t, pos, c), False)

    def farm_target(self, obs, me, npcs):
        pos = me['pos']
        eye = [pos[0], pos[1], pos[2] + me['stats'].get('eye_height', 8)]
        opts = []
        for n in npcs:
            dd = d2(n['pos'], pos)
            if dd < 140 and me['hp'] > me['max_hp'] * 0.6:
                opts.append((dd, n))
        for cr in obs['crates']:
            dd = d2(cr['pos'], pos)
            if dd < 120:
                opts.append((dd, dict(cr, kind='crate', vel=[0, 0, 0])))
        opts.sort(key=lambda x: x[0])
        for dd, o in opts[:4]:
            h = 1.2 if o['kind'] == 'crate' else 1.75
            if self.los(eye, [o['pos'][0], o['pos'][1], o['pos'][2] + h]):
                return o
        return None

    def wanted_pickup(self, obs, me):
        held = [w['kind'] if w else None for w in me['weapons']]
        worst = None
        if held[1] is None:
            worst = -1
        else:
            worst = min((WEAPON_RANK[k] for k in held if k and k != 'gun'))
        best = None
        for p in obs['pickups']:
            if p['kind'] in held or abs(p['pos'][2] - me['pos'][2]) > 4:
                continue
            if WEAPON_RANK[p['kind']] > worst and self.point_in_safe(obs, p['pos'], 1.0):
                if best is None or (WEAPON_RANK[p['kind']], -d2(p['pos'], me['pos'])) > (WEAPON_RANK[best['kind']], -d2(best['pos'], me['pos'])):
                    best = p
        return best

    def handle_loot(self, obs, me, act):
        held = [w['kind'] if w else None for w in me['weapons']]
        near = [p for p in obs['pickups'] if d2(p['pos'], me['pos']) < self.reach - 0.3 and abs(p['pos'][2] - me['pos'][2]) < 3.8 and (p['kind'] not in held) and (p['kind'] != 'gun')]
        if not near:
            return
        p = max(near, key=lambda p: WEAPON_RANK[p['kind']])
        if held[1] is None:
            act['take'] = p['id']
            return
        w1 = me['weapons'][1]
        if WEAPON_RANK[p['kind']] > WEAPON_RANK[held[1]] and (not (w1['kind'] == 'laser' and w1.get('charge', 0) > 0)):
            act['take'] = p['id']
            act['drop'] = 1

    def weapons(self, obs, me, tgt):
        fire = [False, False]
        gun_ws = None
        for w in me['weapons']:
            if w and w['kind'] == 'gun':
                gun_ws = w['stats']
        aim, dist, p = self.aim_at(me, tgt, gun_ws['speed'] if gun_ws else 252)
        err = abs(wrap(aim[0] - me['aim'][0])) + abs(aim[1] - me['aim'][1]) * 0.5
        is_mech = tgt.get('kind') == 'mech'
        mortar = None
        for s, w in enumerate(me['weapons']):
            if not w:
                continue
            k = w['kind']
            ws = w['stats']
            if k == 'gun':
                tol = max(0.02, math.atan2(2.2, max(dist, 1)))
                fire[s] = err < tol * 1.5 and dist < ws['range'] - 5 and (w['reload'] <= 0)
            elif k == 'lightning':
                fire[s] = dist < ws['range'] - 3 and abs(wrap(aim[0] - me['aim'][0])) < math.radians(ws['cone_deg']) / 2 * 0.8 and (me['energy'] > 8)
            elif k == 'laser':
                a2, _, _ = self.aim_at(me, tgt, None)
                e2 = abs(wrap(a2[0] - me['aim'][0])) + abs(a2[1] - me['aim'][1]) * 0.5
                charge = w.get('charge', 0)
                if charge > 0:
                    full = charge >= ws['charge'] - 0.06
                    on = e2 < math.atan2(2.0, max(dist, 1))
                    fire[s] = not (full and on)
                else:
                    fire[s] = dist < ws['range'] - 10 and w['cooldown'] <= 0.05 and (is_mech or tgt.get('kind') == 'npc')
            elif k == 'mortar' and is_mech and (45 < dist < 210) and (w['cooldown'] <= 0):
                mortar = (s, ws)
        if mortar is not None and (not any((f for f in fire if f))) or (mortar is not None and self.rng.random() < 0.35):
            pt = self.mortar_pitch(me, p, mortar[1])
            if pt is not None:
                aim = [aim[0], pt]
                if abs(pt - me['aim'][1]) < 0.05 and abs(wrap(aim[0] - me['aim'][0])) < 0.03:
                    fire = [False, False]
                    fire[mortar[0]] = True
        return (aim, fire)
mf.run(Bot)
