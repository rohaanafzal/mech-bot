import json
import math
import os
import numpy as np
INF = 1000000000.0
LAYERS = 4

def _points_in_poly(px, py, ring):
    xs = np.asarray(ring[0::2], dtype=np.float64)
    ys = np.asarray(ring[1::2], dtype=np.float64)
    inside = np.zeros(px.shape, dtype=bool)
    n = len(xs)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = (xs[i], ys[i], xs[j], ys[j])
        cond = (yi > py) != (yj > py)
        if cond.any():
            with np.errstate(divide='ignore', invalid='ignore'):
                xint = (xj - xi) * (py - yi) / (yj - yi) + xi
            inside ^= cond & (px < xint)
        j = i
    return inside

class World:

    def __init__(self, arena_dir):
        self.dir = os.path.abspath(arena_dir)
        w = json.load(open(os.path.join(arena_dir, 'v2', 'world.json')))
        self.raw = w
        t = w['terrain']
        self.nx, self.ny = (t['nx'], t['ny'])
        self.minX, self.minY, self.step = (t['minX'], t['minY'], t['step'])
        terr = np.fromfile(os.path.join(arena_dir, 'v2', t['file']), '<i2').astype(np.float32)
        self.terrain = (terr * t.get('scale', 0.01)).reshape(self.ny, self.nx)
        b = w['bounds']
        self.bounds = (b['minX'], b['maxX'], b['minY'], b['maxY'])
        self.prisms = w['prisms']
        self._raster()

    def _raster(self):
        ny, nx = (self.ny, self.nx)
        base = np.full((LAYERS, ny, nx), INF, dtype=np.float32)
        top = np.full((LAYERS, ny, nx), -INF, dtype=np.float32)
        cnt = np.zeros((ny, nx), dtype=np.int8)
        self.roof_cells = []
        for p in self.prisms:
            ring = p['ring']
            xs, ys = (ring[0::2], ring[1::2])
            i0 = max(0, int(math.floor(min(xs) - self.minX)))
            i1 = min(nx - 1, int(math.ceil(max(xs) - self.minX)))
            j0 = max(0, int(math.floor(min(ys) - self.minY)))
            j1 = min(ny - 1, int(math.ceil(max(ys) - self.minY)))
            if i1 < i0 or j1 < j0:
                continue
            gi, gj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
            px = self.minX + gi * self.step
            py = self.minY + gj * self.step
            m = _points_in_poly(px, py, ring)
            for h in p.get('holes') or []:
                m &= ~_points_in_poly(px, py, h)
            if not m.any():
                continue
            jj, ii = (gj[m], gi[m])
            for a, bb in zip(jj, ii):
                k = cnt[a, bb]
                if k >= LAYERS:
                    continue
                base[k, a, bb] = p['base']
                top[k, a, bb] = p['top']
                cnt[a, bb] = k + 1
            if p['kind'] == 'building' and p['top'] - p['base'] > 4:
                sel = np.random.default_rng(len(self.roof_cells)).choice(len(jj), size=min(len(jj), 40), replace=False)
                for s in sel:
                    self.roof_cells.append((float(self.minX + ii[s]), float(self.minY + jj[s]), float(p['top'])))
        self.base, self.top, self.cnt = (base, top, cnt)
        self.fbase = [base[k].ravel() for k in range(LAYERS)]
        self.ftop = [top[k].ravel() for k in range(LAYERS)]
        self.fcnt = cnt.ravel()
        self.fterr = self.terrain.ravel()
        self.maxtop = np.where(cnt > 0, top.max(axis=0), -INF)

    def _idx(self, x, y):
        i = int(round(x - self.minX))
        j = int(round(y - self.minY))
        if i < 0 or j < 0 or i >= self.nx or (j >= self.ny):
            return -1
        return j * self.nx + i

    def ground(self, x, y):
        fx = (x - self.minX) / self.step
        fy = (y - self.minY) / self.step
        i = min(max(int(fx), 0), self.nx - 2)
        j = min(max(int(fy), 0), self.ny - 2)
        ax, ay = (min(max(fx - i, 0.0), 1.0), min(max(fy - j, 0.0), 1.0))
        t = self.terrain
        return float((t[j, i] * (1 - ax) + t[j, i + 1] * ax) * (1 - ay) + (t[j + 1, i] * (1 - ax) + t[j + 1, i + 1] * ax) * ay)

    def surface(self, x, y, z, step_up):
        g = self.ground(x, y)
        k = self._idx(x, y)
        if k < 0:
            return g
        n = self.fcnt[k]
        lim = z + step_up + 0.01
        for L in range(n):
            tp = self.ftop[L][k]
            if g < tp <= lim:
                g = float(tp)
        return g

    def ceiling(self, x, y, z):
        k = self._idx(x, y)
        c = INF
        if k < 0:
            return c
        for L in range(self.fcnt[k]):
            b = self.fbase[L][k]
            if z < b < c:
                c = float(b)
        return c

    def wall_at(self, x, y, z, height, step_up):
        k = self._idx(x, y)
        if k < 0:
            return True
        lo = z + step_up
        hi = z + height
        for L in range(self.fcnt[k]):
            if self.fbase[L][k] < hi and self.ftop[L][k] > lo:
                return True
        return False

    def body_blocked(self, x, y, z, r, height, step_up):
        bx0, bx1, by0, by1 = self.bounds
        if x - r < bx0 or x + r > bx1 or y - r < by0 or (y + r > by1):
            return True
        if self.wall_at(x, y, z, height, step_up):
            return True
        d = r * 0.85
        for ox, oy in ((d, 0), (-d, 0), (0, d), (0, -d), (0.7 * d, 0.7 * d), (-0.7 * d, 0.7 * d), (0.7 * d, -0.7 * d), (-0.7 * d, -0.7 * d)):
            if self.wall_at(x + ox, y + oy, z, height, step_up):
                return True
        return False

    def solid(self, x, y, z):
        k = self._idx(x, y)
        if k < 0:
            return True
        if z < self.fterr[k] - 0.05:
            return True
        for L in range(self.fcnt[k]):
            if self.fbase[L][k] <= z < self.ftop[L][k]:
                return True
        return False

    def open_ground(self, x, y, clear=3.0):
        i = int(round(x - self.minX))
        j = int(round(y - self.minY))
        c = int(clear)
        if i - c < 0 or j - c < 0 or i + c >= self.nx or (j + c >= self.ny):
            return False
        return not (self.cnt[j - c:j + c + 1, i - c:i + c + 1] > 0).any()

    def solid_many(self, X, Y, Z):
        i = np.rint(X - self.minX).astype(np.int64)
        j = np.rint(Y - self.minY).astype(np.int64)
        out = (i < 0) | (j < 0) | (i >= self.nx) | (j >= self.ny)
        i = np.clip(i, 0, self.nx - 1)
        j = np.clip(j, 0, self.ny - 1)
        s = out | (Z < self.terrain[j, i] - 0.05)
        c = self.cnt[j, i]
        for L in range(LAYERS):
            m = c > L
            if not m.any():
                break
            s |= m & (self.base[L, j, i] <= Z) & (Z < self.top[L, j, i])
        return s

    def los_many(self, A, B, spacing=1.5):
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        if len(A) == 0:
            return np.zeros(0, dtype=bool)
        d = B - A
        L = np.linalg.norm(d, axis=1)
        K = max(2, int(L.max() / spacing) + 2)
        t = np.linspace(0.02, 0.98, K)
        P = A[:, None, :] + d[:, None, :] * t[None, :, None]
        s = self.solid_many(P[..., 0].ravel(), P[..., 1].ravel(), P[..., 2].ravel()).reshape(len(A), K)
        return ~s.any(axis=1)

    def first_hit(self, a, d, maxlen, spacing=0.75):
        n = int(maxlen / spacing) + 1
        t = np.arange(1, n + 1) * spacing
        t = t[t <= maxlen]
        if len(t) == 0:
            return maxlen
        P = np.asarray(a)[None, :] + np.asarray(d)[None, :] * t[:, None]
        s = self.solid_many(P[:, 0], P[:, 1], P[:, 2])
        hit = np.flatnonzero(s)
        return float(t[hit[0]] - spacing * 0.5) if len(hit) else float(maxlen)

class Nav:

    def __init__(self, world, cell=4.0, r=2.5, h=9.0, step_up=1.2):
        self.w = world
        self.cell = cell
        bx0, bx1, by0, by1 = world.bounds
        self.x0, self.y0 = (bx0 + cell / 2, by0 + cell / 2)
        self.nx = int((bx1 - bx0) / cell)
        self.ny = int((by1 - by0) / cell)
        ok = np.zeros((self.ny, self.nx), dtype=bool)
        gz = np.zeros((self.ny, self.nx), dtype=np.float32)
        for j in range(self.ny):
            for i in range(self.nx):
                x, y = (self.x0 + i * cell, self.y0 + j * cell)
                z = world.ground(x, y)
                gz[j, i] = z
                ok[j, i] = not world.body_blocked(x, y, z, r, h, step_up)
        self.ok, self.gz = (ok, gz)
        self.maxrise = max(step_up, cell * 1.19)

    def ij(self, x, y):
        i = int(round((x - self.x0) / self.cell))
        j = int(round((y - self.y0) / self.cell))
        return (min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1))

    def xy(self, i, j):
        return (self.x0 + i * self.cell, self.y0 + j * self.cell)

    def nearest_ok(self, i, j, rad=6):
        if self.ok[j, i]:
            return (i, j)
        best = None
        for dj in range(-rad, rad + 1):
            for di in range(-rad, rad + 1):
                a, b = (i + di, j + dj)
                if 0 <= a < self.nx and 0 <= b < self.ny and self.ok[b, a]:
                    d = di * di + dj * dj
                    if best is None or d < best[0]:
                        best = (d, a, b)
        return (best[1], best[2]) if best else (i, j)

    def path(self, a, b, max_nodes=6000):
        import heapq
        si, sj = self.nearest_ok(*self.ij(a[0], a[1]))
        gi, gj = self.nearest_ok(*self.ij(b[0], b[1]))
        if (si, sj) == (gi, gj):
            return [b]
        ok, gz, mr = (self.ok, self.gz, self.maxrise)
        openh = [(0.0, 0.0, si, sj)]
        came = {(si, sj): None}
        g = {(si, sj): 0.0}
        n = 0
        best = (1e+18, (si, sj))
        while openh and n < max_nodes:
            f, gc, i, j = heapq.heappop(openh)
            n += 1
            hcur = math.hypot(gi - i, gj - j)
            if hcur < best[0]:
                best = (hcur, (i, j))
            if (i, j) == (gi, gj):
                break
            for di, dj, c in ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)):
                a2, b2 = (i + di, j + dj)
                if not (0 <= a2 < self.nx and 0 <= b2 < self.ny) or not ok[b2, a2]:
                    continue
                if di and dj and (not (ok[j, a2] and ok[b2, i])):
                    continue
                if gz[b2, a2] - gz[j, i] > mr * c:
                    continue
                ng = gc + c
                if ng < g.get((a2, b2), 1e+18):
                    g[a2, b2] = ng
                    came[a2, b2] = (i, j)
                    heapq.heappush(openh, (ng + math.hypot(gi - a2, gj - b2), ng, a2, b2))
        end = (gi, gj) if (gi, gj) in came else best[1]
        pts = []
        cur = end
        while cur is not None:
            pts.append(self.xy(*cur))
            cur = came[cur]
        pts.reverse()
        if end == (gi, gj):
            pts[-1] = (b[0], b[1])
        return pts
