"""Market emulator: scoreline grids of the in-match goal process as a function of the
pre-match *mean* goals (ARCHITECTURE.md §7.3, gaps 4–5).

The team-only simulator is run on a log-spaced grid of nominal rates (per-90 base
rates in a level state). Common random numbers make the surfaces smooth, and a
smoothing bicubic spline per output is fitted on top:

* mean home and away goals → ``nominal_for_means`` inverts them by 2-D Newton, so a
  fixture's M1 or market mean goals map to the process's nominal rates;
* every scoreline cell (0–10 each, goals above 10 pooled) → ``grid``.

A 24² grid × 4·10⁴ simulations (≈ 10⁹ team-minutes) replaces the spec's 60² × 10⁵
(gap 5); the spline smoothing recovers the accuracy (checked against direct simulation
in ``evaluate.goal_process``). ``as_goal_model`` wraps it as a ``GoalModel`` taking mean
goals, so market inversion, fusion and the Phase 2 scoring code work unchanged.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RectBivariateSpline

from fplh.models.goal_benchmarks import MAX_GOALS, GoalModel
from fplh.models.goal_process import GoalProcessParams
from fplh.sim.team import score_grid, simulate_team

Array = npt.NDArray[np.float64]
CELLS = MAX_GOALS + 1


def params_sha(params: GoalProcessParams) -> str:
    return hashlib.sha256(json.dumps(params.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Emulator:
    params: GoalProcessParams
    nodes: Array  # log nominal rate per axis
    mean_home: Array  # (G, G)
    mean_away: Array
    cells: Array  # (G, G, CELLS, CELLS)
    n_sims: int
    _splines: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls,
        params: GoalProcessParams,
        *,
        grid: int = 24,
        lo: float = 0.15,
        hi: float = 4.5,
        n_sims: int = 40_000,
        seed: int = 0,
    ) -> Emulator:
        nodes = np.linspace(np.log(lo), np.log(hi), grid)
        mh = np.zeros((grid, grid))
        ma = np.zeros((grid, grid))
        cells = np.zeros((grid, grid, CELLS, CELLS))
        for i in range(grid):
            nominal = np.column_stack([np.full(grid, np.exp(nodes[i])), np.exp(nodes)])
            sim = simulate_team(params, nominal, n_sims, seed)  # same seed: common numbers
            mh[i], ma[i] = sim.home.mean(axis=1), sim.away.mean(axis=1)
            cells[i] = score_grid(sim, MAX_GOALS)
        return cls(params, nodes, mh, ma, cells, n_sims)

    # splines -----------------------------------------------------------------------
    def _spline(self, name: str) -> RectBivariateSpline:
        if name not in self._splines:
            x = self.nodes
            if name in ("mean_home", "mean_away"):
                z = np.log(getattr(self, name))
                s = 0.0
            else:
                i, j = (int(v) for v in name.split(":"))
                z = self.cells[:, :, i, j]
                s = float(np.sum(z * (1 - z)) / self.n_sims)  # Monte Carlo variance
            self._splines[name] = RectBivariateSpline(x, x, z, kx=3, ky=3, s=s)
        out: RectBivariateSpline = self._splines[name]
        return out

    def means_at(self, log_nh: Array, log_na: Array) -> tuple[Array, Array]:
        h = np.exp(self._spline("mean_home").ev(log_nh, log_na))
        a = np.exp(self._spline("mean_away").ev(log_nh, log_na))
        return h, a

    def nominal_for_means(self, mu_h: npt.ArrayLike, mu_a: npt.ArrayLike) -> tuple[Array, Array]:
        """Log nominal rates whose simulated mean goals equal ``(mu_h, mu_a)``."""
        th, ta = np.log(np.asarray(mu_h, float)), np.log(np.asarray(mu_a, float))
        x, y = th.copy(), ta.copy()
        lo, hi = self.nodes[0], self.nodes[-1]
        sh, sa = self._spline("mean_home"), self._spline("mean_away")
        for _ in range(30):
            fh, fa = sh.ev(x, y) - th, sa.ev(x, y) - ta
            a11, a12 = sh.ev(x, y, dx=1), sh.ev(x, y, dy=1)
            a21, a22 = sa.ev(x, y, dx=1), sa.ev(x, y, dy=1)
            det = a11 * a22 - a12 * a21
            dx = (a22 * fh - a12 * fa) / det
            dy = (-a21 * fh + a11 * fa) / det
            x, y = np.clip(x - dx, lo, hi), np.clip(y - dy, lo, hi)
            if max(np.abs(dx).max(initial=0), np.abs(dy).max(initial=0)) < 1e-10:
                break
        return x, y

    def grid_at_nominal(self, log_nh: float, log_na: float) -> Array:
        g = np.array(
            [
                [self._spline(f"{i}:{j}").ev(log_nh, log_na) for j in range(CELLS)]
                for i in range(CELLS)
            ],
            dtype=float,
        )
        g = np.clip(g, 0.0, None)
        out: Array = g / g.sum()
        return out

    def grid(self, mu_h: float, mu_a: float) -> Array:
        """Scoreline grid (CELLS × CELLS) for pre-match mean goals ``(mu_h, mu_a)``."""
        x, y = self.nominal_for_means([mu_h], [mu_a])
        return self.grid_at_nominal(float(x[0]), float(y[0]))

    def as_goal_model(self) -> GoalModel:
        def grid_fn(lh: float, la: float) -> Array:
            return self.grid(lh, la)

        return GoalModel(self.params.level, grid_fn)

    # persistence --------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            params=np.frombuffer(self.params.dump().encode(), dtype=np.uint8),
            nodes=self.nodes,
            mean_home=self.mean_home,
            mean_away=self.mean_away,
            cells=self.cells,
            n_sims=np.array([self.n_sims]),
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> Emulator:
        import yaml

        z = np.load(io.BytesIO(data))
        params = GoalProcessParams.from_dict(yaml.safe_load(bytes(z["params"]).decode()))
        return cls(
            params, z["nodes"], z["mean_home"], z["mean_away"], z["cells"], int(z["n_sims"][0])
        )
