#!/usr/bin/env python3
"""
VSE-lite: epistemic noise + energy model
  t in [0,1]  — knowledge  (0 = pure noise, 1 = perfect)
  l in [0,1]  — energy     (0 = bullet-vote, 1 = full ballot)

Ground truth: 2-D spatial model, utility = -squared_distance.
Noise model : u_tilde = t*u + (1-t)*eta,  eta ~ N(mean_u, std_u)
Ballot model: voter considers top K = ceil(l*m) candidates.

Code written by Anthropic's Claude, slightly modified by me
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, time
warnings.filterwarnings('ignore')

rng = np.random.default_rng(42)

# ── params ────────────────────────────────────────────────────────────────────
N_VOTERS = 120
N_CANDS  = 8          # N candidates → K ∈ {1,2,..,N} for l in 10 steps
N_TRIALS = 150
N_DIM    = 2

N_T = 10
N_L = N_T
T_VALS = np.linspace(0.0, 1.0, N_T)
L_VALS = np.linspace(1/N_CANDS, 1.0, N_L)   # min = consider 1 candidate
OUTPUT_DIR = 'output/plots'
USE_SCORE_INSTEAD_OF_BORDA = False

# ── dark palette ──────────────────────────────────────────────────────────────
BG       = '#0d0d0d'
PANEL    = '#161616'
GRID_C   = '#282828'
TEXT_C   = '#d8d8d8'
MUTED    = '#666666'

COLORS = {
    'Plurality' : '#e05555',
    'RCV'       : '#e09944',
    'Borda'     : '#d4c94a',
    'SCORE'     : '#d4c94a',
    'Approval'  : '#4ec96a',
    'STAR'      : '#4ab8e0',
    'Condorcet' : '#9b6be0',
}

# ── core utilities ────────────────────────────────────────────────────────────

def make_election(nv, nc, nd=2):
    v = rng.uniform(-1, 1, (nv, nd))
    c = rng.uniform(-1, 1, (nc, nd))
    diff = v[:, None, :] - c[None, :, :]        # (nv, nc, nd)
    return -np.sum(diff**2, axis=2)              # (nv, nc)

def add_noise(u, t):
    if t >= 1.0:
        return u.copy()
    mu, sg = u.mean(), u.std()
    sg = sg if sg > 1e-9 else 1.0
    eta = rng.normal(mu, sg, u.shape)
    return t * u + (1 - t) * eta

def K_of(l, nc):
    return max(1, int(np.ceil(l * nc)))

def top_idx(u, l):
    """Returns (n_voters, K) index array of top-K candidates per voter."""
    K = K_of(l, u.shape[1])
    return np.argsort(-u, axis=1)[:, :K], K

def vse(sw_w, sw_rand, sw_opt):
    d = sw_opt - sw_rand
    return (sw_w - sw_rand) / d if d > 1e-9 else 1.0

# ── voting methods ────────────────────────────────────────────────────────────

def plurality(pu, l):
    nc = pu.shape[1]
    top1 = np.argmax(pu, axis=1)
    return int(np.argmax(np.bincount(top1, minlength=nc)))

def approval(pu, l):
    nv, nc = pu.shape
    tidx, K = top_idx(pu, l)

    # considered mask
    cons = np.zeros((nv, nc), dtype=bool)
    np.put_along_axis(cons, tidx, True, axis=1)

    # global personal mean threshold
    thresh = pu.mean(axis=1, keepdims=True)
    appr = cons & (pu > thresh)

    # fallback: approve top-1 if empty
    empty = ~appr.any(axis=1)
    appr[empty, tidx[empty, 0]] = True

    return int(np.argmax(appr.sum(axis=0)))

def borda(pu, l):
    nv, nc = pu.shape
    tidx, K = top_idx(pu, l)
    scores = np.zeros(nc)
    pts = np.arange(K, 0, -1, dtype=float)
    for rank in range(K):
        np.add.at(scores, tidx[:, rank], pts[rank])
    return int(np.argmax(scores))

def star(pu, l):
    nv, nc = pu.shape
    tidx, K = top_idx(pu, l)
    top_u = np.take_along_axis(pu, tidx, axis=1)   # (nv, K)

    lo = top_u.min(axis=1, keepdims=True)
    hi = top_u.max(axis=1, keepdims=True)
    denom = hi - lo
    flat  = (denom < 1e-9).squeeze(axis=1)  # voters where lo == hi
    denom[denom < 1e-9] = 1.0
    scaled = 5 * (top_u - lo) / denom
    scaled[flat] = 5.0                       # give max score to all considered

    ballots = np.zeros((nv, nc))
    np.put_along_axis(ballots, tidx, scaled, axis=1)

    totals = ballots.sum(axis=0)
    f1, f2 = np.argsort(-totals)[:2]
    return int(f1 if (ballots[:, f1] > ballots[:, f2]).sum()
                   >= (ballots[:, f2] > ballots[:, f1]).sum() else f2)

def score(pu, l):
    nv, nc = pu.shape
    tidx, K = top_idx(pu, l)
    top_u = np.take_along_axis(pu, tidx, axis=1)   # (nv, K)

    lo = top_u.min(axis=1, keepdims=True)
    hi = top_u.max(axis=1, keepdims=True)
    denom = hi - lo
    flat  = (denom < 1e-9).squeeze(axis=1)  # voters where lo == hi
    denom[denom < 1e-9] = 1.0
    scaled = 5 * (top_u - lo) / denom
    scaled[flat] = 5.0                       # give max score to all considered

    ballots = np.zeros((nv, nc))
    np.put_along_axis(ballots, tidx, scaled, axis=1)
    totals = ballots.sum(axis=0)
    return int(np.argmax(totals))

def _rank_mat(pu, l):
    """rank_mat[i,c] = rank of c for voter i (0=best); unranked → K."""
    nv, nc = pu.shape
    tidx, K = top_idx(pu, l)
    rm = np.full((nv, nc), K, dtype=np.int32)
    for k in range(K):
        rm[np.arange(nv), tidx[:, k]] = k
    return rm, K

def condorcet(pu, l):
    nv, nc = pu.shape
    rm, K = _rank_mat(pu, l)
    # vectorised pairwise: pw[a,b] = # voters strictly preferring a over b
    ra = rm[:, :, None]      # (nv, nc, 1)
    rb = rm[:, None, :]      # (nv, 1, nc)
    pw = (ra < rb).sum(axis=0)   # (nc, nc)

    # Minimax fallback: choose the candidate with the smallest worst defeat margin.
    margins = pw.T - pw
    np.fill_diagonal(margins, 0)
    worst_defeat = np.maximum(margins, 0).max(axis=1)
    return int(np.argmin(worst_defeat))

def irv(pu, l):
    nv, nc = pu.shape
    tidx, K = top_idx(pu, l)
    ballots = np.argsort(-pu, axis=1)[:, :K]   # (nv, K) preference order

    remaining = np.ones(nc, dtype=bool)

    for _ in range(nc - 1):
        r_idx = np.where(remaining)[0]
        if len(r_idx) == 1:
            break

        # vectorised first-preference count
        in_rem = remaining[ballots]                    # (nv, K) bool
        first  = np.argmax(in_rem, axis=1)             # (nv,)  index within ballots
        has    = in_rem.any(axis=1)

        fp = np.zeros(nc)
        if has.any():
            np.add.at(fp, ballots[has, first[has]], 1)

        total = fp.sum()
        if fp.max() > total / 2:
            return int(np.argmax(fp))

        # eliminate lowest among remaining
        fp_rem = np.where(remaining, fp, np.inf)
        elim = int(np.argmin(fp_rem))
        remaining[elim] = False

    return int(np.where(remaining)[0][0])

def build_methods(use_score_instead_of_borda=False):
    methods = {
        'Plurality' : plurality,
        'RCV'       : irv,
        'Approval'  : approval,
        'STAR'      : star,
        'Condorcet' : condorcet,
    }
    if use_score_instead_of_borda:
        methods['SCORE'] = score
    else:
        methods['Borda'] = borda
    return methods

METHODS = build_methods(USE_SCORE_INSTEAD_OF_BORDA)

# ── simulation grid ───────────────────────────────────────────────────────────

def run_grid(t_vals, l_vals, nv=N_VOTERS, nc=N_CANDS, ntr=N_TRIALS):
    nt, nl = len(t_vals), len(l_vals)
    res = {m: np.zeros((nl, nt)) for m in METHODS}
    total = nt * nl
    done  = 0
    grid_t0 = time.time()
    total_sims = total * ntr

    for ti, t in enumerate(t_vals):
        for li, l in enumerate(l_vals):
            acc = {m: 0.0 for m in METHODS}
            for _ in range(ntr):
                u  = make_election(nv, nc)
                pu = add_noise(u, t)
                sw_r = u.mean()
                sw_o = u.mean(axis=0).max()
                for name, fn in METHODS.items():
                    w = fn(pu, l)
                    acc[name] += vse(u[:, w].mean(), sw_r, sw_o)
            for name in METHODS:
                res[name][li, ti] = acc[name] / ntr
            done += 1
            if done % 12 == 0 or done == total:
                pct = 100 * done / total
                elapsed = time.time() - grid_t0
                sims_done = done * ntr
                avg_per_grid = elapsed / done
                avg_per_sim = elapsed / sims_done
                eta = (total - done) * avg_per_grid
                print(
                    f"  {done}/{total} ({pct:.0f}%)"
                    f" | elapsed {elapsed:.1f}s"
                    f" | eta {eta:.1f}s"
                    f" | avg/grid {avg_per_grid:.3f}s"
                    f" | avg/sim {avg_per_sim:.6f}s"
                    f" | sims {sims_done}/{total_sims}"
                )

    return res

# ── plotting helpers ──────────────────────────────────────────────────────────

def _dark_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig

def _style_ax(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.tick_params(colors=TEXT_C, which='both')
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.title.set_color(TEXT_C)
    ax.grid(color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.6)

def _cb(fig, im, ax, label='VSE'):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=TEXT_C)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_C)
    cb.set_label(label, color=TEXT_C)
    cb.outline.set_edgecolor(GRID_C)

# ── plot 1: axis slices ───────────────────────────────────────────────────────

def plot_slices(res, tv, lv, path='out_slices.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)
    for ax in (ax1, ax2):
        _style_ax(ax)

    li1 = -1                  # l = 1.0 (last index)
    ti1 = -1                  # t = 1.0

    for name, fn in METHODS.items():
        c = COLORS[name]
        ax1.plot(tv, res[name][li1, :], color=c, lw=2, label=name)
        ax2.plot(lv, res[name][:, ti1], color=c, lw=2, label=name)

    ax1.set(xlabel='Knowledge  t', ylabel='VSE',
            title='Full energy (ℓ = 1)  —  VSE vs knowledge')
    ax2.set(xlabel='Energy  ℓ', ylabel='VSE',
            title='Perfect knowledge (t = 1)  —  VSE vs energy')

    for ax in (ax1, ax2):
        ax.set_ylim(0, 1.08)
        ax.legend(facecolor=PANEL, edgecolor=GRID_C,
                  labelcolor=TEXT_C, fontsize=9)

    fig.suptitle('Axis Slices', color=TEXT_C, fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

# ── plot 2: heatmaps ──────────────────────────────────────────────────────────

def plot_heatmaps(res, tv, lv, path='out_heatmaps.png'):
    fig = _dark_fig(17, 10)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)
    extent = [tv[0], tv[-1], lv[0], lv[-1]]

    for idx, name in enumerate(METHODS):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        _style_ax(ax)
        im = ax.imshow(res[name], origin='lower', aspect='auto',
                       extent=extent, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xlabel('Knowledge  t')
        ax.set_ylabel('Energy  ℓ')
        _cb(fig, im, ax)

    fig.suptitle('VSE Surface per Method', color=TEXT_C, fontsize=13, y=1.01)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

# ── plot 3: robustness curves ─────────────────────────────────────────────────

def plot_robustness(res, tv, lv, path='out_robustness.png'):
    tau_range   = np.linspace(0.4, 1.0, 80)
    dt  = tv[1] - tv[0]
    dl  = lv[1] - lv[0]
    tot = (tv[-1] - tv[0]) * (lv[-1] - lv[0])

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG)
    _style_ax(ax)

    for name in METHODS:
        fracs = [(dt * dl * (res[name] >= tau).sum()) / tot
                 for tau in tau_range]
        ax.plot(tau_range, fracs, color=COLORS[name], lw=2, label=name)

    ax.axvline(0.9, color=MUTED, lw=1, ls='--')
    ax.text(0.905, 0.97, 'τ = 0.9', color=MUTED, fontsize=8, va='top',
            transform=ax.get_xaxis_transform())

    ax.set(xlabel='VSE threshold  τ',
           ylabel='Fraction of (t, ℓ) space with VSE ≥ τ',
           title='Robustness: how much of the parameter space stays above threshold?')
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

# ── plot 4: scenario bar charts ───────────────────────────────────────────────

def plot_scenarios(res, tv, lv, path='out_scenarios.png'):
    def idx(arr, v):
        return int(np.argmin(np.abs(arr - v)))

    scenarios = [
        ('Ideal\n(t=1, ℓ=1)',          idx(tv,1.0), idx(lv,1.0)),
        ('Low energy\n(t=1, ℓ=0.35)',  idx(tv,1.0), idx(lv,0.35)),
        ('Low knowledge\n(t=0.3, ℓ=1)',idx(tv,0.3), idx(lv,1.0)),
        ('Both low\n(t=0.3, ℓ=0.35)', idx(tv,0.3), idx(lv,0.35)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True, facecolor=BG)

    for ax, (label, ti, li) in zip(axes, scenarios):
        _style_ax(ax)
        vals   = [res[m][li, ti] for m in METHODS]
        clrs   = [COLORS[m] for m in METHODS]
        bars   = ax.bar(list(METHODS.keys()), vals, color=clrs, width=0.65,
                        edgecolor=GRID_C, linewidth=0.5)
        ax.set_title(label, fontsize=9)
        ax.set_xticklabels(list(METHODS.keys()), rotation=35, ha='right', fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.012,
                    f'{v:.2f}', ha='center', va='bottom', color=TEXT_C, fontsize=7.5)
        ax.set_ylim(0, 1.12)

    axes[0].set_ylabel('VSE', color=TEXT_C)
    fig.suptitle('VSE at Key (t, ℓ) Scenarios', color=TEXT_C, fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

# ── plot 5: approval difference maps ─────────────────────────────────────────

def plot_approval_diff(res, tv, lv, path='out_approval_diff.png'):
    others  = [m for m in METHODS if m != 'Approval']
    n       = len(others)
    extent  = [tv[0], tv[-1], lv[0], lv[-1]]
    vmax    = 0.15

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5), facecolor=BG)
    fig.suptitle('Approval VSE minus each other method\n'
                 '(green = Approval better, red = other better)',
                 color=TEXT_C, fontsize=11, y=1.02)

    for ax, name in zip(axes, others):
        _style_ax(ax)
        diff = res['Approval'] - res[name]
        im = ax.imshow(diff, origin='lower', aspect='auto', extent=extent,
                       cmap='RdYlGn', vmin=-vmax, vmax=vmax)
        ax.contour(diff, levels=[0], colors=[TEXT_C], linewidths=0.8,
                   extent=extent, origin='lower')
        ax.set_title(f'Approval − {name}')
        ax.set_xlabel('Knowledge  t')
        ax.set_ylabel('Energy  ℓ')
        _cb(fig, im, ax, label='ΔVSE')

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

# ── plot 6: weighted avg VSE under different priors ───────────────────────────

def plot_weighted(res, tv, lv, path='out_weighted.png'):
    """Weighted average VSE under three priors over (t, l) space."""
    T, L = np.meshgrid(tv, lv)     # both (nl, nt)

    def w_avg(weights):
        w = weights / weights.sum()
        return {m: (res[m] * w).sum() for m in METHODS}

    # Prior 1: uniform
    p1 = np.ones_like(T)
    # Prior 2: high t, low l  (informed but easily exhausted)
    p2 = np.exp(3*T - 3*(1-L))
    # Prior 3: low t, high l  (energetic but uninformed)
    p3 = np.exp(-3*T + 3*L)

    priors = [
        ('Uniform prior', p1),
        ('Informed & exhausted\n(high t, low ℓ)', p2),
        ('Uninformed & energetic\n(low t, high ℓ)', p3),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True, facecolor=BG)
    for ax, (label, w) in zip(axes, priors):
        _style_ax(ax)
        avgs = w_avg(w)
        vals  = [avgs[m] for m in METHODS]
        clrs  = [COLORS[m] for m in METHODS]
        bars  = ax.bar(list(METHODS.keys()), vals, color=clrs, width=0.65,
                       edgecolor=GRID_C, linewidth=0.5)
        ax.set_title(label, fontsize=9)
        ax.set_xticklabels(list(METHODS.keys()), rotation=35, ha='right', fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', color=TEXT_C, fontsize=7.5)
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel('Weighted average VSE', color=TEXT_C)
    fig.suptitle('Weighted Average VSE Under Different Priors on (t, ℓ)',
                 color=TEXT_C, fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    script_t0 = time.time()
    print(f'Grid: {N_T}×{N_L}  |  Trials: {N_TRIALS}  |  '
          f'Voters: {N_VOTERS}  |  Candidates: {N_CANDS}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    res = run_grid(T_VALS, L_VALS)
    sim_elapsed = time.time() - t0
    print(f'Simulation done in {sim_elapsed:.1f}s\nGenerating plots...')

    plot_t0 = time.time()
    plot_slices(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'out_slices.png'))
    plot_heatmaps(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'out_heatmaps.png'))
    plot_robustness(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'out_robustness.png'))
    plot_scenarios(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'out_scenarios.png'))
    plot_approval_diff(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'out_approval_diff.png'))
    plot_weighted(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'out_weighted.png'))
    plot_elapsed = time.time() - plot_t0
    total_elapsed = time.time() - script_t0

    total_grid_points = N_T * N_L
    total_sims = total_grid_points * N_TRIALS
    print(
        'Timing summary: '
        f'total {total_elapsed:.1f}s | simulation {sim_elapsed:.1f}s | plotting {plot_elapsed:.1f}s '
        f'| avg/grid {sim_elapsed / total_grid_points:.3f}s '
        f'| avg/simulation {sim_elapsed / total_sims:.6f}s'
    )

    print('All plots saved.')
