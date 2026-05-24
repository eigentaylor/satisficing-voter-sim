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
from datetime import datetime
from typing import Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, time
from scipy.stats import ttest_1samp, wilcoxon, t as student_t
warnings.filterwarnings('ignore')

rng = np.random.default_rng(42)

# ── params ────────────────────────────────────────────────────────────────────
N_VOTERS = 99
N_CANDS  = 8          # N candidates → K ∈ {1,2,..,N} for l in 10 steps
N_TRIALS = 100
N_DIM    = 2

N_T = 8
N_L = N_T
T_VALS = np.linspace(0.0, 1.0, N_T)
L_VALS = np.linspace(1/N_CANDS, 1.0, N_L)   # min = consider 1 candidate
OUTPUT_DIR = 'output/plots'
LOG_DIR = 'output/logs'

USE_STRATEGY = True
STRATEGY_SHARE = 1.0

RUN_PAIRED_HYPOTHESIS_TESTS = True
HYPOTHESIS_ALPHA = 0.01
HYPOTHESIS_PAIRS = [
    ('Approval', 'Plurality'),
    ('RCV', 'Plurality'),
    ('Approval', 'RCV'),
    ('STAR', 'Approval'),
    ('Condorcet', 'Approval'),
    ('Borda', 'Approval'),
    ('Score', 'Approval'),
]

SUMMARY_LINES = []

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
    'Score'     : '#b8e04a',
    'Approval'  : '#4ec96a',
    'STAR'      : '#4ab8e0',
    'Condorcet' : '#9b6be0',
}


def _summary(msg=''):
    SUMMARY_LINES.append(msg)


def _summary_kv_line(values_by_method, stars_by_method, precision=4):
    ordered = sorted(values_by_method.items(), key=lambda kv: kv[1], reverse=True)
    parts = [
        f"{name}={val:.{precision}f} (stars={stars_by_method.get(name, 0)})"
        for name, val in ordered
    ]
    return ' | '.join(parts)


def _write_summary_log(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(SUMMARY_LINES) + '\n')


def _write_pvalue_log(path, test_res, run_stamp):
    """Write dedicated hypothesis-test p-value output file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    alpha = test_res['alpha']
    correction = test_res['correction']
    tests_total = test_res['tests_per_family']
    t_vals = test_res['t_vals']
    l_vals = test_res['l_vals']

    lines = []
    lines.append('Satisficing Voter Simulation - Paired Hypothesis P-Values')
    lines.append(f'run_stamp={run_stamp}')
    lines.append(
        f'alpha={alpha:.6f} | correction={correction} | per_grid_tests={tests_total}'
    )
    lines.append(
        'interpretation_note=Per-grid adjusted p-values are Bonferroni-corrected over all pair-cell tests; '
        'pooled tests are reported separately and aggregate across all grid-trial observations.'
    )
    lines.append('')
    lines.append('[POOLED_RESULTS]')
    lines.append(
        'pair\tmean_delta\tt_stat\tt_p\tt_ci_lower\tw_stat\tw_p\tties\tt_sig_nominal\tw_sig_nominal'
    )

    for mx, my in test_res['pairs']:
        pooled = test_res['pooled'][(mx, my)]
        pooled_t = pooled['t']
        pooled_w = pooled['w']
        lines.append(
            f'{mx}>{my}\t{pooled_t["mean"]:.10f}\t{pooled_t["t"]:.10f}\t{pooled_t["p"]:.10e}'
            f'\t{pooled_t["lower_ci"]:.10f}\t{pooled_w["w"]:.10f}\t{pooled_w["p"]:.10e}'
            f'\t{pooled_w["ties"]}\t{int(pooled_t["p"] < alpha)}\t{int(pooled_w["p"] < alpha)}'
        )

    lines.append('')
    lines.append('[PER_GRID_RESULTS]')
    lines.append(
        'pair\tt\tl\tmean_delta\tt_stat\tt_p\tt_p_adj\tt_sig_nominal\tt_sig_adj'
        '\tw_stat\tw_p\tw_p_adj\tw_sig_nominal\tw_sig_adj\tties\tnear_zero_var'
    )

    for mx, my in test_res['pairs']:
        pg = test_res['per_grid'][(mx, my)]
        for li in range(len(l_vals)):
            for ti in range(len(t_vals)):
                t_p = float(pg['t_p'][li, ti])
                w_p = float(pg['w_p'][li, ti])
                t_adj = float(pg['t_p_adj'][li, ti])
                w_adj = float(pg['w_p_adj'][li, ti])
                lines.append(
                    f'{mx}>{my}\t{t_vals[ti]:.6f}\t{l_vals[li]:.6f}'
                    f'\t{pg["mean_delta"][li, ti]:.10f}\t{pg["t_stat"][li, ti]:.10f}'
                    f'\t{t_p:.10e}\t{t_adj:.10e}\t{int(t_p < alpha)}\t{int(t_adj < alpha)}'
                    f'\t{pg["w_stat"][li, ti]:.10f}\t{w_p:.10e}\t{w_adj:.10e}'
                    f'\t{int(w_p < alpha)}\t{int(w_adj < alpha)}'
                    f'\t{int(pg["ties"][li, ti])}\t{int(pg["near_zero_var"][li, ti])}'
                )

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def _write_hypothesis_decision_log(path, test_res, run_stamp):
    """Write a lay-readable hypothesis decision report at the configured alpha."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    alpha = test_res['alpha']
    correction = test_res['correction']
    tests_total = test_res['tests_per_family']

    lines = []
    lines.append('Satisficing Voter Simulation - Hypothesis Decisions')
    lines.append(f'run_stamp={run_stamp}')
    lines.append(
        f'alpha={alpha:.6f} | correction={correction} | per_grid_tests={tests_total}'
    )
    lines.append('')
    lines.append('INTERPRETATION')
    lines.append('- Pooled tests ask whether X tends to beat Y overall across all grid-trial observations.')
    lines.append('- Per-grid adjusted counts use Bonferroni correction and are strict cell-level evidence.')
    lines.append('- A significant p-value indicates statistical evidence, not proof or practical importance by itself.')
    lines.append('')
    lines.append('HYPOTHESIS_ANSWERS')

    for mx, my in test_res['pairs']:
        pg = test_res['per_grid'][(mx, my)]
        pooled_t = test_res['pooled'][(mx, my)]['t']
        pooled_w = test_res['pooled'][(mx, my)]['w']

        pooled_mean_delta = float(pooled_t['mean'])
        pooled_t_p = float(pooled_t['p'])
        pooled_w_p = float(pooled_w['p'])
        pooled_t_sig = pooled_t_p < alpha
        pooled_w_sig = pooled_w_p < alpha
        grid_t_sig = int(pg['t_sig_adj'].sum())
        grid_w_sig = int(pg['w_sig_adj'].sum())
        grid_total = int(pg['t_sig_adj'].size)

        if pooled_mean_delta <= 0:
            verdict = 'NO_CLEAR_EVIDENCE'
            plain = f'No evidence that {mx} is better than {my} overall.'
        elif pooled_t_sig and pooled_w_sig and grid_t_sig > 0:
            verdict = 'SUPPORTED'
            plain = f'Evidence supports that {mx} tends to outperform {my}.'
        elif pooled_t_sig or pooled_w_sig or grid_t_sig > 0:
            verdict = 'MIXED_OR_WEAK'
            plain = f'Results are mixed/weak for {mx} being better than {my}.'
        else:
            verdict = 'NO_CLEAR_EVIDENCE'
            plain = f'No clear evidence that {mx} is better than {my}.'

        lines.append(f'- Hypothesis: {mx} > {my}')
        lines.append(f'  verdict={verdict}')
        lines.append(f'  plain_language={plain}')
        lines.append(
            f'  pooled_mean_delta={pooled_mean_delta:.6f} | pooled_t_p={pooled_t_p:.6e} | '
            f'pooled_w_p={pooled_w_p:.6e}'
        )
        lines.append(
            f'  per_grid_sig_t_adj={grid_t_sig}/{grid_total} | '
            f'per_grid_sig_w_adj={grid_w_sig}/{grid_total}'
        )
        lines.append('')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

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


def _front_and_target(polls, use_third=False):
    order = np.argsort(-polls)
    front = int(order[0])
    if len(order) == 1:
        return front, front
    if use_third and len(order) >= 3:
        return front, int(order[2])
    return front, int(order[1])


def _topk_mask(pu, l):
    nv, nc = pu.shape
    tidx, K = top_idx(pu, l)
    cons = np.zeros((nv, nc), dtype=bool)
    np.put_along_axis(cons, tidx, True, axis=1)
    return tidx, K, cons


def _strat_voter_mask(nv, share):
    return rng.random(nv) < float(np.clip(share, 0.0, 1.0))


def _score_row_from_front_target(row, front, target, top_rank=5):
    pref = target if row[target] > row[front] else front
    other = front if pref == target else target
    hi = row[pref]
    lo = row[other]
    if abs(hi - lo) < 1e-12:
        return np.where(row >= hi, float(top_rank), 0.0)
    scaled = (top_rank + 0.99) * (row - lo) / (hi - lo)
    return np.clip(np.floor(scaled), 0, top_rank)

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

def global_score_ballots(pu, l):
    nv, nc = pu.shape
    tidx, _ = top_idx(pu, l)

    lo = pu.min(axis=1, keepdims=True)
    hi = pu.max(axis=1, keepdims=True)
    denom = hi - lo
    flat  = (denom < 1e-9).squeeze(axis=1)  # voters where lo == hi
    denom[denom < 1e-9] = 1.0
    scaled = 5 * (pu - lo) / denom
    scaled[flat] = 5.0                       # when utilities are flat, give max score to all candidates

    ballots = np.zeros((nv, nc))
    np.put_along_axis(ballots, tidx, np.take_along_axis(scaled, tidx, axis=1), axis=1)
    return ballots

def star(pu, l):
    ballots = global_score_ballots(pu, l)

    totals = ballots.sum(axis=0)
    f1, f2 = np.argsort(-totals)[:2]
    return int(f1 if (ballots[:, f1] > ballots[:, f2]).sum()
                   >= (ballots[:, f2] > ballots[:, f1]).sum() else f2)

def score(pu, l):
    ballots = global_score_ballots(pu, l)
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

def build_methods():
    return {
        'Plurality' : plurality,
        'RCV'       : irv,
        'Borda'     : borda,
        'Score'     : score,
        'Approval'  : approval,
        'STAR'      : star,
        'Condorcet' : condorcet,
    }

METHODS = build_methods()


def _irv_winner_from_ballots(ballots, nc):
    remaining = np.ones(nc, dtype=bool)
    for _ in range(nc - 1):
        r_idx = np.where(remaining)[0]
        if len(r_idx) == 1:
            break

        in_rem = remaining[ballots]
        first = np.argmax(in_rem, axis=1)
        has = in_rem.any(axis=1)

        fp = np.zeros(nc)
        if has.any():
            np.add.at(fp, ballots[has, first[has]], 1)

        total = fp.sum()
        if fp.max() > total / 2:
            return int(np.argmax(fp))

        fp_rem = np.where(remaining, fp, np.inf)
        elim = int(np.argmin(fp_rem))
        remaining[elim] = False

    return int(np.where(remaining)[0][0])


def _condorcet_winner_from_rank_mat(rm):
    ra = rm[:, :, None]
    rb = rm[:, None, :]
    pw = (ra < rb).sum(axis=0)
    margins = pw.T - pw
    np.fill_diagonal(margins, 0)
    worst_defeat = np.maximum(margins, 0).max(axis=1)
    return int(np.argmin(worst_defeat)), pw


def _plurality_winner(pu, l, use_strategy=False, strategy_share=1.0):
    del l
    nv, nc = pu.shape
    top1 = np.argmax(pu, axis=1)
    polls = np.bincount(top1, minlength=nc).astype(float)
    if not use_strategy:
        return int(np.argmax(polls)), polls

    front, target = _front_and_target(polls)
    strat_mask = _strat_voter_mask(nv, strategy_share)
    top1_strat = top1.copy()
    if strat_mask.any():
        choose_target = pu[strat_mask, target] > pu[strat_mask, front]
        top1_strat[strat_mask] = np.where(choose_target, target, front)
    counts = np.bincount(top1_strat, minlength=nc).astype(float)
    return int(np.argmax(counts)), polls


def _approval_winner(pu, l, use_strategy=False, strategy_share=1.0):
    nv, nc = pu.shape
    tidx, _, cons = _topk_mask(pu, l)
    thresh = pu.mean(axis=1, keepdims=True)
    appr = cons & (pu > thresh)
    empty = ~appr.any(axis=1)
    appr[empty, tidx[empty, 0]] = True
    polls = appr.sum(axis=0).astype(float)
    if not use_strategy:
        return int(np.argmax(polls)), polls

    front, target = _front_and_target(polls)
    strat_mask = _strat_voter_mask(nv, strategy_share)
    if strat_mask.any():
        strat_rows = np.where(strat_mask)[0]
        pivot = ((pu[:, front] + pu[:, target]) / 2.0)[:, None]
        strat_appr = cons & (pu >= pivot)
        prefers_target = pu[:, target] > pu[:, front]

        target_rows = strat_rows[prefers_target[strat_rows]]
        front_rows = strat_rows[~prefers_target[strat_rows]]

        if target_rows.size > 0:
            strat_appr[target_rows, front] = False
            strat_appr[target_rows, target] = cons[target_rows, target]
        if front_rows.size > 0:
            strat_appr[front_rows, target] = False
            strat_appr[front_rows, front] = cons[front_rows, front]

        appr[strat_rows] = strat_appr[strat_rows]
        empty = ~appr.any(axis=1)
        appr[empty, tidx[empty, 0]] = True

    return int(np.argmax(appr.sum(axis=0))), polls


def _borda_winner(pu, l, use_strategy=False, strategy_share=1.0):
    nv, nc = pu.shape
    tidx, K = top_idx(pu, l)
    pts = np.arange(K, 0, -1, dtype=float)

    honest_scores = np.zeros(nc)
    for rank in range(K):
        np.add.at(honest_scores, tidx[:, rank], pts[rank])
    polls = honest_scores.copy()
    if not use_strategy:
        return int(np.argmax(honest_scores)), polls

    front, target = _front_and_target(polls)
    strat_mask = _strat_voter_mask(nv, strategy_share)
    scores = np.zeros(nc)
    poll_order_desc = list(np.argsort(-polls))

    for i in range(nv):
        considered = list(tidx[i])
        if (not strat_mask[i]) or K <= 1:
            order = considered
        else:
            order = considered.copy()
            if front in order and target in order:
                middle = [c for c in poll_order_desc if c in order and c not in (front, target)][::-1]
                if pu[i, target] > pu[i, front]:
                    order = [target] + middle + [front]
                else:
                    order = [front] + middle + [target]
        for rank, cand in enumerate(order):
            scores[cand] += pts[rank]

    return int(np.argmax(scores)), polls


def _score_ballots(pu, l):
    nv, nc = pu.shape
    tidx, _ = top_idx(pu, l)

    lo = pu.min(axis=1, keepdims=True)
    hi = pu.max(axis=1, keepdims=True)
    denom = hi - lo
    flat = (denom < 1e-9).squeeze(axis=1)
    denom[denom < 1e-9] = 1.0
    scaled = 5 * (pu - lo) / denom
    scaled[flat] = 5.0

    ballots = np.zeros((nv, nc))
    np.put_along_axis(ballots, tidx, np.take_along_axis(scaled, tidx, axis=1), axis=1)
    return ballots


def _score_winner(pu, l, use_strategy=False, strategy_share=1.0):
    nv, _ = pu.shape
    ballots = _score_ballots(pu, l)
    polls = ballots.sum(axis=0)
    if not use_strategy:
        return int(np.argmax(polls)), polls, ballots

    front, target = _front_and_target(polls)
    tidx, _, cons = _topk_mask(pu, l)
    strat_mask = _strat_voter_mask(nv, strategy_share)
    if strat_mask.any():
        for i in np.where(strat_mask)[0]:
            row = _score_row_from_front_target(pu[i], front, target, top_rank=5)
            row = np.where(cons[i], row, 0.0)
            if not np.any(row > 0):
                row[tidx[i, 0]] = 5.0
            ballots[i] = row
    return int(np.argmax(ballots.sum(axis=0))), polls, ballots


def _star_winner(pu, l, use_strategy=False, strategy_share=1.0):
    _, polls, ballots = _score_winner(pu, l, use_strategy=use_strategy, strategy_share=strategy_share)
    totals = ballots.sum(axis=0)
    f1, f2 = np.argsort(-totals)[:2]
    winner = int(f1 if (ballots[:, f1] > ballots[:, f2]).sum() >= (ballots[:, f2] > ballots[:, f1]).sum() else f2)
    return winner, polls


def _irv_winner(pu, l, use_strategy=False, strategy_share=1.0):
    nv, nc = pu.shape
    tidx, K = top_idx(pu, l)
    ballots = np.argsort(-pu, axis=1)[:, :K]
    polls = np.bincount(ballots[:, 0], minlength=nc).astype(float)
    if not use_strategy:
        return _irv_winner_from_ballots(ballots, nc), polls

    front, target = _front_and_target(polls)
    poll_order_desc = list(np.argsort(-polls))
    strat_mask = _strat_voter_mask(nv, strategy_share)
    for i in np.where(strat_mask)[0]:
        order = list(ballots[i])
        winner_q = pu[i, front]
        targ_q = pu[i, target]

        by_poll_loser_to_winner = [
            c for c in poll_order_desc[::-1]
            if c in order and c != front
        ]

        strat_order = []
        if target in order and targ_q > winner_q:
            strat_order.append(target)
            by_poll_loser_to_winner = [c for c in by_poll_loser_to_winner if c != target]

        for c in by_poll_loser_to_winner:
            if pu[i, c] > winner_q:
                strat_order.append(c)

        if front in order:
            strat_order.append(front)

        for c in by_poll_loser_to_winner:
            if pu[i, c] <= winner_q:
                strat_order.append(c)

        missing = [c for c in order if c not in strat_order]
        if missing:
            strat_order.extend(sorted(missing, key=lambda c: pu[i, c], reverse=True))

        order = strat_order[:K]
        ballots[i] = np.array(order, dtype=ballots.dtype)

    return _irv_winner_from_ballots(ballots, nc), polls


def _condorcet_winner(pu, l, use_strategy=False, strategy_share=1.0):
    nv, nc = pu.shape
    tidx, K = top_idx(pu, l)
    rm, _ = _rank_mat(pu, l)
    honest_winner, pw = _condorcet_winner_from_rank_mat(rm)
    polls = (pw > pw.T).sum(axis=1).astype(float)
    if not use_strategy:
        return honest_winner, polls

    front, target = _front_and_target(polls)
    strat_mask = _strat_voter_mask(nv, strategy_share)
    poll_order = list(np.argsort(-polls))

    for i in np.where(strat_mask)[0]:
        order = list(tidx[i])
        if pu[i, target] > pu[i, front] and front in order and target in order:
            others = [c for c in poll_order if c in order and c not in (front, target)]
            not_too_bad = min(pu[i, front], pu[i, target])
            decent = [c for c in others if pu[i, c] >= not_too_bad]
            bad = [c for c in others if pu[i, c] < not_too_bad]

            decent = sorted(decent, key=lambda c: pu[i, c], reverse=True)
            bad = sorted(bad, key=lambda c: pu[i, c], reverse=True)
            order = decent + [target] + bad + [front]
        else:
            others = [c for c in poll_order if c in order and c != front]
            order = [front] + sorted(others, key=lambda c: pu[i, c], reverse=True)

        rm[i] = K
        for k, cand in enumerate(order):
            rm[i, cand] = k

    winner, _ = _condorcet_winner_from_rank_mat(rm)
    return winner, polls


def _method_winner(name, pu, l, use_strategy=False, strategy_share=1.0):
    if name == 'Plurality':
        winner, _ = _plurality_winner(pu, l, use_strategy=use_strategy, strategy_share=strategy_share)
        return winner
    if name == 'Approval':
        winner, _ = _approval_winner(pu, l, use_strategy=use_strategy, strategy_share=strategy_share)
        return winner
    if name == 'Borda':
        winner, _ = _borda_winner(pu, l, use_strategy=use_strategy, strategy_share=strategy_share)
        return winner
    if name == 'Score':
        winner, _, _ = _score_winner(pu, l, use_strategy=use_strategy, strategy_share=strategy_share)
        return winner
    if name == 'STAR':
        winner, _ = _star_winner(pu, l, use_strategy=use_strategy, strategy_share=strategy_share)
        return winner
    if name == 'RCV':
        winner, _ = _irv_winner(pu, l, use_strategy=use_strategy, strategy_share=strategy_share)
        return winner
    if name == 'Condorcet':
        winner, _ = _condorcet_winner(pu, l, use_strategy=use_strategy, strategy_share=strategy_share)
        return winner
    raise KeyError(f'Unknown method: {name}')


def _one_sided_ttest_greater(deltas):
    """One-sided paired t-test for H1: mean(deltas) > 0 with safeguards."""
    deltas = np.asarray(deltas, dtype=float)
    n = deltas.size
    mean_d = float(deltas.mean())
    sd = float(deltas.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else np.nan

    if n < 2:
        return {
            'n': int(n),
            'mean': mean_d,
            'sd': sd,
            'se': se,
            't': np.nan,
            'p': np.nan,
            'lower_ci': np.nan,
            'near_zero_var': True,
        }

    near_zero_var = sd < 1e-12
    if near_zero_var:
        t_stat = np.inf if mean_d > 0 else -np.inf
        p_val = 0.0 if mean_d > 0 else 1.0
        lower_ci = mean_d
    else:
        ttest_out: Any = ttest_1samp(deltas, popmean=0.0)
        t_stat = float(ttest_out.statistic)
        p_two_sided = float(ttest_out.pvalue)
        p_val = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
        tcrit = float(student_t.ppf(1 - HYPOTHESIS_ALPHA, n - 1))
        lower_ci = mean_d - tcrit * se

    return {
        'n': int(n),
        'mean': mean_d,
        'sd': sd,
        'se': se,
        't': t_stat,
        'p': float(np.clip(p_val, 0.0, 1.0)),
        'lower_ci': float(lower_ci),
        'near_zero_var': near_zero_var,
    }


def _one_sided_wilcoxon_greater(deltas):
    """One-sided Wilcoxon signed-rank test for H1: median(deltas) > 0."""
    deltas = np.asarray(deltas, dtype=float)
    ties = int(np.sum(np.isclose(deltas, 0.0)))
    non_zero = int(deltas.size - ties)

    if non_zero == 0:
        return {
            'w': np.nan,
            'p': 1.0,
            'ties': ties,
            'non_zero': non_zero,
        }

    try:
        w_out: Any = wilcoxon(deltas, alternative='greater', zero_method='wilcox')
        w_stat = float(w_out.statistic)
        p_val = float(w_out.pvalue)
    except ValueError:
        w_stat = np.nan
        p_val = 1.0

    return {
        'w': w_stat,
        'p': float(np.clip(p_val, 0.0, 1.0)),
        'ties': ties,
        'non_zero': non_zero,
    }


def _bonferroni_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = int(pvals.size)
    if m == 0:
        return pvals
    adjusted = np.minimum(1.0, pvals * m)
    return adjusted


def analyze_pairwise_hypotheses(trial_vse, tv, lv, pairs, alpha=HYPOTHESIS_ALPHA):
    """Analyze X>Y hypotheses from per-trial VSE arrays."""
    first_method = next(iter(trial_vse))
    nl, nt, _ = trial_vse[first_method].shape

    per_grid = {}
    pooled = {}

    all_t_pvals = []
    all_w_pvals = []
    grid_refs = []

    for mx, my in pairs:
        deltas = trial_vse[mx] - trial_vse[my]   # (nl, nt, ntr)
        t_p_grid = np.zeros((nl, nt), dtype=float)
        w_p_grid = np.zeros((nl, nt), dtype=float)
        mean_grid = np.zeros((nl, nt), dtype=float)
        lower_ci_grid = np.zeros((nl, nt), dtype=float)
        t_stat_grid = np.zeros((nl, nt), dtype=float)
        w_stat_grid = np.zeros((nl, nt), dtype=float)
        ties_grid = np.zeros((nl, nt), dtype=int)
        nzv_grid = np.zeros((nl, nt), dtype=bool)

        for li in range(nl):
            for ti in range(nt):
                d = deltas[li, ti, :]
                t_stats = _one_sided_ttest_greater(d)
                w_stats = _one_sided_wilcoxon_greater(d)

                t_p_grid[li, ti] = t_stats['p']
                w_p_grid[li, ti] = w_stats['p']
                mean_grid[li, ti] = t_stats['mean']
                lower_ci_grid[li, ti] = t_stats['lower_ci']
                t_stat_grid[li, ti] = t_stats['t']
                w_stat_grid[li, ti] = w_stats['w']
                ties_grid[li, ti] = w_stats['ties']
                nzv_grid[li, ti] = t_stats['near_zero_var']

                all_t_pvals.append(t_stats['p'])
                all_w_pvals.append(w_stats['p'])
                grid_refs.append((mx, my, li, ti))

        pooled_d = deltas.reshape(-1)
        pooled_t = _one_sided_ttest_greater(pooled_d)
        pooled_w = _one_sided_wilcoxon_greater(pooled_d)

        per_grid[(mx, my)] = {
            'mean_delta': mean_grid,
            'lower_ci': lower_ci_grid,
            't_stat': t_stat_grid,
            't_p': t_p_grid,
            'w_stat': w_stat_grid,
            'w_p': w_p_grid,
            'ties': ties_grid,
            'near_zero_var': nzv_grid,
        }
        pooled[(mx, my)] = {
            't': pooled_t,
            'w': pooled_w,
        }

    t_adj_flat = _bonferroni_adjust(np.array(all_t_pvals, dtype=float))
    w_adj_flat = _bonferroni_adjust(np.array(all_w_pvals, dtype=float))

    for idx, (mx, my, li, ti) in enumerate(grid_refs):
        pair_data = per_grid[(mx, my)]
        if 't_p_adj' not in pair_data:
            pair_data['t_p_adj'] = np.zeros((nl, nt), dtype=float)
            pair_data['w_p_adj'] = np.zeros((nl, nt), dtype=float)
            pair_data['t_sig_adj'] = np.zeros((nl, nt), dtype=bool)
            pair_data['w_sig_adj'] = np.zeros((nl, nt), dtype=bool)
        pair_data['t_p_adj'][li, ti] = t_adj_flat[idx]
        pair_data['w_p_adj'][li, ti] = w_adj_flat[idx]
        pair_data['t_sig_adj'][li, ti] = t_adj_flat[idx] < alpha
        pair_data['w_sig_adj'][li, ti] = w_adj_flat[idx] < alpha

    return {
        'alpha': alpha,
        'correction': 'bonferroni',
        'pairs': pairs,
        'per_grid': per_grid,
        'pooled': pooled,
        't_vals': np.array(tv, dtype=float),
        'l_vals': np.array(lv, dtype=float),
        'tests_per_family': len(all_t_pvals),
    }


def report_pairwise_hypotheses(test_res):
    """Print and summarize paired hypothesis test results."""
    alpha = test_res['alpha']
    tests_total = test_res['tests_per_family']

    print('\nPaired hypothesis tests (one-sided: X > Y)')
    print(
        f"  correction={test_res['correction']} | alpha={alpha:.3f}"
        f" | per-grid tests={tests_total}"
    )

    _summary('')
    _summary('PAIRED_HYPOTHESIS_TESTS')
    _summary(
        f"alpha={alpha:.4f} | correction={test_res['correction']}"
        f" | per_grid_tests={tests_total}"
    )

    for mx, my in test_res['pairs']:
        pg = test_res['per_grid'][(mx, my)]
        pooled = test_res['pooled'][(mx, my)]
        pooled_t = pooled['t']
        pooled_w = pooled['w']

        sig_t_count = int(pg['t_sig_adj'].sum())
        sig_w_count = int(pg['w_sig_adj'].sum())
        disagree_count = int((pg['t_sig_adj'] != pg['w_sig_adj']).sum())
        nzv_count = int(pg['near_zero_var'].sum())

        print(
            f"  {mx} > {my}: pooled mean Δ={pooled_t['mean']:.5f}"
            f" | t p={pooled_t['p']:.3e}"
            f" | wilcoxon p={pooled_w['p']:.3e}"
            f" | per-grid sig (t/w)={sig_t_count}/{sig_w_count}"
        )

        _summary(
            f"PAIRED_POOL | {mx}>{my} | mean_delta={pooled_t['mean']:.6f}"
            f" | t={pooled_t['t']:.4f} | t_p={pooled_t['p']:.6e}"
            f" | t_ci_lower={pooled_t['lower_ci']:.6f}"
            f" | w={pooled_w['w']:.4f} | w_p={pooled_w['p']:.6e}"
            f" | ties={pooled_w['ties']}"
        )
        _summary(
            f"PAIRED_GRID_COUNTS | {mx}>{my}"
            f" | t_sig_adj={sig_t_count}"
            f" | w_sig_adj={sig_w_count}"
            f" | disagree={disagree_count}"
            f" | near_zero_var={nzv_count}"
        )

        t_adj = pg['t_p_adj']
        best_idx = np.unravel_index(np.argmin(t_adj), t_adj.shape)
        worst_idx = np.unravel_index(np.argmax(t_adj), t_adj.shape)
        best_li, best_ti = int(best_idx[0]), int(best_idx[1])
        worst_li, worst_ti = int(worst_idx[0]), int(worst_idx[1])

        _summary(
            f"PAIRED_GRID_EXTREMES | {mx}>{my}"
            f" | best_t={test_res['t_vals'][best_ti]:.3f}"
            f" | best_l={test_res['l_vals'][best_li]:.3f}"
            f" | best_mean_delta={pg['mean_delta'][best_li, best_ti]:.6f}"
            f" | best_t_p_adj={pg['t_p_adj'][best_li, best_ti]:.6e}"
            f" | worst_t={test_res['t_vals'][worst_ti]:.3f}"
            f" | worst_l={test_res['l_vals'][worst_li]:.3f}"
            f" | worst_mean_delta={pg['mean_delta'][worst_li, worst_ti]:.6f}"
            f" | worst_t_p_adj={pg['t_p_adj'][worst_li, worst_ti]:.6e}"
        )

# ── simulation grid ───────────────────────────────────────────────────────────

def run_grid(t_vals, l_vals, nv=N_VOTERS, nc=N_CANDS, ntr=N_TRIALS,
             return_trial_vse=False, use_strategy=USE_STRATEGY, strategy_share=STRATEGY_SHARE):
    nt, nl = len(t_vals), len(l_vals)
    res = {m: np.zeros((nl, nt)) for m in METHODS}
    trial_vse = None
    if return_trial_vse:
        trial_vse = {m: np.zeros((nl, nt, ntr), dtype=float) for m in METHODS}
    total = nt * nl
    done  = 0
    grid_t0 = time.time()
    total_sims = total * ntr

    for ti, t in enumerate(t_vals):
        for li, l in enumerate(l_vals):
            acc = {m: 0.0 for m in METHODS}
            for tri in range(ntr):
                u  = make_election(nv, nc)
                pu = add_noise(u, t)
                sw_r = u.mean()
                sw_o = u.mean(axis=0).max()
                for name, fn in METHODS.items():
                    if use_strategy:
                        w = _method_winner(name, pu, l, use_strategy=True, strategy_share=strategy_share)
                    else:
                        w = fn(pu, l)
                    vse_val = vse(u[:, w].mean(), sw_r, sw_o)
                    acc[name] += vse_val
                    if trial_vse is not None:
                        trial_vse[name][li, ti, tri] = vse_val
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

    return res, trial_vse

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

def _detect_tier_mins(values, alpha=0.5):
    """Return tier boundary minimums (mirrors JS detectTierMins)."""
    clean = sorted(v for v in values if np.isfinite(v))
    if len(clean) == 0:
        return []
    if len(clean) == 1:
        return [clean[0]]
    gaps = [clean[i+1] - clean[i] for i in range(len(clean)-1)]
    threshold = alpha * np.mean(gaps)
    mins = [clean[0]]
    for i, g in enumerate(gaps):
        if g > threshold:
            mins.append(clean[i+1])
    return mins


def _tier_star_counts(values, alpha=0.5, max_stars=3):
    """Return stars per value using same tier logic as chart annotation."""
    positive_values = [v for v in values if np.isfinite(v) and v > 0]
    positive_tier_mins = _detect_tier_mins(positive_values, alpha)
    out = []
    for v in values:
        if not np.isfinite(v) or v <= 0:
            out.append(0)
            continue
        positive_tier_idx = sum(v >= tier_min for tier_min in positive_tier_mins) - 1
        stars = max(0, positive_tier_idx)
        out.append(min(stars, max_stars))
    return out

def _annotate_stars(ax, bars, values, alpha=0.5, max_stars=3,
                   color='#ffd54a', size=11, label_gap=0.038, star_spacing=0.16):
    """Draw tiered gold stars above bars (mirrors JS TOP_TIER_STARS_PLUGIN tiered mode).

    Each bar gets stars equal to its tier index (capped at max_stars).
    Tier 0 (lowest cluster) gets 0 stars; each higher tier adds one more.
    Tiers are computed from all finite values, but only strictly positive bars can earn stars.
    Positive tiers are renumbered after filtering, but the lowest positive tier still gets 0 stars.
    """
    stars_per_bar = _tier_star_counts(values, alpha=alpha, max_stars=max_stars)
    for bar, v, stars in zip(bars, values, stars_per_bar):
        if not np.isfinite(v) or v <= 0:
            continue
        if stars <= 0:
            continue
        cx = bar.get_x() + bar.get_width() / 2
        y = v + label_gap
        for i in range(stars):
            x_off = (i - (stars - 1) / 2) * star_spacing
            ax.plot(cx + x_off, y, marker='*', color=color, markersize=size,
                    linestyle='none', clip_on=False, zorder=5)

def _cb(fig, im, ax, label='VSE'):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=TEXT_C)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_C)
    cb.set_label(label, color=TEXT_C)
    cb.outline.set_edgecolor(GRID_C)

def _weighted_priors(tv, lv):
    """Return the shared prior grids used for weighted summaries over (t, l)."""
    T, L = np.meshgrid(tv, lv)     # both (nl, nt)
    sigma = 0.2
    center_focus = np.exp(-(((T - 0.5) ** 2 + (L - 0.5) ** 2) / (2 * sigma ** 2)))
    return [
        ('Uniform prior', np.ones_like(T)),
        ('Ideal focus\n(high t, high ℓ)', np.exp(3 * T + 3 * L)),
        ('Center focus\n(mid t, mid ℓ)', center_focus),
        ('Informed & exhausted\n(high t, low ℓ)', np.exp(3 * T - 3 * (1 - L))),
        ('Energetic & uninformed\n(low t, high ℓ)', np.exp(-3 * T + 3 * L)),
        ('Uninformed & exhausted\n(low t, low ℓ)', np.exp(-3 * T + 3 * (1 - L))),
    ]

def _weighted_method_scores(res, weights):
    w = weights / weights.sum()
    return {m: float((res[m] * w).sum()) for m in METHODS}

def _scenario_points(tv, lv):
    """Return shared named scenarios as (label, t_idx, l_idx)."""
    def idx(arr, v):
        return int(np.argmin(np.abs(arr - v)))

    return [
        ('Ideal\n(t=1, ℓ=1)',           idx(tv, 1.0), idx(lv, 1.0)),
        ('Low energy\n(t=1, ℓ=0.35)',   idx(tv, 1.0), idx(lv, 0.35)),
        ('Low knowledge\n(t=0.3, ℓ=1)', idx(tv, 0.3), idx(lv, 1.0)),
        ('Both low\n(t=0.3, ℓ=0.35)',   idx(tv, 0.3), idx(lv, 0.35)),
    ]

def _scenario_method_scores(res, scenarios):
    """Return per-scenario method scores keyed by scenario label."""
    return {
        label: {m: float(res[m][li, ti]) for m in METHODS}
        for (label, ti, li) in scenarios
    }

def _label_bars(ax, bars, values, pad=0.008, fmt='{:.3f}', positive_va='bottom', negative_va='top'):
    for bar, v in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        if v >= 0:
            y = v + pad
            va = positive_va
        else:
            y = v - pad
            va = negative_va
        ax.text(x, y, fmt.format(v), ha='center', va=va, color=TEXT_C, fontsize=7.5)


def _plot_nonnegative_or_mark_negative(ax, labels, values, colors, width=0.65,
                                       edgecolor=GRID_C, linewidth=0.5,
                                       neg_text_fmt='NEGATIVE\n{:.2f}'):
    x = np.arange(len(labels))
    bars = []
    for xi, v, c in zip(x, values, colors):
        if v >= 0:
            bar = ax.bar(xi, v, color=c, width=width, edgecolor=edgecolor, linewidth=linewidth)[0]
            ax.text(xi, v + 0.012, f'{v:.2f}', ha='center', va='bottom', color=TEXT_C, fontsize=7.5)
        else:
            bar = ax.bar(xi, 0.0, color='none', width=width, edgecolor='none', linewidth=0.0)[0]
            ax.text(xi, 0.035, neg_text_fmt.format(v), ha='center', va='bottom', color=TEXT_C, fontsize=7.0)
        bars.append(bar)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    return bars

def _plot_delta_heatmaps(res, tv, lv, base_method, compared_methods, title, path):
    n = len(compared_methods)
    extent = [tv[0], tv[-1], lv[0], lv[-1]]
    vmax = max(np.max(np.abs(res[name] - res[base_method])) for name in compared_methods)
    vmax = max(vmax, 1e-9)

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5), facecolor=BG)
    if n == 1:
        axes = [axes]
    fig.suptitle(title, color=TEXT_C, fontsize=11, y=1.02)

    for ax, name in zip(axes, compared_methods):
        _style_ax(ax)
        diff = res[name] - res[base_method]
        im = ax.imshow(diff, origin='lower', aspect='auto', extent=extent,
                       cmap='RdYlGn', vmin=-vmax, vmax=vmax)
        ax.contour(diff, levels=[0], colors=[TEXT_C], linewidths=0.8,
                   extent=extent, origin='lower')
        ax.set_title(f'{name} − {base_method}')
        ax.set_xlabel('Knowledge  t')
        ax.set_ylabel('Energy  ℓ')
        _cb(fig, im, ax, label='ΔVSE')

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

# ── plot 1: axis slices ───────────────────────────────────────────────────────

def plot_slices(res, tv, lv, path='strat_slices.png'):
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

def plot_heatmaps(res, tv, lv, path='strat_heatmaps.png'):
    import math
    n_methods = len(METHODS)
    n_cols    = 3
    n_rows    = math.ceil(n_methods / n_cols)
    fig = _dark_fig(17, 5.5 * n_rows)
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.38, wspace=0.32)
    extent = [tv[0], tv[-1], lv[0], lv[-1]]

    for idx, name in enumerate(METHODS):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        _style_ax(ax)
        im = ax.imshow(res[name], origin='lower', aspect='auto',
                       extent=extent, cmap='viridis', vmin=0, vmax=1) # type: ignore
        ax.set_title(name)
        ax.set_xlabel('Knowledge  t')
        ax.set_ylabel('Energy  ℓ')
        _cb(fig, im, ax)

    # hide any unused cells in the last row
    for idx in range(n_methods, n_rows * n_cols):
        fig.add_subplot(gs[idx // n_cols, idx % n_cols]).set_visible(False)

    fig.suptitle('VSE Surface per Method', color=TEXT_C, fontsize=13, y=1.01)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

# ── plot 3: robustness curves ─────────────────────────────────────────────────

def plot_robustness(res, tv, lv, path='strat_robustness.png'):
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

def plot_scenarios(res, tv, lv, path='strat_scenarios.png'):
    scenarios = _scenario_points(tv, lv)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True, facecolor=BG)
    axes = axes.flatten()

    for ax, (label, ti, li) in zip(axes, scenarios):
        _style_ax(ax)
        vals   = [res[m][li, ti] for m in METHODS]
        labels = list(METHODS.keys())
        clrs   = [COLORS[m] for m in METHODS]
        bars = _plot_nonnegative_or_mark_negative(ax, labels, vals, clrs)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0, 1.24)
        _annotate_stars(ax, bars, vals, label_gap=0.06)

        stars = _tier_star_counts(vals)
        vals_by_method = {m: float(v) for m, v in zip(METHODS.keys(), vals)}
        stars_by_method = {m: int(s) for m, s in zip(METHODS.keys(), stars)}
        _summary(
            f"SCENARIO | {label.replace(chr(10), ' ')} | t={tv[ti]:.3f} | l={lv[li]:.3f} | "
            + _summary_kv_line(vals_by_method, stars_by_method)
        )

    axes[0].set_ylabel('VSE', color=TEXT_C)
    axes[2].set_ylabel('VSE', color=TEXT_C)
    fig.suptitle('VSE at Key (t, ℓ) Scenarios', color=TEXT_C, fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

# ── plot 4b: scenario improvement vs reform baselines ───────────────────────

def _plot_scenario_improvement_against(res, tv, lv, base_method, target_methods, path):
    """Scenario VSE gain relative to a single reform baseline."""
    scenarios = _scenario_points(tv, lv)
    scenario_scores = _scenario_method_scores(res, scenarios)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True, facecolor=BG)
    axes = axes.flatten()

    all_vals = []
    for label, _, _ in scenarios:
        scores = scenario_scores[label]
        all_vals.extend(scores[name] - scores[base_method] for name in target_methods)

    span = max(all_vals) - min(all_vals)
    pad = max(0.01, 0.2 * max(span, max(abs(v) for v in all_vals)))
    y_min = min(all_vals) - pad
    y_max = max(all_vals) + pad

    for ax, (label, _, _) in zip(axes, scenarios):
        _style_ax(ax)
        scores = scenario_scores[label]
        vals = [scores[name] - scores[base_method] for name in target_methods]
        clrs = [COLORS[name] for name in target_methods]
        bars = ax.bar(target_methods, vals, color=clrs, width=0.65,
                      edgecolor=GRID_C, linewidth=0.5)
        ax.axhline(0, color=MUTED, lw=1, ls='--')
        ax.set_title(label, fontsize=9)
        ax.set_xticklabels(target_methods, rotation=35, ha='right', fontsize=8)
        label_pad = max(0.012, pad * 0.12)
        _label_bars(ax, bars, vals, pad=label_pad)
        ax.set_ylim(y_min, y_max)
        _annotate_stars(ax, bars, vals, label_gap=label_pad + max(0.02, pad * 0.18))

        stars = _tier_star_counts(vals)
        vals_by_method = {m: float(v) for m, v in zip(target_methods, vals)}
        stars_by_method = {m: int(s) for m, s in zip(target_methods, stars)}
        _summary(
            f"SCENARIO_IMPROVEMENT | baseline={base_method} | scenario={label.replace(chr(10), ' ')} | "
            + _summary_kv_line(vals_by_method, stars_by_method)
        )

    axes[0].set_ylabel('Scenario improvement score', color=TEXT_C)
    axes[2].set_ylabel('Scenario improvement score', color=TEXT_C)
    fig.suptitle(
        f'Scenario Improvement vs {base_method}\n'
        '(ΔVSE at each scenario; positive = reform target better)',
        color=TEXT_C, fontsize=12, y=1.01
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

def plot_scenario_plurality_improvement(res, tv, lv,
                                        path='strat_scenario_plurality_improvement.png'):
    _plot_scenario_improvement_against(
        res, tv, lv,
        base_method='Plurality',
        target_methods=[m for m in METHODS if m != 'Plurality'],
        path=path,
    )

def plot_scenario_approval_improvement(res, tv, lv,
                                       path='strat_scenario_approval_improvement.png'):
    _plot_scenario_improvement_against(
        res, tv, lv,
        base_method='Approval',
        target_methods=[m for m in METHODS if m not in ('Plurality', 'Approval')],
        path=path,
    )

def plot_scenario_overall_improvement(res, tv, lv,
                                      path='strat_scenario_overall_improvement.png'):
    """Grouped comparison of scenario improvement vs Plurality and Approval."""
    scenarios = _scenario_points(tv, lv)
    scenario_scores = _scenario_method_scores(res, scenarios)
    target_methods = [m for m in METHODS if m not in ('Plurality', 'Approval')]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True, facecolor=BG)
    axes = axes.flatten()

    all_vals = []
    for label, _, _ in scenarios:
        scores = scenario_scores[label]
        all_vals.extend(scores[name] - scores['Plurality'] for name in target_methods)
        all_vals.extend(scores[name] - scores['Approval'] for name in target_methods)

    span = max(all_vals) - min(all_vals)
    pad = max(0.01, 0.2 * max(span, max(abs(v) for v in all_vals)))
    y_min = min(all_vals) - pad
    y_max = max(all_vals) + pad
    x = np.arange(len(target_methods))
    width = 0.34

    for ax, (label, _, _) in zip(axes, scenarios):
        _style_ax(ax)
        scores = scenario_scores[label]
        plurality_vals = [scores[name] - scores['Plurality'] for name in target_methods]
        approval_vals = [scores[name] - scores['Approval'] for name in target_methods]

        bars_plurality = ax.bar(
            x - width / 2,
            plurality_vals,
            width=width,
            color=COLORS['Plurality'],
            alpha=0.85,
            edgecolor=GRID_C,
            linewidth=0.5,
            label='vs Plurality',
        )
        bars_approval = ax.bar(
            x + width / 2,
            approval_vals,
            width=width,
            color=COLORS['Approval'],
            alpha=0.85,
            edgecolor=GRID_C,
            linewidth=0.5,
            label='vs Approval',
        )

        ax.axhline(0, color=MUTED, lw=1, ls='--')
        ax.set_title(label, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(target_methods, rotation=35, ha='right', fontsize=8)
        ax.set_ylim(y_min, y_max)

        label_pad = max(0.012, pad * 0.12)
        _label_bars(ax, bars_plurality, plurality_vals, pad=label_pad, fmt='{:.3f}')
        _label_bars(ax, bars_approval, approval_vals, pad=label_pad, fmt='{:.3f}')

        plurality_stars = _tier_star_counts(plurality_vals)
        approval_stars = _tier_star_counts(approval_vals)
        pl_vals_map = {m: float(v) for m, v in zip(target_methods, plurality_vals)}
        ap_vals_map = {m: float(v) for m, v in zip(target_methods, approval_vals)}
        pl_stars_map = {m: int(s) for m, s in zip(target_methods, plurality_stars)}
        ap_stars_map = {m: int(s) for m, s in zip(target_methods, approval_stars)}
        clean_label = label.replace(chr(10), ' ')
        _summary(
            f"SCENARIO_OVERALL | baseline=Plurality | scenario={clean_label} | "
            + _summary_kv_line(pl_vals_map, pl_stars_map)
        )
        _summary(
            f"SCENARIO_OVERALL | baseline=Approval | scenario={clean_label} | "
            + _summary_kv_line(ap_vals_map, ap_stars_map)
        )

    axes[0].set_ylabel('Scenario improvement score', color=TEXT_C)
    axes[2].set_ylabel('Scenario improvement score', color=TEXT_C)
    axes[0].legend(facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=9)
    fig.suptitle(
        'Overall Scenario Improvement by Reform Baseline\n'
        '(grouped bars compare each reform target against Plurality and Approval)',
        color=TEXT_C, fontsize=12, y=1.01
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

# ── plot 5: approval difference maps ─────────────────────────────────────────

def plot_approval_diff(res, tv, lv, path='strat_approval_diff.png'):
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

def plot_weighted(res, tv, lv, path='strat_weighted.png'):
    """Weighted average VSE under four priors over (t, l) space."""
    priors = _weighted_priors(tv, lv)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True, facecolor=BG)
    axes = axes.flatten()
    for ax, (label, w) in zip(axes, priors):
        _style_ax(ax)
        avgs = _weighted_method_scores(res, w)
        vals  = [avgs[m] for m in METHODS]
        labels = list(METHODS.keys())
        clrs  = [COLORS[m] for m in METHODS]
        bars = _plot_nonnegative_or_mark_negative(ax, labels, vals, clrs)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0, 1.18)
        _annotate_stars(ax, bars, vals, label_gap=0.075)

        stars = _tier_star_counts(vals)
        stars_by_method = {m: int(s) for m, s in zip(METHODS.keys(), stars)}
        _summary(
            f"WEIGHTED_VSE | {label.replace(chr(10), ' ')} | "
            + _summary_kv_line(avgs, stars_by_method)
        )

    axes[0].set_ylabel('Weighted average VSE', color=TEXT_C)
    axes[2].set_ylabel('Weighted average VSE', color=TEXT_C)
    fig.suptitle('Weighted Average VSE Under Different Priors on (t, ℓ)',
                 color=TEXT_C, fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

# ── plot 7: weighted improvement over reform baselines ───────────────────────

def _plot_weighted_improvement_against(res, tv, lv, base_method, target_methods, path):
    """Weighted VSE gain relative to a single reform baseline under shared priors."""
    priors = _weighted_priors(tv, lv)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True, facecolor=BG)
    axes = axes.flatten()

    all_vals = []
    for _, weights in priors:
        scores = _weighted_method_scores(res, weights)
        all_vals.extend(scores[name] - scores[base_method] for name in target_methods)

    span = max(all_vals) - min(all_vals)
    pad = max(0.01, 0.2 * max(span, max(abs(v) for v in all_vals)))
    y_min = min(all_vals) - pad
    y_max = max(all_vals) + pad

    for ax, (prior_label, weights) in zip(axes, priors):
        _style_ax(ax)
        scores = _weighted_method_scores(res, weights)
        vals = [scores[name] - scores[base_method] for name in target_methods]
        clrs = [COLORS[name] for name in target_methods]
        bars = ax.bar(target_methods, vals, color=clrs, width=0.65,
                      edgecolor=GRID_C, linewidth=0.5)
        ax.axhline(0, color=MUTED, lw=1, ls='--')
        ax.set_title(prior_label, fontsize=9)
        ax.set_xticklabels(target_methods, rotation=35, ha='right', fontsize=8)
        label_pad = max(0.012, pad * 0.12)
        _label_bars(ax, bars, vals, pad=label_pad)
        ax.set_ylim(y_min, y_max)
        _annotate_stars(ax, bars, vals, label_gap=label_pad + max(0.02, pad * 0.18))

        stars = _tier_star_counts(vals)
        vals_by_method = {m: float(v) for m, v in zip(target_methods, vals)}
        stars_by_method = {m: int(s) for m, s in zip(target_methods, stars)}
        _summary(
            f"WEIGHTED_IMPROVEMENT | baseline={base_method} | prior={prior_label.replace(chr(10), ' ')} | "
            + _summary_kv_line(vals_by_method, stars_by_method)
        )

    axes[0].set_ylabel('Weighted improvement score', color=TEXT_C)
    axes[2].set_ylabel('Weighted improvement score', color=TEXT_C)
    fig.suptitle(
        f'Weighted Improvement vs {base_method}\n'
        '(integral of ΔVSE against each prior; positive = reform target better)',
        color=TEXT_C, fontsize=12, y=1.01
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

def plot_weighted_plurality_improvement(res, tv, lv,
                                        path='strat_weighted_plurality_improvement.png'):
    _plot_weighted_improvement_against(
        res, tv, lv,
        base_method='Plurality',
        target_methods=[m for m in METHODS if m != 'Plurality'],
        path=path,
    )

def plot_weighted_approval_improvement(res, tv, lv,
                                       path='strat_weighted_approval_improvement.png'):
    _plot_weighted_improvement_against(
        res, tv, lv,
        base_method='Approval',
        target_methods=[m for m in METHODS if m not in ('Plurality', 'Approval')],
        path=path,
    )

def plot_weighted_overall_improvement(res, tv, lv,
                                      path='strat_weighted_overall_improvement.png'):
    """Grouped comparison of weighted improvement vs Plurality and Approval."""
    priors = _weighted_priors(tv, lv)
    target_methods = [m for m in METHODS if m not in ('Plurality', 'Approval')]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True, facecolor=BG)
    axes = axes.flatten()

    all_vals = []
    for _, weights in priors:
        scores = _weighted_method_scores(res, weights)
        all_vals.extend(scores[name] - scores['Plurality'] for name in target_methods)
        all_vals.extend(scores[name] - scores['Approval'] for name in target_methods)

    span = max(all_vals) - min(all_vals)
    pad = max(0.01, 0.2 * max(span, max(abs(v) for v in all_vals)))
    y_min = min(all_vals) - pad
    y_max = max(all_vals) + pad
    x = np.arange(len(target_methods))
    width = 0.34

    for ax, (prior_label, weights) in zip(axes, priors):
        _style_ax(ax)
        scores = _weighted_method_scores(res, weights)
        plurality_vals = [scores[name] - scores['Plurality'] for name in target_methods]
        approval_vals = [scores[name] - scores['Approval'] for name in target_methods]

        bars_plurality = ax.bar(
            x - width / 2,
            plurality_vals,
            width=width,
            color=COLORS['Plurality'],
            alpha=0.85,
            edgecolor=GRID_C,
            linewidth=0.5,
            label='vs Plurality',
        )
        bars_approval = ax.bar(
            x + width / 2,
            approval_vals,
            width=width,
            color=COLORS['Approval'],
            alpha=0.85,
            edgecolor=GRID_C,
            linewidth=0.5,
            label='vs Approval',
        )

        ax.axhline(0, color=MUTED, lw=1, ls='--')
        ax.set_title(prior_label, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(target_methods, rotation=35, ha='right', fontsize=8)
        ax.set_ylim(y_min, y_max)

        label_pad = max(0.012, pad * 0.12)
        _label_bars(ax, bars_plurality, plurality_vals, pad=label_pad, fmt='{:.3f}')
        _label_bars(ax, bars_approval, approval_vals, pad=label_pad, fmt='{:.3f}')

        plurality_stars = _tier_star_counts(plurality_vals)
        approval_stars = _tier_star_counts(approval_vals)
        pl_vals_map = {m: float(v) for m, v in zip(target_methods, plurality_vals)}
        ap_vals_map = {m: float(v) for m, v in zip(target_methods, approval_vals)}
        pl_stars_map = {m: int(s) for m, s in zip(target_methods, plurality_stars)}
        ap_stars_map = {m: int(s) for m, s in zip(target_methods, approval_stars)}
        clean_prior = prior_label.replace(chr(10), ' ')
        _summary(
            f"WEIGHTED_OVERALL | baseline=Plurality | prior={clean_prior} | "
            + _summary_kv_line(pl_vals_map, pl_stars_map)
        )
        _summary(
            f"WEIGHTED_OVERALL | baseline=Approval | prior={clean_prior} | "
            + _summary_kv_line(ap_vals_map, ap_stars_map)
        )

    axes[0].set_ylabel('Weighted improvement score', color=TEXT_C)
    axes[2].set_ylabel('Weighted improvement score', color=TEXT_C)
    axes[0].legend(facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=9)
    fig.suptitle(
        'Overall Weighted Improvement by Reform Baseline\n'
        '(grouped bars compare each reform target against Plurality and Approval)',
        color=TEXT_C, fontsize=12, y=1.01
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓  {path}')

# ── plot 8: reform delta heatmaps ─────────────────────────────────────────────

def plot_plurality_delta_heatmap(res, tv, lv, path='strat_plurality_delta_heatmap.png'):
    others = [m for m in METHODS if m != 'Plurality']
    _plot_delta_heatmaps(
        res, tv, lv,
        base_method='Plurality',
        compared_methods=others,
        title='Method VSE minus Plurality\n(green = reform better, red = Plurality better)',
        path=path,
    )

def plot_approval_delta_heatmap(res, tv, lv, path='strat_approval_delta_heatmap.png'):
    others = [m for m in METHODS if m not in ('Plurality', 'Approval')]
    _plot_delta_heatmaps(
        res, tv, lv,
        base_method='Approval',
        compared_methods=others,
        title='Non-Plurality Method VSE minus Approval\n(green = reform better, red = Approval better)',
        path=path,
    )

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    script_t0 = time.time()
    print(f'Grid: {N_T}×{N_L}  |  Trials: {N_TRIALS}  |  '
          f'Voters: {N_VOTERS}  |  Candidates: {N_CANDS}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = os.path.join(LOG_DIR, f'summary_{run_stamp}.txt')
    pvalues_path = os.path.join(LOG_DIR, f'pvalues_{run_stamp}.txt')
    hypothesis_path = os.path.join(LOG_DIR, f'hypothesis_decisions_{run_stamp}.txt')

    SUMMARY_LINES.clear()
    _summary('Satisficing Voter Simulation Summary')
    _summary(f'run_started={datetime.now().isoformat(timespec="seconds")}')
    _summary(f'grid={N_T}x{N_L} | trials={N_TRIALS} | voters={N_VOTERS} | candidates={N_CANDS}')
    _summary(f'strategy_mode={int(USE_STRATEGY)} | strategy_share={STRATEGY_SHARE:.2f} | polling=per_method_honest | condorcet_target=second')
    _summary('')
    t0 = time.time()
    res, trial_vse = run_grid(
        T_VALS,
        L_VALS,
        return_trial_vse=RUN_PAIRED_HYPOTHESIS_TESTS,
        use_strategy=USE_STRATEGY,
        strategy_share=STRATEGY_SHARE,
    )
    sim_elapsed = time.time() - t0

    if RUN_PAIRED_HYPOTHESIS_TESTS and trial_vse is not None:
        print('Running paired-difference hypothesis tests...')
        test_results = analyze_pairwise_hypotheses(
            trial_vse,
            T_VALS,
            L_VALS,
            HYPOTHESIS_PAIRS,
            alpha=HYPOTHESIS_ALPHA,
        )
        report_pairwise_hypotheses(test_results)
        _write_pvalue_log(pvalues_path, test_results, run_stamp)
        _write_hypothesis_decision_log(hypothesis_path, test_results, run_stamp)
        print(f'P-value log saved: {pvalues_path}')
        print(f'Hypothesis decision log saved: {hypothesis_path}')

    print(f'Simulation done in {sim_elapsed:.1f}s\nGenerating plots...')

    plot_t0 = time.time()
    plot_slices(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'strat_slices.png'))
    plot_heatmaps(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'strat_heatmaps.png'))
    plot_robustness(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'strat_robustness.png'))
    plot_scenarios(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'strat_scenarios.png'))
    # plot_scenario_plurality_improvement(
    #     res, T_VALS, L_VALS,
    #     path=os.path.join(OUTPUT_DIR, 'strat_scenario_plurality_improvement.png')
    # )
    # plot_scenario_approval_improvement(
    #     res, T_VALS, L_VALS,
    #     path=os.path.join(OUTPUT_DIR, 'strat_scenario_approval_improvement.png')
    # )
    plot_scenario_overall_improvement(
        res, T_VALS, L_VALS,
        path=os.path.join(OUTPUT_DIR, 'strat_scenario_overall_improvement.png')
    )
    plot_approval_diff(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'strat_approval_diff.png'))
    plot_weighted(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'strat_weighted.png'))
    # plot_weighted_plurality_improvement(
    #     res, T_VALS, L_VALS,
    #     path=os.path.join(OUTPUT_DIR, 'strat_weighted_plurality_improvement.png')
    # )
    # plot_weighted_approval_improvement(
    #     res, T_VALS, L_VALS,
    #     path=os.path.join(OUTPUT_DIR, 'strat_weighted_approval_improvement.png')
    # )
    plot_weighted_overall_improvement(
        res, T_VALS, L_VALS,
        path=os.path.join(OUTPUT_DIR, 'strat_weighted_overall_improvement.png')
    )
    plot_plurality_delta_heatmap(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'strat_plurality_delta_heatmap.png'))
    plot_approval_delta_heatmap(res, T_VALS, L_VALS, path=os.path.join(OUTPUT_DIR, 'strat_approval_delta_heatmap.png'))
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

    _summary('')
    _summary('RUN_TIMING')
    _summary(
        f'total={total_elapsed:.3f}s | simulation={sim_elapsed:.3f}s | plotting={plot_elapsed:.3f}s '
        f'| avg_grid={sim_elapsed / total_grid_points:.6f}s | avg_sim={sim_elapsed / total_sims:.8f}s'
    )
    _summary(f'run_completed={datetime.now().isoformat(timespec="seconds")}')
    _write_summary_log(summary_path)
    print(f'Summary log saved: {summary_path}')

    print('All plots saved.')
