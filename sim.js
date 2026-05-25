/**
 * Satisficing Voter Model — browser simulation
 * Ports the Python logic from vse_sim_1.py to JavaScript.
 * Runs synchronously in chunks via setTimeout to keep the UI responsive.
 */

'use strict';

// ── RNG (seeded Mulberry32) ───────────────────────────────────────────────────
function mulberry32(seed) {
    return function () {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

let rng = mulberry32(42);

function randn() {
    // Box–Muller
    const u = 1 - rng(), v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ── Color palette ─────────────────────────────────────────────────────────────
const METHOD_COLORS = {
    Plurality: '#e05555',
    Approval: '#4ec96a',
    RCV: '#e09944',
    STAR: '#4ab8e0',
    Condorcet: '#9b6be0',
    Score: '#b8e04a',
    Borda: '#d4c94a',
};

// ── Core simulation utilities ─────────────────────────────────────────────────

/** True utilities: (nv × nc) matrix, u[i][c] = -||v_i - p_c||^2 */
function makeElection(nv, nc, nd = 2) {
    const v = Array.from({ length: nv }, () => Array.from({ length: nd }, () => rng() * 2 - 1));
    const c = Array.from({ length: nc }, () => Array.from({ length: nd }, () => rng() * 2 - 1));
    const u = Array.from({ length: nv }, (_, i) =>
        Array.from({ length: nc }, (_, j) => {
            let d = 0;
            for (let k = 0; k < nd; k++) d += (v[i][k] - c[j][k]) ** 2;
            return -d;
        })
    );
    return u;
}

/** Add epistemic noise: ũ_i = t·u_i + (1-t)·η_i */
function addNoise(u, t) {
    if (t >= 1) return u.map(r => r.slice());
    const nv = u.length, nc = u[0].length;
    // compute global mean and std
    let sum = 0, sum2 = 0, n = nv * nc;
    for (let i = 0; i < nv; i++) for (let j = 0; j < nc; j++) { sum += u[i][j]; sum2 += u[i][j] ** 2; }
    const mu = sum / n;
    const sg = Math.sqrt(sum2 / n - mu * mu);
    const safeSg = sg > 1e-9 ? sg : 1.0;

    return u.map(row =>
        row.map(x => {
            const eta = mu + safeSg * randn();
            return t * x + (1 - t) * eta;
        })
    );
}

function Kof(l, nc) { return Math.max(1, Math.ceil(l * nc)); }

/** Returns array of top-K candidate indices per voter (nv × K) */
function topIdx(pu, l) {
    const nc = pu[0].length;
    const K = Kof(l, nc);
    return pu.map(row => {
        const idx = Array.from({ length: nc }, (_, i) => i);
        idx.sort((a, b) => row[b] - row[a]);
        return idx.slice(0, K);
    });
}

function vse(swW, swRand, swOpt) {
    const d = swOpt - swRand;
    return d > 1e-9 ? (swW - swRand) / d : 1;
}

// ── Social welfare helpers ────────────────────────────────────────────────────

function colMeans(u) {
    const nv = u.length, nc = u[0].length;
    const means = new Array(nc).fill(0);
    for (let i = 0; i < nv; i++) for (let j = 0; j < nc; j++) means[j] += u[i][j];
    return means.map(x => x / nv);
}

function argmax(arr) {
    let best = 0;
    for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
    return best;
}

function clip(x, lo, hi) {
    return Math.min(hi, Math.max(lo, x));
}

function bincount(values, length) {
    const out = new Array(length).fill(0);
    for (const v of values) out[v]++;
    return out;
}

function strategicVoterMask(nv, share) {
    const p = clip(share, 0, 1);
    const mask = new Array(nv);
    for (let i = 0; i < nv; i++) mask[i] = rng() < p;
    return mask;
}

function frontAndTarget(polls, useThird = false) {
    const order = Array.from({ length: polls.length }, (_, i) => i).sort((a, b) => polls[b] - polls[a]);
    const front = order[0];
    if (order.length === 1) return [front, front];
    if (useThird && order.length >= 3) return [front, order[2]];
    return [front, order[1]];
}

function topKMask(pu, l) {
    const nv = pu.length;
    const nc = pu[0].length;
    const tidxAll = topIdx(pu, l);
    const K = tidxAll[0].length;
    const considered = Array.from({ length: nv }, () => new Array(nc).fill(false));
    for (let i = 0; i < nv; i++) {
        for (const c of tidxAll[i]) considered[i][c] = true;
    }
    return { tidxAll, K, considered };
}

function scoreRowFromFrontTarget(row, front, target, topRank = 5) {
    const pref = row[target] > row[front] ? target : front;
    const other = pref === target ? front : target;
    const hi = row[pref];
    const lo = row[other];
    if (Math.abs(hi - lo) < 1e-12) {
        return row.map(v => (v >= hi ? topRank : 0));
    }
    const out = new Array(row.length).fill(0);
    for (let c = 0; c < row.length; c++) {
        const scaled = (topRank + 0.99) * (row[c] - lo) / (hi - lo);
        out[c] = clip(Math.floor(scaled), 0, topRank);
    }
    return out;
}

function irvWinnerFromBallots(ballots, nc) {
    const nv = ballots.length;
    const K = ballots[0].length;
    const remaining = new Array(nc).fill(true);
    for (let round = 0; round < nc - 1; round++) {
        const active = [];
        for (let c = 0; c < nc; c++) if (remaining[c]) active.push(c);
        if (active.length === 1) return active[0];

        const fp = new Array(nc).fill(0);
        for (let i = 0; i < nv; i++) {
            for (let k = 0; k < K; k++) {
                const cand = ballots[i][k];
                if (remaining[cand]) {
                    fp[cand]++;
                    break;
                }
            }
        }
        const total = fp.reduce((s, x) => s + x, 0);
        const majority = active.find(c => fp[c] > total / 2);
        if (majority !== undefined) return majority;

        let elim = active[0];
        for (const c of active) if (fp[c] < fp[elim]) elim = c;
        remaining[elim] = false;
    }
    return remaining.findIndex(x => x);
}

function rankMat(pu, l) {
    const nv = pu.length;
    const nc = pu[0].length;
    const tidxAll = topIdx(pu, l);
    const K = tidxAll[0].length;
    const rm = Array.from({ length: nv }, () => new Array(nc).fill(K));
    for (let i = 0; i < nv; i++) {
        for (let k = 0; k < K; k++) rm[i][tidxAll[i][k]] = k;
    }
    return { rm, K, tidxAll };
}

function condorcetWinnerFromRankMat(rm) {
    const nv = rm.length;
    const nc = rm[0].length;
    const pairwise = Array.from({ length: nc }, () => new Array(nc).fill(0));
    for (let a = 0; a < nc; a++) {
        for (let b = a + 1; b < nc; b++) {
            let wa = 0;
            let wb = 0;
            for (let i = 0; i < nv; i++) {
                if (rm[i][a] < rm[i][b]) wa++;
                else if (rm[i][b] < rm[i][a]) wb++;
            }
            pairwise[a][b] = wa;
            pairwise[b][a] = wb;
        }
    }
    const worstDefeat = pairwise.map((row, a) => {
        let worst = 0;
        for (let b = 0; b < nc; b++) {
            if (b === a) continue;
            worst = Math.max(worst, pairwise[b][a] - row[b]);
        }
        return worst;
    });
    return { winner: worstDefeat.indexOf(Math.min(...worstDefeat)), pairwise };
}

// ── Voting methods ────────────────────────────────────────────────────────────

function pluralityWinner(pu, useStrategy = false, strategyShare = 1.0) {
    const nv = pu.length, nc = pu[0].length;
    const top1 = pu.map(row => argmax(row));
    const polls = bincount(top1, nc);
    if (!useStrategy) return { winner: argmax(polls), polls };

    const [front, target] = frontAndTarget(polls);
    const stratMask = strategicVoterMask(nv, strategyShare);
    const top1Strat = top1.slice();
    for (let i = 0; i < nv; i++) {
        if (!stratMask[i]) continue;
        top1Strat[i] = pu[i][target] > pu[i][front] ? target : front;
    }
    const counts = bincount(top1Strat, nc);
    return { winner: argmax(counts), polls };
}

function approvalWinner(pu, l, useStrategy = false, strategyShare = 1.0) {
    const nv = pu.length;
    const nc = pu[0].length;
    const { tidxAll, considered } = topKMask(pu, l);
    const approvals = Array.from({ length: nv }, () => new Array(nc).fill(false));
    for (let i = 0; i < nv; i++) {
        const row = pu[i];
        const globalMean = row.reduce((s, x) => s + x, 0) / nc;
        for (let c = 0; c < nc; c++) {
            approvals[i][c] = considered[i][c] && row[c] > globalMean;
        }
        if (!approvals[i].some(Boolean)) approvals[i][tidxAll[i][0]] = true;
    }
    const polls = new Array(nc).fill(0);
    for (let i = 0; i < nv; i++) {
        for (let c = 0; c < nc; c++) if (approvals[i][c]) polls[c]++;
    }
    if (useStrategy) {
        const [front, target] = frontAndTarget(polls);
        const stratMask = strategicVoterMask(nv, strategyShare);
        for (let i = 0; i < nv; i++) {
            if (!stratMask[i]) continue;
            const pivot = (pu[i][front] + pu[i][target]) / 2;
            const rowApprovals = new Array(nc).fill(false);
            for (let c = 0; c < nc; c++) {
                rowApprovals[c] = considered[i][c] && pu[i][c] >= pivot;
            }
            const prefersTarget = pu[i][target] > pu[i][front];
            if (prefersTarget) {
                rowApprovals[front] = false;
                rowApprovals[target] = considered[i][target];
            } else {
                rowApprovals[target] = false;
                rowApprovals[front] = considered[i][front];
            }
            if (!rowApprovals.some(Boolean)) rowApprovals[tidxAll[i][0]] = true;
            approvals[i] = rowApprovals;
        }
    }

    const totals = new Array(nc).fill(0);
    for (let i = 0; i < nv; i++) {
        for (let c = 0; c < nc; c++) if (approvals[i][c]) totals[c]++;
    }
    return { winner: argmax(totals), polls };
}

function bordaWinner(pu, l, useStrategy = false, strategyShare = 1.0) {
    const nv = pu.length;
    const nc = pu[0].length;
    const tidxAll = topIdx(pu, l);
    const K = tidxAll[0].length;
    const pts = Array.from({ length: K }, (_, i) => K - i);
    const honestScores = new Array(nc).fill(0);
    for (let i = 0; i < nv; i++) {
        for (let k = 0; k < K; k++) honestScores[tidxAll[i][k]] += pts[k];
    }
    const polls = honestScores.slice();
    if (!useStrategy) return { winner: argmax(honestScores), polls };

    const [front, target] = frontAndTarget(polls);
    const stratMask = strategicVoterMask(nv, strategyShare);
    const pollOrderDesc = Array.from({ length: nc }, (_, i) => i).sort((a, b) => polls[b] - polls[a]);
    const scores = new Array(nc).fill(0);
    for (let i = 0; i < nv; i++) {
        let order = tidxAll[i].slice();
        if (stratMask[i] && K > 1 && order.includes(front) && order.includes(target)) {
            const middle = pollOrderDesc.filter(c => order.includes(c) && c !== front && c !== target).reverse();
            order = pu[i][target] > pu[i][front] ? [target, ...middle, front] : [front, ...middle, target];
        }
        for (let k = 0; k < K; k++) scores[order[k]] += pts[k];
    }
    return { winner: argmax(scores), polls };
}

function globalScoreBallots(pu, l) {
    const nv = pu.length, nc = pu[0].length;
    const tidxAll = topIdx(pu, l);
    const ballots = Array.from({ length: nv }, () => new Array(nc).fill(0));
    for (let i = 0; i < nv; i++) {
        const row = pu[i];
        const lo = Math.min(...row), hi = Math.max(...row);
        const flat = hi - lo < 1e-9;
        const denom = flat ? 1 : hi - lo;
        for (const c of tidxAll[i]) {
            ballots[i][c] = flat ? 5 : 5 * (row[c] - lo) / denom;
        }
    }
    return ballots;
}

function scoreWinner(pu, l, useStrategy = false, strategyShare = 1.0) {
    const nv = pu.length;
    const nc = pu[0].length;
    const ballots = globalScoreBallots(pu, l);
    const polls = new Array(nc).fill(0);
    for (let i = 0; i < nv; i++) {
        for (let c = 0; c < nc; c++) polls[c] += ballots[i][c];
    }
    if (useStrategy) {
        const [front, target] = frontAndTarget(polls);
        const { tidxAll, considered } = topKMask(pu, l);
        const stratMask = strategicVoterMask(nv, strategyShare);
        for (let i = 0; i < nv; i++) {
            if (!stratMask[i]) continue;
            const row = scoreRowFromFrontTarget(pu[i], front, target, 5);
            const stratRow = new Array(nc).fill(0);
            for (let c = 0; c < nc; c++) {
                if (considered[i][c]) stratRow[c] = row[c];
            }
            if (!stratRow.some(x => x > 0)) stratRow[tidxAll[i][0]] = 5;
            ballots[i] = stratRow;
        }
    }

    const totals = new Array(nc).fill(0);
    for (let i = 0; i < nv; i++) {
        for (let c = 0; c < nc; c++) totals[c] += ballots[i][c];
    }
    return { winner: argmax(totals), polls, ballots };
}

function starWinner(pu, l, useStrategy = false, strategyShare = 1.0) {
    const { ballots, polls } = scoreWinner(pu, l, useStrategy, strategyShare);
    const nv = ballots.length;
    const nc = ballots[0].length;
    const totals = new Array(nc).fill(0);
    for (let i = 0; i < nv; i++) {
        for (let c = 0; c < nc; c++) totals[c] += ballots[i][c];
    }
    const sorted = Array.from({ length: nc }, (_, i) => i).sort((a, b) => totals[b] - totals[a]);
    const [f1, f2] = sorted;
    let w1 = 0;
    let w2 = 0;
    for (let i = 0; i < nv; i++) {
        if (ballots[i][f1] > ballots[i][f2]) w1++;
        else if (ballots[i][f2] > ballots[i][f1]) w2++;
    }
    return { winner: w1 >= w2 ? f1 : f2, polls };
}

function condorcetWinner(pu, l, useStrategy = false, strategyShare = 1.0) {
    const nv = pu.length;
    const { rm, K, tidxAll } = rankMat(pu, l);
    const honest = condorcetWinnerFromRankMat(rm);
    const nc = rm[0].length;
    const polls = new Array(nc).fill(0);
    for (let a = 0; a < nc; a++) {
        for (let b = 0; b < nc; b++) {
            if (a !== b && honest.pairwise[a][b] > honest.pairwise[b][a]) polls[a]++;
        }
    }
    if (!useStrategy) return { winner: honest.winner, polls };

    const [front, target] = frontAndTarget(polls);
    const stratMask = strategicVoterMask(nv, strategyShare);
    const pollOrder = Array.from({ length: nc }, (_, i) => i).sort((a, b) => polls[b] - polls[a]);

    for (let i = 0; i < nv; i++) {
        if (!stratMask[i]) continue;
        let order = tidxAll[i].slice();
        if (pu[i][target] > pu[i][front] && order.includes(front) && order.includes(target)) {
            const others = pollOrder.filter(c => order.includes(c) && c !== front && c !== target);
            const notTooBad = Math.min(pu[i][front], pu[i][target]);
            const decent = others.filter(c => pu[i][c] >= notTooBad).sort((a, b) => pu[i][b] - pu[i][a]);
            const bad = others.filter(c => pu[i][c] < notTooBad).sort((a, b) => pu[i][b] - pu[i][a]);
            order = [...decent, target, ...bad, front];
        } else {
            const others = pollOrder.filter(c => order.includes(c) && c !== front).sort((a, b) => pu[i][b] - pu[i][a]);
            order = [front, ...others];
        }

        rm[i].fill(K);
        for (let k = 0; k < order.length; k++) rm[i][order[k]] = k;
    }

    return { winner: condorcetWinnerFromRankMat(rm).winner, polls };
}

function irvWinner(pu, l, useStrategy = false, strategyShare = 1.0) {
    const nv = pu.length;
    const nc = pu[0].length;
    const ballots = topIdx(pu, l);
    const K = ballots[0].length;
    const polls = new Array(nc).fill(0);
    for (let i = 0; i < nv; i++) polls[ballots[i][0]]++;
    if (!useStrategy) return { winner: irvWinnerFromBallots(ballots, nc), polls };

    const [front, target] = frontAndTarget(polls);
    const pollOrderDesc = Array.from({ length: nc }, (_, i) => i).sort((a, b) => polls[b] - polls[a]);
    const stratMask = strategicVoterMask(nv, strategyShare);
    for (let i = 0; i < nv; i++) {
        if (!stratMask[i]) continue;
        const order = ballots[i].slice();
        const winnerQ = pu[i][front];
        const targQ = pu[i][target];

        let byPollLoserToWinner = pollOrderDesc.slice().reverse().filter(c => order.includes(c) && c !== front);
        const stratOrder = [];
        if (order.includes(target) && targQ > winnerQ) {
            stratOrder.push(target);
            byPollLoserToWinner = byPollLoserToWinner.filter(c => c !== target);
        }

        for (const c of byPollLoserToWinner) if (pu[i][c] > winnerQ) stratOrder.push(c);
        if (order.includes(front)) stratOrder.push(front);
        for (const c of byPollLoserToWinner) if (pu[i][c] <= winnerQ) stratOrder.push(c);

        const missing = order.filter(c => !stratOrder.includes(c)).sort((a, b) => pu[i][b] - pu[i][a]);
        if (missing.length) stratOrder.push(...missing);

        ballots[i] = stratOrder.slice(0, K);
    }

    return { winner: irvWinnerFromBallots(ballots, nc), polls };
}

// ── Method registry ───────────────────────────────────────────────────────────

function buildMethods(enabled, options = {}) {
    const useStrategy = Boolean(options.useStrategy);
    const strategyShare = Number.isFinite(options.strategyShare) ? options.strategyShare : 1.0;
    const all = {
        Plurality: (pu, l) => pluralityWinner(pu, useStrategy, strategyShare).winner,
        Approval: (pu, l) => approvalWinner(pu, l, useStrategy, strategyShare).winner,
        RCV: (pu, l) => irvWinner(pu, l, useStrategy, strategyShare).winner,
        STAR: (pu, l) => starWinner(pu, l, useStrategy, strategyShare).winner,
        Condorcet: (pu, l) => condorcetWinner(pu, l, useStrategy, strategyShare).winner,
        Score: (pu, l) => scoreWinner(pu, l, useStrategy, strategyShare).winner,
        Borda: (pu, l) => bordaWinner(pu, l, useStrategy, strategyShare).winner,
    };
    const out = {};
    for (const k of Object.keys(all)) if (enabled[k]) out[k] = all[k];
    return out;
}

// ── Grid runner (async-friendly) ──────────────────────────────────────────────

let simAborted = false;
let isSimRunning = false;
let lastSimResult = null;

const HYPOTHESIS_ALPHA = 0.01;
let hypothesisTableSort = { key: 'pooledMeanP', dir: 'asc' };

function orderedEnabledMethods(methods) {
    return Object.keys(METHOD_COLORS).filter(name => methods[name]);
}

function erfApprox(x) {
    const sign = x < 0 ? -1 : 1;
    const ax = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * ax);
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
    const y = 1 - poly * Math.exp(-ax * ax);
    return sign * y;
}

function normalCdf(x) {
    return 0.5 * (1 + erfApprox(x / Math.sqrt(2)));
}

function oneSidedMeanDiffTest(deltas) {
    const n = deltas.length;
    if (n < 2) {
        return { mean: NaN, sd: NaN, se: NaN, z: NaN, p: NaN, lowerCi: NaN, nearZeroVar: true };
    }

    let sum = 0;
    for (const d of deltas) sum += d;
    const mean = sum / n;

    let ss = 0;
    for (const d of deltas) ss += (d - mean) ** 2;
    const sd = Math.sqrt(ss / (n - 1));
    const se = sd / Math.sqrt(n);

    if (sd < 1e-12) {
        return {
            mean,
            sd,
            se,
            z: mean > 0 ? Infinity : -Infinity,
            p: mean > 0 ? 0 : 1,
            lowerCi: mean,
            nearZeroVar: true,
        };
    }

    const z = mean / se;
    const p = 1 - normalCdf(z);  // one-sided for H1: mean > 0
    const zCrit = 1.6448536269514722;  // lower bound for one-sided 95%
    const lowerCi = mean - zCrit * se;
    return { mean, sd, se, z, p: Math.max(0, Math.min(1, p)), lowerCi, nearZeroVar: false };
}

function logChoose(n, k) {
    if (k < 0 || k > n) return -Infinity;
    if (k === 0 || k === n) return 0;
    const kk = Math.min(k, n - k);
    let out = 0;
    for (let i = 1; i <= kk; i++) out += Math.log(n - kk + i) - Math.log(i);
    return out;
}

function binomUpperTailHalf(n, kMin) {
    if (kMin <= 0) return 1;
    if (kMin > n) return 0;
    let sum = 0;
    const ln2 = Math.log(2);
    for (let k = kMin; k <= n; k++) {
        sum += Math.exp(logChoose(n, k) - n * ln2);
    }
    return Math.max(0, Math.min(1, sum));
}

function oneSidedSignTest(deltas) {
    let pos = 0;
    let neg = 0;
    let ties = 0;
    for (const d of deltas) {
        if (Math.abs(d) < 1e-12) ties++;
        else if (d > 0) pos++;
        else neg++;
    }
    const nEff = pos + neg;
    const p = nEff > 0 ? binomUpperTailHalf(nEff, pos) : 1;
    return { pos, neg, ties, nEff, p };
}

function bonferroniAdjust(pvals) {
    const m = pvals.length;
    if (m === 0) return [];
    return pvals.map(p => Math.min(1, p * m));
}

function analyzeHypotheses(trialVse, tVals, lVals, methods) {
    if (!trialVse) return null;

    const enabledMethodNames = orderedEnabledMethods(methods);
    const enabledPairs = [];
    for (const x of enabledMethodNames) {
        for (const y of enabledMethodNames) {
            if (x !== y) enabledPairs.push([x, y]);
        }
    }
    if (!enabledPairs.length) {
        return {
            alpha: HYPOTHESIS_ALPHA,
            correction: 'bonferroni',
            pairs: [],
            perGridTests: 0,
            perPair: {},
            methodNames: enabledMethodNames,
        };
    }

    const ng = lVals.length;
    const perPair = {};
    const allMeanP = [];
    const allSignP = [];
    const refs = [];

    for (const [x, y] of enabledPairs) {
        const pooled = [];
        const cells = [];
        const meanPGrid = Array.from({ length: ng }, () => new Float64Array(ng));
        const signPGrid = Array.from({ length: ng }, () => new Float64Array(ng));
        const meanDeltaGrid = Array.from({ length: ng }, () => new Float64Array(ng));

        for (let li = 0; li < ng; li++) {
            for (let ti = 0; ti < ng; ti++) {
                const xVals = trialVse[x][li][ti];
                const yVals = trialVse[y][li][ti];
                const deltas = new Float64Array(xVals.length);
                for (let i = 0; i < xVals.length; i++) {
                    deltas[i] = xVals[i] - yVals[i];
                    pooled.push(deltas[i]);
                }

                const meanStats = oneSidedMeanDiffTest(deltas);
                const signStats = oneSidedSignTest(deltas);

                meanPGrid[li][ti] = meanStats.p;
                signPGrid[li][ti] = signStats.p;
                meanDeltaGrid[li][ti] = meanStats.mean;

                allMeanP.push(meanStats.p);
                allSignP.push(signStats.p);
                refs.push({ pair: `${x}>${y}`, li, ti });
                cells.push({ li, ti, t: tVals[ti], l: lVals[li], meanStats, signStats });
            }
        }

        perPair[`${x}>${y}`] = {
            x,
            y,
            cells,
            meanPGrid,
            signPGrid,
            meanDeltaGrid,
            pooledMean: oneSidedMeanDiffTest(pooled),
            pooledSign: oneSidedSignTest(pooled),
        };
    }

    const meanAdj = bonferroniAdjust(allMeanP);
    const signAdj = bonferroniAdjust(allSignP);
    for (let i = 0; i < refs.length; i++) {
        const ref = refs[i];
        const info = perPair[ref.pair];
        if (!info.meanPAdjGrid) {
            info.meanPAdjGrid = Array.from({ length: ng }, () => new Float64Array(ng));
            info.signPAdjGrid = Array.from({ length: ng }, () => new Float64Array(ng));
        }
        info.meanPAdjGrid[ref.li][ref.ti] = meanAdj[i];
        info.signPAdjGrid[ref.li][ref.ti] = signAdj[i];
    }

    return {
        alpha: HYPOTHESIS_ALPHA,
        correction: 'bonferroni',
        pairs: enabledPairs,
        perPair,
        perGridTests: allMeanP.length,
        methodNames: enabledMethodNames,
    };
}

async function runGrid(params, onProgress) {
    const { nv, nc, ntr, ng, nd, methods } = params;
    const methodNames = Object.keys(methods);
    const tVals = linspace(0, 1, ng);
    const lVals = linspace(1 / nc, 1, ng);
    const res = {};
    for (const m of methodNames) res[m] = Array.from({ length: ng }, () => new Float64Array(ng));
    const trialVse = {};
    for (const m of methodNames) {
        trialVse[m] = Array.from({ length: ng }, () =>
            Array.from({ length: ng }, () => new Float64Array(ntr))
        );
    }

    simAborted = false;
    const total = ng * ng;
    let done = 0;
    const t0 = performance.now();

    for (let ti = 0; ti < ng; ti++) {
        for (let li = 0; li < ng; li++) {
            if (simAborted) return null;

            const t = tVals[ti], l = lVals[li];
            const acc = {};
            for (const m of methodNames) acc[m] = 0;

            for (let tr = 0; tr < ntr; tr++) {
                if (simAborted) return null;
                const u = makeElection(nv, nc, nd);
                const pu = addNoise(u, t);
                const cm = colMeans(u);
                const swRand = cm.reduce((s, x) => s + x, 0) / nc;
                const swOpt = Math.max(...cm);
                for (const [name, fn] of Object.entries(methods)) {
                    const w = fn(pu, l);
                    const vseVal = vse(cm[w], swRand, swOpt);
                    acc[name] += vseVal;
                    trialVse[name][li][ti][tr] = vseVal;
                }
            }

            for (const m of methodNames) res[m][li][ti] = acc[m] / ntr;

            done++;
            const elapsed = (performance.now() - t0) / 1000;
            const eta = (total - done) * (elapsed / done);
            onProgress(done, total, elapsed, eta);

            // yield to browser every 4 grid points
            if (done % 4 === 0) await sleep(0);
        }
    }
    return { res, tVals, lVals, trialVse };
}

function fmtP(p) {
    if (!Number.isFinite(p)) return 'NA';
    if (p < 1e-4) return p.toExponential(2);
    return p.toFixed(4);
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderHypothesisTests(state) {
    const summaryEl = document.getElementById('hypothesis-summary');
    const tableWrap = document.getElementById('hypothesis-table-wrap');
    const matrixWrap = document.getElementById('hypothesis-matrix-wrap');
    const detailEl = document.getElementById('hypothesis-detail');
    const emptyEl = document.getElementById('hypothesis-empty');

    if (!summaryEl || !tableWrap || !matrixWrap || !detailEl || !emptyEl) return;

    const analysis = analyzeHypotheses(state.trialVse, state.tVals, state.lVals, state.methods);
    if (!analysis || analysis.pairs.length === 0) {
        matrixWrap.innerHTML = '';
        tableWrap.innerHTML = '';
        detailEl.innerHTML = '';
        summaryEl.textContent = '';
        emptyEl.style.display = 'block';
        emptyEl.textContent = 'Enable at least two methods to see pairwise evidence between methods.';
        return;
    }

    emptyEl.style.display = 'none';
    const modeText = state.useStrategy ? 'Strategic mode (all voters strategic)' : 'Honest mode';
    summaryEl.textContent = `${modeText}. Alpha = ${analysis.alpha.toFixed(2)}. The matrix below compares every enabled method against every other enabled method. Each cell answers the directional question “is the row method better than the column method?” using the pooled evidence, with Bonferroni-adjusted per-grid counts shown in the detailed table.`;

    const rows = [];
    for (const [x, y] of analysis.pairs) {
        const key = `${x}>${y}`;
        const info = analysis.perPair[key];
        const ng = state.tVals.length;
        let sigMeanAdj = 0;
        let sigSignAdj = 0;
        let sigMeanNom = 0;
        for (let li = 0; li < ng; li++) {
            for (let ti = 0; ti < ng; ti++) {
                if (info.meanPGrid[li][ti] < analysis.alpha) sigMeanNom++;
                if (info.meanPAdjGrid[li][ti] < analysis.alpha) sigMeanAdj++;
                if (info.signPAdjGrid[li][ti] < analysis.alpha) sigSignAdj++;
            }
        }

        const pooledMeanDelta = info.pooledMean.mean;
        const pooledMeanSig = info.pooledMean.p < analysis.alpha;
        const pooledSignSig = info.pooledSign.p < analysis.alpha;

        let takeaway = `No clear evidence that ${x} is better than ${y}.`;
        let takeawayClass = 'sig-no';
        if (pooledMeanDelta > 0 && pooledMeanSig && pooledSignSig && sigMeanAdj > 0) {
            takeaway = `Evidence supports ${x} is better than ${y}.`;
            takeawayClass = 'sig-yes';
        } else if (pooledMeanDelta > 0 && (pooledMeanSig || pooledSignSig || sigMeanAdj > 0)) {
            takeaway = `Evidence is mixed or weak that ${x} is better than ${y}.`;
            takeawayClass = 'sig-mixed';
        }

        rows.push({
            x,
            y,
            label: `${x} > ${y}`,
            pooledMeanDelta,
            pooledMeanP: info.pooledMean.p,
            pooledSignP: info.pooledSign.p,
            pooledMeanSig,
            pooledSignSig,
            sigMeanNom,
            sigMeanAdj,
            sigSignAdj,
            totalCells: ng * ng,
            takeaway,
            takeawayClass,
        });
    }

    const rowByKey = Object.fromEntries(rows.map(row => [`${row.x}>${row.y}`, row]));
    const methodNames = analysis.methodNames;
    const detailText = row => `${row.takeaway} Pooled mean Δ = ${Number.isFinite(row.pooledMeanDelta) ? row.pooledMeanDelta.toFixed(5) : 'NA'}, pooled p (mean-diff) = ${fmtP(row.pooledMeanP)}, pooled p (sign test) = ${fmtP(row.pooledSignP)}, Bonferroni-significant cells = ${row.sigMeanAdj}/${row.totalCells}.`;
    const tooltipHtml = row => `
        <div class="pairwise-hover-title">${escapeHtml(row.x)} vs ${escapeHtml(row.y)}</div>
        <div class="pairwise-hover-body">${escapeHtml(row.takeaway)}</div>
        <div class="pairwise-hover-metrics">
            <span>Pooled mean Δ: ${Number.isFinite(row.pooledMeanDelta) ? row.pooledMeanDelta.toFixed(5) : 'NA'}</span>
            <span>Mean-diff p: ${fmtP(row.pooledMeanP)}</span>
            <span>Sign-test p: ${fmtP(row.pooledSignP)}</span>
            <span>Bonferroni cells: ${row.sigMeanAdj}/${row.totalCells}</span>
        </div>
    `;

    matrixWrap.innerHTML = `
        <div class="hypothesis-color-legend" aria-label="Pairwise evidence color legend">
            <span class="legend-title">Matrix Colors:</span>
            <span class="legend-item"><span class="legend-swatch yes"></span>Green = Evidence supports row &gt; column</span>
            <span class="legend-item"><span class="legend-swatch mixed"></span>Yellow = Mixed or weak evidence</span>
            <span class="legend-item"><span class="legend-swatch no"></span>Red = No clear evidence</span>
        </div>
        <div class="matrix-scroll">
            <table class="pairwise-matrix">
                <thead>
                    <tr>
                        <th>Row &gt; Col</th>
                        ${methodNames.map(name => `<th>${name}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${methodNames.map(rowName => `
                        <tr>
                            <th>${rowName}</th>
                            ${methodNames.map(colName => {
        if (rowName === colName) {
            return '<td class="pairwise-diag">-</td>';
        }
        const row = rowByKey[`${rowName}>${colName}`];
        const cellClass = row ? row.takeawayClass : 'sig-no';
        const shortLabel = cellClass === 'sig-yes' ? 'Yes' : (cellClass === 'sig-mixed' ? 'Mixed' : 'No');
        const explanation = row ? detailText(row) : `No result for ${rowName} > ${colName}.`;
        const hoverCard = row ? tooltipHtml(row) : `<div class="pairwise-hover-title">${escapeHtml(rowName)} vs ${escapeHtml(colName)}</div><div class="pairwise-hover-body">No result available for this comparison.</div>`;
        return `
                                    <td class="pairwise-cell-slot">
                                        <button
                                            type="button"
                                            class="pairwise-cell ${cellClass}"
                                            data-detail="${escapeHtml(explanation)}"
                                        >${shortLabel}</button>
                                        <div class="pairwise-hover-template" hidden>${hoverCard}</div>
                                    </td>
                                `;
    }).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
        <div class="hypothesis-hover-card" aria-hidden="true"></div>
    `;

    detailEl.innerHTML = '<strong>Hover, focus, or tap a matrix cell</strong> to see a plain-language explanation here.';
    const hoverCardEl = matrixWrap.querySelector('.hypothesis-hover-card');
    const positionHoverCard = cell => {
        if (!hoverCardEl) return;
        hoverCardEl.style.left = '0px';
        hoverCardEl.style.top = '0px';
        hoverCardEl.classList.add('is-visible');

        const rect = cell.getBoundingClientRect();
        const cardRect = hoverCardEl.getBoundingClientRect();
        const gap = 12;
        const maxLeft = Math.max(gap, window.innerWidth - cardRect.width - gap);
        const centeredLeft = rect.left + rect.width / 2 - cardRect.width / 2;
        const left = Math.min(Math.max(gap, centeredLeft), maxLeft);
        const placeAbove = rect.top >= cardRect.height + gap * 2;
        const top = placeAbove ? rect.top - cardRect.height - gap : rect.bottom + gap;

        hoverCardEl.style.left = `${left}px`;
        hoverCardEl.style.top = `${Math.max(gap, top)}px`;
        hoverCardEl.dataset.side = placeAbove ? 'top' : 'bottom';
    };
    const hideHoverCard = () => {
        if (!hoverCardEl) return;
        hoverCardEl.classList.remove('is-visible');
        hoverCardEl.setAttribute('aria-hidden', 'true');
    };

    matrixWrap.querySelectorAll('.pairwise-cell').forEach(cell => {
        const update = () => {
            if (detailEl) detailEl.textContent = cell.dataset.detail || '';
            if (!hoverCardEl) return;
            const template = cell.parentElement?.querySelector('.pairwise-hover-template');
            if (!template) return;
            hoverCardEl.innerHTML = template.innerHTML;
            hoverCardEl.setAttribute('aria-hidden', 'false');
            positionHoverCard(cell);
        };
        cell.addEventListener('mouseenter', update);
        cell.addEventListener('focus', update);
        cell.addEventListener('click', update);
        cell.addEventListener('mouseleave', hideHoverCard);
        cell.addEventListener('blur', hideHoverCard);
    });

    matrixWrap.addEventListener('mouseleave', hideHoverCard);

    const sortColumns = [
        { key: 'label', label: 'Hypothesis', type: 'string' },
        { key: 'pooledMeanDelta', label: 'Pooled mean Δ', type: 'number' },
        { key: 'pooledMeanP', label: 'Pooled p (mean-diff)', type: 'number' },
        { key: 'pooledSignP', label: 'Pooled p (sign test)', type: 'number' },
        { key: 'sigMeanNom', label: 'Per-grid nominal sig', type: 'number' },
        { key: 'sigMeanAdj', label: 'Per-grid Bonf sig (mean-diff)', type: 'number' },
        { key: 'sigSignAdj', label: 'Per-grid Bonf sig (sign)', type: 'number' },
    ];

    const sortedRows = () => {
        const active = sortColumns.find(c => c.key === hypothesisTableSort.key) || sortColumns[0];
        const dirMul = hypothesisTableSort.dir === 'asc' ? 1 : -1;
        const toNumber = v => (Number.isFinite(v) ? v : (hypothesisTableSort.dir === 'asc' ? Infinity : -Infinity));
        return [...rows].sort((a, b) => {
            if (active.type === 'string') {
                return dirMul * String(a[active.key]).localeCompare(String(b[active.key]));
            }
            const av = toNumber(a[active.key]);
            const bv = toNumber(b[active.key]);
            if (av === bv) return a.label.localeCompare(b.label);
            return dirMul * (av - bv);
        });
    };

    const renderHypothesisTable = () => {
        const rowsToRender = sortedRows();
        tableWrap.innerHTML = `
            <table class="result-table">
                <thead>
                    <tr>
                        ${sortColumns.map(col => {
            const active = col.key === hypothesisTableSort.key;
            const arrow = active ? (hypothesisTableSort.dir === 'asc' ? '▲' : '▼') : '↕';
            return `
                                <th>
                                    <button
                                        type="button"
                                        class="sortable-th ${active ? 'is-active' : ''}"
                                        data-sort-key="${col.key}"
                                        aria-label="Sort by ${escapeHtml(col.label)}"
                                    >
                                        <span>${escapeHtml(col.label)}</span>
                                        <span class="sort-arrow" aria-hidden="true">${arrow}</span>
                                    </button>
                                </th>
                            `;
        }).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${rowsToRender.map(r => `
                        <tr>
                            <td>${escapeHtml(r.label)}</td>
                            <td>${Number.isFinite(r.pooledMeanDelta) ? r.pooledMeanDelta.toFixed(5) : 'NA'}</td>
                            <td class="${r.pooledMeanSig ? 'sig-yes' : 'sig-no'}">${fmtP(r.pooledMeanP)}</td>
                            <td class="${r.pooledSignSig ? 'sig-yes' : 'sig-no'}">${fmtP(r.pooledSignP)}</td>
                            <td>${r.sigMeanNom}/${r.totalCells}</td>
                            <td>${r.sigMeanAdj}/${r.totalCells}</td>
                            <td>${r.sigSignAdj}/${r.totalCells}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;

        tableWrap.querySelectorAll('.sortable-th').forEach(btn => {
            btn.addEventListener('click', () => {
                const key = btn.dataset.sortKey;
                if (!key) return;
                if (hypothesisTableSort.key === key) {
                    hypothesisTableSort.dir = hypothesisTableSort.dir === 'asc' ? 'desc' : 'asc';
                } else {
                    hypothesisTableSort.key = key;
                    hypothesisTableSort.dir = 'asc';
                }
                renderHypothesisTable();
            });
        });
    };

    renderHypothesisTable();
}

function linspace(a, b, n) {
    if (n === 1) return [a];
    return Array.from({ length: n }, (_, i) => a + i * (b - a) / (n - 1));
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Plotting ──────────────────────────────────────────────────────────────────

Chart.defaults.color = '#888';
Chart.defaults.borderColor = '#2a2a2a';

const DARK_PLUGIN = {
    id: 'darkBackground',
    beforeDraw(chart) {
        chart.ctx.save();
        chart.ctx.fillStyle = '#1d1d1d';
        chart.ctx.fillRect(0, 0, chart.width, chart.height);
        chart.ctx.restore();
    }
};

const BAR_VALUE_LABELS_PLUGIN = {
    id: 'barValueLabels',
    afterDatasetsDraw(chart, args, pluginOptions) {
        if (chart.config.type !== 'bar') return;

        const { ctx } = chart;
        const color = pluginOptions?.color || '#d8d8d8';
        const fontSize = pluginOptions?.fontSize || 11;
        const formatter = pluginOptions?.formatter || (value => Number(value).toFixed(2));
        const offset = pluginOptions?.offset || 4;

        ctx.save();
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.font = `600 ${fontSize}px sans-serif`;

        chart.data.datasets.forEach((dataset, datasetIndex) => {
            const meta = chart.getDatasetMeta(datasetIndex);
            if (meta.hidden) return;

            meta.data.forEach((bar, index) => {
                const value = dataset.data[index];
                if (value === null || value === undefined) return;

                const { x, y } = bar.tooltipPosition();
                ctx.fillText(formatter(value), x, y - offset);
            });
        });

        ctx.restore();
    },
};

function detectTopTierMin(values, alpha = 0.5) {
    const clean = values.filter(v => Number.isFinite(v));
    if (clean.length <= 1) return clean[0] ?? -Infinity;

    const sorted = [...clean].sort((a, b) => a - b);
    const gaps = sorted.slice(1).map((v, i) => v - sorted[i]);
    const meanGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    const threshold = alpha * meanGap;

    // Scan downward from the top; first sufficiently large gap defines top tier cutoff.
    for (let i = gaps.length - 1; i >= 0; i--) {
        if (gaps[i] > threshold) return sorted[i + 1];
    }
    return sorted[0];
}

function detectTierMins(values, alpha = 0.5) {
    const clean = values.filter(v => Number.isFinite(v));
    if (clean.length === 0) return [];
    if (clean.length === 1) return [clean[0]];

    const sorted = [...clean].sort((a, b) => a - b);
    const gaps = sorted.slice(1).map((v, i) => v - sorted[i]);
    const meanGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    const threshold = alpha * meanGap;

    const mins = [sorted[0]];
    for (let i = 0; i < gaps.length; i++) {
        if (gaps[i] > threshold) mins.push(sorted[i + 1]);
    }
    return mins;
}

function drawStar(ctx, cx, cy, outerR = 8, innerR = 3.8) {
    const spikes = 5;
    let rot = -Math.PI / 2;
    const step = Math.PI / spikes;

    ctx.beginPath();
    for (let i = 0; i < spikes; i++) {
        let x = cx + Math.cos(rot) * outerR;
        let y = cy + Math.sin(rot) * outerR;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        rot += step;

        x = cx + Math.cos(rot) * innerR;
        y = cy + Math.sin(rot) * innerR;
        ctx.lineTo(x, y);
        rot += step;
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
}

function drawStarsRow(ctx, cx, cy, count, size = 8, gap = 3) {
    if (count <= 0) return;
    const step = 2 * size + gap;
    const startX = cx - ((count - 1) * step) / 2;
    for (let i = 0; i < count; i++) {
        drawStar(ctx, startX + i * step, cy, size);
    }
}

const TOP_TIER_STARS_PLUGIN = {
    id: 'topTierStars',
    afterDraw(chart) {
        if (chart.config.type !== 'bar') return;

        const pluginOpts = chart.options?.plugins?.topTierStars;
        if (!pluginOpts?.enabled) return;

        const dataset = chart.data?.datasets?.[0];
        if (!dataset?.data?.length) return;

        const values = dataset.data.map(v => Number(v));
        const alpha = pluginOpts.alpha ?? 0.5;
        const tiered = !!pluginOpts.tiered;
        const tierMins = tiered ? detectTierMins(values, alpha) : [];
        const topMin = tiered ? -Infinity : detectTopTierMin(values, alpha);
        const meta = chart.getDatasetMeta(0);
        if (!meta?.data?.length) return;

        const { ctx } = chart;
        const yScale = chart.scales?.y;
        const starSize = Number.isFinite(pluginOpts.size) ? pluginOpts.size : 8;
        const labelOffset = Number.isFinite(pluginOpts.labelOffset) ? pluginOpts.labelOffset : 4;
        const labelFontSize = Number.isFinite(pluginOpts.labelFontSize) ? pluginOpts.labelFontSize : 11;
        const topPad = Number.isFinite(pluginOpts.topPadding)
            ? pluginOpts.topPadding
            : (labelOffset + labelFontSize + starSize + 8);
        const starGap = Number.isFinite(pluginOpts.starGap) ? pluginOpts.starGap : 3;
        const maxStars = Number.isFinite(pluginOpts.maxStars) ? Math.max(0, pluginOpts.maxStars) : Infinity;
        const starColor = pluginOpts.color || '#ffd54a';

        ctx.save();
        ctx.fillStyle = starColor;
        ctx.strokeStyle = 'rgba(0,0,0,0.55)';
        ctx.lineWidth = 1;

        meta.data.forEach((bar, i) => {
            const v = values[i];
            if (!Number.isFinite(v)) return;

            let stars = 0;
            if (tiered) {
                let tierIdx = 0;
                for (let k = 0; k < tierMins.length; k++) {
                    if (v >= tierMins[k]) tierIdx = k;
                }
                stars = Math.min(tierIdx, maxStars);
            } else if (v >= topMin) {
                stars = 1;
            }

            if (stars <= 0) return;

            let y = bar.y - topPad;
            if (yScale) y = Math.max(yScale.top + starSize + 2, y);
            drawStarsRow(ctx, bar.x, y, stars, starSize, starGap);
        });

        ctx.restore();
    },
};

Chart.register(DARK_PLUGIN, BAR_VALUE_LABELS_PLUGIN, TOP_TIER_STARS_PLUGIN);

function currentVotingModeLabel() {
    const useStrategy = document.getElementById('strategy-on')?.checked ?? false;
    return useStrategy ? 'Strategic voting' : 'Honest voting';
}

function composeChartTitle(title) {
    const modeLabel = currentVotingModeLabel();
    if (!title) return modeLabel;
    return [title, modeLabel];
}

function chartLineOpts(title, xLabel, yLabel) {
    const chartTitle = composeChartTitle(title);
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: { labels: { color: '#d8d8d8', boxWidth: 12, font: { size: 11 } } },
            title: { display: true, text: chartTitle, color: '#d8d8d8', font: { size: 12 } },
        },
        scales: {
            x: {
                title: { display: true, text: xLabel, color: '#888' },
                ticks: { color: '#888', maxTicksLimit: 8 },
                grid: { color: '#282828' },
            },
            y: {
                min: 0, max: 1.05,
                title: { display: true, text: yLabel, color: '#888' },
                ticks: { color: '#888' },
                grid: { color: '#282828' },
            },
        },
    };
}

function chartBarOpts(title, topTier = null, values = []) {
    const chartTitle = composeChartTitle(title);
    const topTierOpts = topTier?.enabled
        ? {
            enabled: true,
            alpha: Number.isFinite(topTier.alpha) ? topTier.alpha : 0.5,
            size: Number.isFinite(topTier.size) ? topTier.size : 8,
            topPadding: Number.isFinite(topTier.topPadding) ? topTier.topPadding : null,
            labelOffset: Number.isFinite(topTier.labelOffset) ? topTier.labelOffset : 4,
            labelFontSize: Number.isFinite(topTier.labelFontSize) ? topTier.labelFontSize : 11,
            tiered: !!topTier.tiered,
            maxStars: Number.isFinite(topTier.maxStars) ? topTier.maxStars : Infinity,
            starGap: Number.isFinite(topTier.starGap) ? topTier.starGap : 3,
            color: topTier.color || '#ffd54a',
        }
        : { enabled: false };

    const barValueLabelFontSize = 11;
    const barValueLabelOffset = 4;
    const maxVal = values.length ? Math.max(...values.filter(v => Number.isFinite(v))) : 1;
    const baseYMax = 1.1;
    // Data headroom avoids label/star collisions when bars are near the axis maximum.
    const minTopHeadroom = topTierOpts.enabled
        ? (Number.isFinite(topTier.minTopHeadroom) ? topTier.minTopHeadroom : 0.18)
        : 0.1;
    const yMax = Math.max(baseYMax, maxVal + minTopHeadroom);
    const topPadding = topTierOpts.enabled
        ? (Number.isFinite(topTierOpts.topPadding)
            ? topTierOpts.topPadding
            : barValueLabelOffset + barValueLabelFontSize + topTierOpts.size + 10)
        : 8;

    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        layout: {
            padding: {
                top: topPadding,
            },
        },
        plugins: {
            legend: { display: false },
            title: { display: true, text: chartTitle, color: '#d8d8d8', font: { size: 11 } },
            barValueLabels: {
                fontSize: barValueLabelFontSize,
                offset: barValueLabelOffset,
            },
            topTierStars: topTierOpts,
        },
        scales: {
            x: { ticks: { color: '#888', font: { size: 10 } }, grid: { color: '#282828' } },
            y: { min: 0, max: yMax, ticks: { color: '#888' }, grid: { color: '#282828' } },
        },
    };
}

function improvementAxisClamp(values) {
    const clean = values.filter(v => Number.isFinite(v));
    if (!clean.length) return null;

    const observedMin = Math.min(...clean);
    const observedMax = Math.max(...clean);
    const eps = 1e-12;

    // Strict subset checks: only clamp when all observed values are fully inside the target window.
    if (observedMin > 0 + eps && observedMax < 0.1 - eps) return { min: 0, max: 0.1 };
    if (observedMax < 0 - eps && observedMin > -0.1 + eps) return { min: -0.1, max: 0 };
    if (observedMin > -0.1 + eps && observedMax < 0.1 - eps) return { min: -0.1, max: 0.1 };
    return null;
}

const chartInstances = {};

function resizeAllCharts() {
    Object.values(chartInstances).forEach(chart => {
        if (chart && typeof chart.resize === 'function') chart.resize();
    });
}

function destroyChart(id) {
    if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; }
}

function newChart(id, type, data, opts) {
    destroyChart(id);
    const ctx = document.getElementById(id).getContext('2d');
    chartInstances[id] = new Chart(ctx, { type, data, options: opts });
}

// ── Slice plots ───────────────────────────────────────────────────────────────

function plotSlices(res, tVals, lVals, methods) {
    const names = Object.keys(methods);
    const liMax = lVals.length - 1;  // l = 1
    const tiMax = tVals.length - 1;  // t = 1

    // VSE vs t (l=1)
    newChart('chart-slice-t', 'line', {
        labels: tVals.map(v => v.toFixed(2)),
        datasets: names.map(m => ({
            label: m,
            data: Array.from(res[m][liMax]),
            borderColor: METHOD_COLORS[m],
            backgroundColor: METHOD_COLORS[m] + '22',
            borderWidth: 2,
            pointRadius: 2,
            tension: 0.3,
        })),
    }, chartLineOpts('Full ballot (ℓ = 1) — VSE vs knowledge t', 'Knowledge  t', 'VSE'));

    // VSE vs l (t=1)
    newChart('chart-slice-l', 'line', {
        labels: lVals.map(v => v.toFixed(2)),
        datasets: names.map(m => ({
            label: m,
            data: Array.from({ length: lVals.length }, (_, li) => res[m][li][tiMax]),
            borderColor: METHOD_COLORS[m],
            backgroundColor: METHOD_COLORS[m] + '22',
            borderWidth: 2,
            pointRadius: 2,
            tension: 0.3,
        })),
    }, chartLineOpts('Perfect knowledge (t = 1) — VSE vs energy ℓ', 'Energy  ℓ', 'VSE'));
}

// ── Heatmaps (Canvas-based) ───────────────────────────────────────────────────

/** viridis-like colormap: [0,1] → [r,g,b] */
function viridis(x) {
    x = Math.max(0, Math.min(1, x));
    // simplified 4-stop version
    const stops = [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
    ];
    const t = x * (stops.length - 1);
    const i = Math.min(Math.floor(t), stops.length - 2);
    const f = t - i;
    return stops[i].map((c, j) => Math.round(c + f * (stops[i + 1][j] - c)));
}

function diverging(x) {
    // red–yellow–green for [-1, 1]
    x = Math.max(-1, Math.min(1, x));
    if (x < 0) {
        const t = 1 + x; // 0 (red) → 1 (yellow)
        return [255, Math.round(t * 220), Math.round(t * 50)];
    } else {
        const t = x;
        return [Math.round((1 - t) * 220), Math.round(50 + t * 180), Math.round(50)];
    }
}

function drawHeatmap(canvas, matrix, colorFn, vmin, vmax) {
    const nl = matrix.length, nt = matrix[0].length;
    canvas.width = nt; canvas.height = nl;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(nt, nl);
    for (let li = 0; li < nl; li++) {
        for (let ti = 0; ti < nt; ti++) {
            const v = matrix[nl - 1 - li][ti];  // flip y
            const norm = (v - vmin) / (vmax - vmin);
            const [r, g, b] = colorFn(norm);
            const idx = (li * nt + ti) * 4;
            img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);
}

function drawColorbar(canvas, colorFn) {
    canvas.width = 200; canvas.height = 1;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(200, 1);
    for (let i = 0; i < 200; i++) {
        const [r, g, b] = colorFn(i / 199);
        const idx = i * 4;
        img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
}

function buildHeatmapContainer(containerId, res, methods, colorFn, vmin, vmax, labelFn) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    for (const m of Object.keys(methods)) {
        const item = document.createElement('div');
        item.className = 'heatmap-item';

        const h4 = document.createElement('h4');
        h4.textContent = labelFn ? labelFn(m) : m;
        h4.style.color = METHOD_COLORS[m] || '#d8d8d8';
        item.appendChild(h4);

        const canvas = document.createElement('canvas');
        drawHeatmap(canvas, res[m], colorFn, vmin, vmax);
        item.appendChild(canvas);

        // colorbar
        const cbWrap = document.createElement('div');
        cbWrap.className = 'colorbar-wrap';
        const cbCanvas = document.createElement('canvas');
        cbCanvas.className = 'colorbar-strip';
        drawColorbar(cbCanvas, colorFn);
        cbWrap.appendChild(cbCanvas);
        const labels = document.createElement('div');
        labels.className = 'colorbar-labels';
        labels.innerHTML = `<span>${vmin.toFixed(1)}</span><span>${((vmin + vmax) / 2).toFixed(2)}</span><span>${vmax.toFixed(1)}</span>`;
        cbWrap.appendChild(labels);
        item.appendChild(cbWrap);

        // axis labels
        const axInfo = document.createElement('div');
        axInfo.style.cssText = 'font-size:0.72rem;color:#555;margin-top:4px;text-align:center';
        axInfo.textContent = '← knowledge t →   (↑ energy ℓ ↑)';
        item.appendChild(axInfo);

        container.appendChild(item);
    }
}

// ── Robustness ────────────────────────────────────────────────────────────────

function plotRobustness(res, tVals, lVals, methods) {
    const nl = lVals.length, nt = tVals.length;
    const dt = tVals[1] - tVals[0], dl = lVals[1] - lVals[0];
    const tot = (tVals[nt - 1] - tVals[0]) * (lVals[nl - 1] - lVals[0]);
    const tauRange = linspace(0.3, 1.0, 80);
    const methodNames = Object.keys(methods);

    const robustnessSeries = {};
    for (const m of methodNames) {
        robustnessSeries[m] = tauRange.map(tau => {
            let count = 0;
            for (let li = 0; li < nl; li++) for (let ti = 0; ti < nt; ti++) if (res[m][li][ti] >= tau) count++;
            return count * dt * dl / tot;
        });
    }

    // Trim left-side tau values until at least one method has visible robustness.
    const detectionEpsilon = 1e-6;
    let firstDetectedIdx = 0;
    for (let i = 0; i < tauRange.length; i++) {
        if (methodNames.some(m => robustnessSeries[m][i] > detectionEpsilon)) {
            firstDetectedIdx = i;
            break;
        }
    }
    const leftPadPoints = 1;
    const startIdx = Math.max(0, firstDetectedIdx - leftPadPoints);
    const tauSlice = tauRange.slice(startIdx);

    newChart('chart-robustness', 'line', {
        labels: tauSlice.map(v => v.toFixed(2)),
        datasets: methodNames.map(m => ({
            label: m,
            data: robustnessSeries[m].slice(startIdx),
            borderColor: METHOD_COLORS[m],
            backgroundColor: METHOD_COLORS[m] + '22',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
        })),
    }, chartLineOpts('Robustness: fraction of (t,ℓ) space with VSE ≥ τ', 'VSE threshold  τ', 'Fraction of space'));
}

// ── Scenarios ─────────────────────────────────────────────────────────────────

function closestIdx(arr, v) {
    let best = 0;
    arr.forEach((x, i) => { if (Math.abs(x - v) < Math.abs(arr[best] - v)) best = i; });
    return best;
}

function scenarioSpecs(tVals, lVals, includeId = false) {
    const specs = [
        { label: 'Ideal\n(t=1, ℓ=1)', ti: closestIdx(tVals, 1.0), li: closestIdx(lVals, 1.0) },
        { label: 'Low energy\n(t=1, ℓ=0.35)', ti: closestIdx(tVals, 1.0), li: closestIdx(lVals, 0.35) },
        { label: 'Low knowledge\n(t=0.3, ℓ=1)', ti: closestIdx(tVals, 0.3), li: closestIdx(lVals, 1.0) },
        { label: 'Both low\n(t=0.3, ℓ=0.35)', ti: closestIdx(tVals, 0.3), li: closestIdx(lVals, 0.35) },
    ];

    if (!includeId) return specs;
    return specs.map((s, i) => ({ ...s, id: `chart-sc-${i + 1}` }));
}

function plotScenarios(res, tVals, lVals, methods) {
    const showStars = shouldShowTierStars();
    const scenarios = scenarioSpecs(tVals, lVals, true);
    const names = Object.keys(methods);

    for (const sc of scenarios) {
        const scenarioValues = names.map(m => res[m][sc.li][sc.ti]);
        newChart(sc.id, 'bar', {
            labels: names,
            datasets: [{
                data: scenarioValues,
                backgroundColor: names.map(m => METHOD_COLORS[m] + 'cc'),
                borderColor: names.map(m => METHOD_COLORS[m]),
                borderWidth: 1,
            }],
        }, chartBarOpts(
            sc.label.replace('\n', ' '),
            { enabled: showStars, alpha: 0.5, tiered: true, maxStars: 3, minTopHeadroom: 0.18 },
            scenarioValues
        ));
    }
}

function plotScenarioOverallImprovement(res, tVals, lVals, methods) {
    const names = Object.keys(methods);
    const tab = document.getElementById('tab-scenario-overall');

    const hasPlurality = names.includes('Plurality');
    const hasApproval = names.includes('Approval');
    const targets = names.filter(m => m !== 'Plurality' && m !== 'Approval');
    const emptyMsg = document.getElementById('scenario-overall-empty');

    if (!hasPlurality || !hasApproval || targets.length === 0) {
        ['chart-so-1', 'chart-so-2', 'chart-so-3', 'chart-so-4'].forEach(id => destroyChart(id));
        if (emptyMsg) {
            emptyMsg.style.display = 'block';
            const missing = [
                !hasPlurality ? 'Plurality' : null,
                !hasApproval ? 'Approval' : null,
            ].filter(Boolean);
            if (targets.length === 0) {
                emptyMsg.textContent = 'Enable at least one reform target (RCV, Borda, Score, STAR, or Condorcet) to view scenario improvement.';
            } else if (missing.length > 0) {
                emptyMsg.textContent = `Enable ${missing.join(' and ')} to view scenario improvement vs baseline methods.`;
            }
        }
        if (tab) tab.classList.add('has-empty');
        return;
    }

    if (emptyMsg) emptyMsg.style.display = 'none';
    if (tab) tab.classList.remove('has-empty');

    const scenarios = scenarioSpecs(tVals, lVals, true).map((sc, i) => ({
        ...sc,
        id: `chart-so-${i + 1}`,
    }));

    for (const sc of scenarios) {
        const pluralityVals = targets.map(name => res[name][sc.li][sc.ti] - res['Plurality'][sc.li][sc.ti]);
        const approvalVals = targets.map(name => res[name][sc.li][sc.ti] - res['Approval'][sc.li][sc.ti]);
        const combined = pluralityVals.concat(approvalVals);
        const yClamp = improvementAxisClamp(combined);

        newChart(sc.id, 'bar', {
            labels: targets,
            datasets: [
                {
                    label: 'vs Plurality',
                    data: pluralityVals,
                    backgroundColor: METHOD_COLORS['Plurality'] + 'cc',
                    borderColor: METHOD_COLORS['Plurality'],
                    borderWidth: 1,
                },
                {
                    label: 'vs Approval',
                    data: approvalVals,
                    backgroundColor: METHOD_COLORS['Approval'] + 'cc',
                    borderColor: METHOD_COLORS['Approval'],
                    borderWidth: 1,
                },
            ],
        }, {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#d8d8d8', boxWidth: 12, font: { size: 11 } },
                },
                title: { display: true, text: composeChartTitle(sc.label.replace('\n', ' ')), color: '#d8d8d8', font: { size: 11 } },
                barValueLabels: {
                    fontSize: 11,
                    offset: 4,
                    formatter: value => Number(value).toFixed(3),
                },
                topTierStars: {
                    enabled: false,
                    alpha: 0.5,
                    tiered: true,
                    maxStars: 3,
                    minTopHeadroom: 0.16,
                },
            },
            scales: {
                x: {
                    ticks: { color: '#888', font: { size: 10 } },
                    grid: { color: '#282828' },
                },
                y: {
                    ticks: { color: '#888' },
                    grid: { color: '#282828' },
                    title: { display: true, text: 'Scenario improvement score', color: '#888' },
                    ...(yClamp ? { min: yClamp.min, max: yClamp.max } : {}),
                },
            },
        });
    }
}

// ── Weighted VSE ──────────────────────────────────────────────────────────────

function plotWeighted(res, tVals, lVals, methods) {
    const showStars = shouldShowTierStars();
    const nl = lVals.length, nt = tVals.length;
    const names = Object.keys(methods);

    // Meshgrid
    const T = Array.from({ length: nl }, (_, li) => Array.from({ length: nt }, (_, ti) => tVals[ti]));
    const L = Array.from({ length: nl }, (_, li) => Array.from({ length: nt }, () => lVals[li]));

    function wAvg(wFn) {
        const w = Array.from({ length: nl }, (_, li) => Array.from({ length: nt }, (_, ti) => wFn(T[li][ti], L[li][ti])));
        let wSum = 0;
        for (let li = 0; li < nl; li++) for (let ti = 0; ti < nt; ti++) wSum += w[li][ti];
        const out = {};
        for (const m of names) {
            let s = 0;
            for (let li = 0; li < nl; li++) for (let ti = 0; ti < nt; ti++) s += res[m][li][ti] * w[li][ti];
            out[m] = s / wSum;
        }
        return out;
    }

    const centerSigma = 0.2;

    const priors = [
        { id: 'chart-w-1', label: 'Uniform prior', fn: () => 1 },
        { id: 'chart-w-2', label: 'Ideal focus\n(high t, high ℓ)', fn: (t, l) => Math.exp(3 * t + 3 * l) },
        {
            id: 'chart-w-3',
            label: 'Center focus\n(mid t, mid ℓ)',
            fn: (t, l) => Math.exp(-(((t - 0.5) ** 2 + (l - 0.5) ** 2) / (2 * centerSigma ** 2))),
        },
        { id: 'chart-w-4', label: 'Informed & exhausted\n(high t, low ℓ)', fn: (t, l) => Math.exp(3 * t - 3 * (1 - l)) },
        { id: 'chart-w-5', label: 'Energetic & uninformed\n(low t, high ℓ)', fn: (t, l) => Math.exp(-3 * t + 3 * l) },
        { id: 'chart-w-6', label: 'Uninformed & exhausted\n(low t, low ℓ)', fn: (t, l) => Math.exp(-3 * t + 3 * (1 - l)) },
    ];

    for (const p of priors) {
        const avgs = wAvg(p.fn);
        const weightedValues = names.map(m => avgs[m]);
        newChart(p.id, 'bar', {
            labels: names,
            datasets: [{
                data: weightedValues,
                backgroundColor: names.map(m => METHOD_COLORS[m] + 'cc'),
                borderColor: names.map(m => METHOD_COLORS[m]),
                borderWidth: 1,
            }],
        }, chartBarOpts(
            p.label.replace('\n', ' '),
            { enabled: showStars, alpha: 0.5, tiered: true, maxStars: 3, minTopHeadroom: 0.18 },
            weightedValues
        ));
    }
}

function weightedPriors() {
    const centerSigma = 0.2;
    return [
        { id: 'chart-wo-1', label: 'Uniform prior', fn: () => 1 },
        { id: 'chart-wo-2', label: 'Ideal focus\n(high t, high ℓ)', fn: (t, l) => Math.exp(3 * t + 3 * l) },
        {
            id: 'chart-wo-3',
            label: 'Center focus\n(mid t, mid ℓ)',
            fn: (t, l) => Math.exp(-(((t - 0.5) ** 2 + (l - 0.5) ** 2) / (2 * centerSigma ** 2))),
        },
        { id: 'chart-wo-4', label: 'Informed & exhausted\n(high t, low ℓ)', fn: (t, l) => Math.exp(3 * t - 3 * (1 - l)) },
        { id: 'chart-wo-5', label: 'Energetic & uninformed\n(low t, high ℓ)', fn: (t, l) => Math.exp(-3 * t + 3 * l) },
        { id: 'chart-wo-6', label: 'Uninformed & exhausted\n(low t, low ℓ)', fn: (t, l) => Math.exp(-3 * t + 3 * (1 - l)) },
    ];
}

function weightedMethodScores(res, tVals, lVals, names, weightFn) {
    const nl = lVals.length;
    const nt = tVals.length;
    let wSum = 0;
    const weighted = {};
    for (const name of names) weighted[name] = 0;

    for (let li = 0; li < nl; li++) {
        for (let ti = 0; ti < nt; ti++) {
            const w = weightFn(tVals[ti], lVals[li]);
            wSum += w;
            for (const name of names) weighted[name] += res[name][li][ti] * w;
        }
    }

    if (wSum <= 0) return Object.fromEntries(names.map(name => [name, 0]));
    for (const name of names) weighted[name] /= wSum;
    return weighted;
}

function plotWeightedOverallImprovement(res, tVals, lVals, methods) {
    const names = Object.keys(methods);
    const tab = document.getElementById('tab-weighted-overall');

    const hasPlurality = names.includes('Plurality');
    const hasApproval = names.includes('Approval');
    const targets = names.filter(m => m !== 'Plurality' && m !== 'Approval');
    const emptyMsg = document.getElementById('weighted-overall-empty');

    if (!hasPlurality || !hasApproval || targets.length === 0) {
        ['chart-wo-1', 'chart-wo-2', 'chart-wo-3', 'chart-wo-4', 'chart-wo-5', 'chart-wo-6'].forEach(id => destroyChart(id));
        if (emptyMsg) {
            emptyMsg.style.display = 'block';
            const missing = [
                !hasPlurality ? 'Plurality' : null,
                !hasApproval ? 'Approval' : null,
            ].filter(Boolean);
            if (targets.length === 0) {
                emptyMsg.textContent = 'Enable at least one reform target (RCV, Borda, Score, STAR, or Condorcet) to view weighted improvement.';
            } else if (missing.length > 0) {
                emptyMsg.textContent = `Enable ${missing.join(' and ')} to view weighted improvement vs baseline methods.`;
            }
        }
        if (tab) tab.classList.add('has-empty');
        return;
    }

    if (emptyMsg) emptyMsg.style.display = 'none';
    if (tab) tab.classList.remove('has-empty');

    const priors = weightedPriors();
    for (const p of priors) {
        const scores = weightedMethodScores(res, tVals, lVals, names, p.fn);
        const pluralityVals = targets.map(name => scores[name] - scores['Plurality']);
        const approvalVals = targets.map(name => scores[name] - scores['Approval']);
        const combined = pluralityVals.concat(approvalVals);
        const yClamp = improvementAxisClamp(combined);

        newChart(p.id, 'bar', {
            labels: targets,
            datasets: [
                {
                    label: 'vs Plurality',
                    data: pluralityVals,
                    backgroundColor: METHOD_COLORS['Plurality'] + 'cc',
                    borderColor: METHOD_COLORS['Plurality'],
                    borderWidth: 1,
                },
                {
                    label: 'vs Approval',
                    data: approvalVals,
                    backgroundColor: METHOD_COLORS['Approval'] + 'cc',
                    borderColor: METHOD_COLORS['Approval'],
                    borderWidth: 1,
                },
            ],
        }, {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#d8d8d8', boxWidth: 12, font: { size: 11 } },
                },
                title: { display: true, text: composeChartTitle(p.label.replace('\n', ' ')), color: '#d8d8d8', font: { size: 11 } },
                barValueLabels: {
                    fontSize: 11,
                    offset: 4,
                    formatter: value => Number(value).toFixed(3),
                },
                topTierStars: {
                    enabled: false,
                    alpha: 0.5,
                    tiered: true,
                    maxStars: 3,
                    minTopHeadroom: 0.16,
                },
            },
            scales: {
                x: {
                    ticks: { color: '#888', font: { size: 10 } },
                    grid: { color: '#282828' },
                },
                y: {
                    ticks: { color: '#888' },
                    grid: { color: '#282828' },
                    title: { display: true, text: 'Weighted improvement score', color: '#888' },
                    ...(yClamp ? { min: yClamp.min, max: yClamp.max } : {}),
                },
            },
        });
    }
}

// ── Approval diff heatmaps ────────────────────────────────────────────────────

function plotApprovalDiff(res, tVals, lVals, methods) {
    if (!res['Approval']) return;
    const others = Object.keys(methods).filter(m => m !== 'Approval');
    if (!others.length) { document.getElementById('diff-container').innerHTML = '<p style="color:#555;padding:1rem">No other methods to compare.</p>'; return; }
    const diffRes = {};
    const nl = lVals.length, nt = tVals.length;
    for (const m of others) {
        diffRes[m] = Array.from({ length: nl }, (_, li) =>
            new Float64Array(nt).map((_, ti) => res['Approval'][li][ti] - res[m][li][ti])
        );
    }
    buildHeatmapContainer('diff-container', diffRes, Object.fromEntries(others.map(m => [m, true])),
        v => diverging(2 * v - 1),  // map normalized [0,1] to [-1,1]
        -0.15, 0.15,
        m => `Approval − ${m}`
    );
}

// ── UI wiring ─────────────────────────────────────────────────────────────────

function getParams() {
    const useStrategy = document.getElementById('strategy-on')?.checked ?? false;
    const enabledMethods = {
        Plurality: document.getElementById('m-plurality').checked,
        RCV: document.getElementById('m-irv').checked,
        Borda: document.getElementById('m-borda').checked,
        Score: document.getElementById('m-score').checked,
        Approval: document.getElementById('m-approval').checked,
        STAR: document.getElementById('m-star').checked,
        Condorcet: document.getElementById('m-condorcet').checked,
    };
    return {
        nv: +document.getElementById('nv').value,
        nc: +document.getElementById('nc').value,
        ntr: +document.getElementById('ntr').value,
        ng: +document.getElementById('ng').value,
        nd: +document.getElementById('nd').value,
        useStrategy,
        strategyShare: 1.0,
        methods: buildMethods(enabledMethods, { useStrategy, strategyShare: 1.0 }),
    };
}

function isMobileLikeViewport() {
    const narrow = window.matchMedia('(max-width: 760px)').matches;
    const coarsePointer = window.matchMedia('(hover: none) and (pointer: coarse)').matches;
    return narrow || coarsePointer;
}

function shouldShowTierStars() {
    const cb = document.getElementById('show-stars');
    return cb ? cb.checked : true;
}

function updateStarLegendVisibility() {
    const show = shouldShowTierStars();
    ['star-legend-scenarios', 'star-legend-weighted'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = show ? 'block' : 'none';
    });
}

function renderChartsFromState(state) {
    const { res, tVals, lVals, methods } = state;
    plotSlices(res, tVals, lVals, methods);
    buildHeatmapContainer('heatmap-container', res, methods, viridis, 0, 1, null);
    plotRobustness(res, tVals, lVals, methods);
    plotScenarios(res, tVals, lVals, methods);
    plotWeighted(res, tVals, lVals, methods);
    plotWeightedOverallImprovement(res, tVals, lVals, methods);
    plotScenarioOverallImprovement(res, tVals, lVals, methods);
    renderHypothesisTests(state);
    plotApprovalDiff(res, tVals, lVals, methods);
    updateStarLegendVisibility();
}

// Sync sliders to badges
document.addEventListener('DOMContentLoaded', () => {
    const starsToggle = document.getElementById('show-stars');
    if (starsToggle) {
        // Desktop default on, mobile default off.
        starsToggle.checked = !isMobileLikeViewport();
        starsToggle.addEventListener('change', () => {
            updateStarLegendVisibility();
            if (lastSimResult) renderChartsFromState(lastSimResult);
        });
    }

    ['nv', 'nc', 'ntr', 'ng', 'nd'].forEach(id => {
        const slider = document.getElementById(id);
        const badge = document.getElementById(id + '-val');
        if (slider && badge) {
            slider.addEventListener('input', () => { badge.textContent = slider.value; });
        }
    });

    updateStarLegendVisibility();
});

// Tabs
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
        requestAnimationFrame(resizeAllCharts);
    });
});

function setRunning(isRunning) {
    const btn = document.getElementById('run-btn');
    const cancelBtn = document.getElementById('cancel-btn');
    const bar = document.getElementById('progress-bar');
    if (isRunning) {
        btn.classList.add('running');
        bar.classList.add('running');
        cancelBtn.disabled = false;
    } else {
        btn.classList.remove('running');
        bar.classList.remove('running');
        cancelBtn.disabled = true;
    }
}

function requestCancel() {
    if (!isSimRunning) return;
    simAborted = true;
    const cancelBtn = document.getElementById('cancel-btn');
    const statusEl = document.getElementById('sim-status');
    cancelBtn.disabled = true;
    statusEl.textContent = 'Cancelling simulation...';
}

document.getElementById('cancel-btn').addEventListener('click', requestCancel);

// Run button
document.getElementById('run-btn').addEventListener('click', async () => {
    if (isSimRunning) return;

    simAborted = true;  // abort any running sim
    await sleep(10);
    simAborted = false;
    isSimRunning = true;

    rng = mulberry32(42);  // reset RNG for reproducibility

    const btn = document.getElementById('run-btn');
    const statusEl = document.getElementById('sim-status');
    const progressBar = document.getElementById('progress-bar');
    const progressLabel = document.getElementById('progress-label');

    btn.disabled = true;
    btn.textContent = '⏳ Running…';
    statusEl.textContent = 'Initialising…';
    progressLabel.textContent = '0%';
    progressBar.style.width = '0%';
    setRunning(true);

    const params = getParams();
    if (Object.keys(params.methods).length === 0) {
        statusEl.textContent = 'Please enable at least one method.';
        btn.disabled = false; btn.textContent = '▶ Run Simulation';
        progressLabel.textContent = 'Ready';
        isSimRunning = false;
        setRunning(false);
        return;
    }

    try {
        const result = await runGrid(params, (done, total, elapsed, eta) => {
            const pct = (100 * done / total).toFixed(0);
            progressBar.style.width = pct + '%';
            progressLabel.textContent = `${pct}%`;
            statusEl.textContent = `${done}/${total} grid points · ${elapsed.toFixed(1)}s elapsed · ETA ${eta.toFixed(0)}s`;
        });

        if (!result) {
            progressLabel.textContent = 'Cancelled';
            statusEl.textContent = 'Simulation cancelled.';
            return;
        }

        const { res, tVals, lVals, trialVse } = result;

        statusEl.textContent = 'Rendering plots…';
        await sleep(0);

        lastSimResult = { res, tVals, lVals, trialVse, methods: params.methods, useStrategy: params.useStrategy };
        renderChartsFromState(lastSimResult);

        document.getElementById('results').style.display = 'block';
        document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });

        progressBar.style.width = '100%';
        progressLabel.textContent = '100%';
        const modeLabel = params.useStrategy ? 'strategic mode' : 'honest mode';
        statusEl.textContent = `Done (${modeLabel}). ${params.ng * params.ng} grid points × ${params.ntr} trials each.`;
    } catch (err) {
        statusEl.textContent = 'Simulation failed. Check console for details.';
        console.error(err);
    } finally {
        simAborted = false;
        isSimRunning = false;
        btn.disabled = false;
        btn.textContent = '▶ Run Simulation';
        setRunning(false);
    }
});
