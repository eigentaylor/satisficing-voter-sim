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
    RCV: '#e09944',
    Borda: '#d4c94a',
    Score: '#b8e04a',
    Approval: '#4ec96a',
    STAR: '#4ab8e0',
    Condorcet: '#9b6be0',
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

// ── Voting methods ────────────────────────────────────────────────────────────

function plurality(pu) {
    const nv = pu.length, nc = pu[0].length;
    const fp = new Array(nc).fill(0);
    for (let i = 0; i < nv; i++) fp[argmax(pu[i])]++;
    return argmax(fp);
}

function approval(pu, l) {
    const nv = pu.length, nc = pu[0].length;
    const tidxAll = topIdx(pu, l);
    const scores = new Array(nc).fill(0);
    for (let i = 0; i < nv; i++) {
        const row = pu[i];
        const globalMean = row.reduce((s, x) => s + x, 0) / nc;
        const considered = new Set(tidxAll[i]);
        let approved = 0;
        for (const c of considered) {
            if (row[c] > globalMean) { scores[c]++; approved++; }
        }
        if (approved === 0) scores[tidxAll[i][0]]++;
    }
    return argmax(scores);
}

function borda(pu, l) {
    const nv = pu.length, nc = pu[0].length;
    const tidxAll = topIdx(pu, l);
    const K = tidxAll[0].length;
    const scores = new Array(nc).fill(0);
    for (let i = 0; i < nv; i++) {
        for (let k = 0; k < K; k++) scores[tidxAll[i][k]] += K - k;
    }
    return argmax(scores);
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

function scoreVote(pu, l) {
    const ballots = globalScoreBallots(pu, l);
    const nc = ballots[0].length;
    const totals = new Array(nc).fill(0);
    for (const ballot of ballots) for (let j = 0; j < nc; j++) totals[j] += ballot[j];
    return argmax(totals);
}

function star(pu, l) {
    const ballots = globalScoreBallots(pu, l);
    const nv = ballots.length, nc = ballots[0].length;
    const totals = new Array(nc).fill(0);
    for (let i = 0; i < nv; i++) for (let j = 0; j < nc; j++) totals[j] += ballots[i][j];
    const sorted = Array.from({ length: nc }, (_, i) => i).sort((a, b) => totals[b] - totals[a]);
    const f1 = sorted[0], f2 = sorted[1];
    let w1 = 0, w2 = 0;
    for (let i = 0; i < nv; i++) {
        if (ballots[i][f1] > ballots[i][f2]) w1++;
        else if (ballots[i][f2] > ballots[i][f1]) w2++;
    }
    return w1 >= w2 ? f1 : f2;
}

function condorcet(pu, l) {
    const nv = pu.length, nc = pu[0].length;
    const tidxAll = topIdx(pu, l);
    const K = tidxAll[0].length;
    // rank matrix: rm[i][c] = rank of c for voter i (0=best), unranked=K
    const rm = Array.from({ length: nv }, () => new Array(nc).fill(K));
    for (let i = 0; i < nv; i++) for (let k = 0; k < K; k++) rm[i][tidxAll[i][k]] = k;
    // pairwise counts: only strict preferences count, ties are ignored
    const pairwise = Array.from({ length: nc }, () => new Array(nc).fill(0));
    for (let a = 0; a < nc; a++) {
        for (let b = a + 1; b < nc; b++) {
            let wa = 0, wb = 0;
            for (let i = 0; i < nv; i++) {
                if (rm[i][a] < rm[i][b]) wa++;
                else if (rm[i][b] < rm[i][a]) wb++;
            }
            pairwise[a][b] = wa;
            pairwise[b][a] = wb;
        }
    }
    // Minimax fallback: choose the candidate with the smallest worst defeat margin.
    const worstDefeat = pairwise.map((row, a) => {
        let worst = 0;
        for (let b = 0; b < nc; b++) {
            if (b === a) continue;
            worst = Math.max(worst, pairwise[b][a] - row[b]);
        }
        return worst;
    });
    return worstDefeat.indexOf(Math.min(...worstDefeat));
}

function irv(pu, l) {
    const nv = pu.length, nc = pu[0].length;
    const tidxAll = topIdx(pu, l);
    const K = tidxAll[0].length;
    // each voter's preference order (truncated)
    const ballots = tidxAll; // already sorted by pu desc
    const remaining = new Array(nc).fill(true);

    for (let round = 0; round < nc - 1; round++) {
        const active = [];
        for (let c = 0; c < nc; c++) if (remaining[c]) active.push(c);
        if (active.length === 1) return active[0];

        const fp = new Array(nc).fill(0);
        for (let i = 0; i < nv; i++) {
            for (let k = 0; k < K; k++) {
                if (remaining[ballots[i][k]]) { fp[ballots[i][k]]++; break; }
            }
        }
        const total = active.reduce((s, c) => s + fp[c], 0);
        const winner = active.find(c => fp[c] > total / 2);
        if (winner !== undefined) return winner;
        // eliminate lowest
        let minFP = Infinity, elim = -1;
        for (const c of active) if (fp[c] < minFP) { minFP = fp[c]; elim = c; }
        remaining[elim] = false;
    }
    return remaining.findIndex(x => x);
}

// ── Method registry ───────────────────────────────────────────────────────────

function buildMethods(enabled) {
    const all = {
        Plurality: (pu, l) => plurality(pu),
        RCV: (pu, l) => irv(pu, l),
        Borda: (pu, l) => borda(pu, l),
        Score: (pu, l) => scoreVote(pu, l),
        Approval: (pu, l) => approval(pu, l),
        STAR: (pu, l) => star(pu, l),
        Condorcet: (pu, l) => condorcet(pu, l),
    };
    const out = {};
    for (const k of Object.keys(all)) if (enabled[k]) out[k] = all[k];
    return out;
}

// ── Grid runner (async-friendly) ──────────────────────────────────────────────

let simAborted = false;
let isSimRunning = false;
let lastSimResult = null;

async function runGrid(params, onProgress) {
    const { nv, nc, ntr, ng, nd, methods } = params;
    const methodNames = Object.keys(methods);
    const tVals = linspace(0, 1, ng);
    const lVals = linspace(1 / nc, 1, ng);
    const res = {};
    for (const m of methodNames) res[m] = Array.from({ length: ng }, () => new Float64Array(ng));

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
                    acc[name] += vse(cm[w], swRand, swOpt);
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
    return { res, tVals, lVals };
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

function chartLineOpts(title, xLabel, yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: { labels: { color: '#d8d8d8', boxWidth: 12, font: { size: 11 } } },
            title: { display: !!title, text: title, color: '#d8d8d8', font: { size: 12 } },
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
            title: { display: !!title, text: title, color: '#d8d8d8', font: { size: 11 } },
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

const chartInstances = {};

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

    newChart('chart-robustness', 'line', {
        labels: tauRange.map(v => v.toFixed(2)),
        datasets: Object.keys(methods).map(m => ({
            label: m,
            data: tauRange.map(tau => {
                let count = 0;
                for (let li = 0; li < nl; li++) for (let ti = 0; ti < nt; ti++) if (res[m][li][ti] >= tau) count++;
                return count * dt * dl / tot;
            }),
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
                title: { display: true, text: sc.label.replace('\n', ' '), color: '#d8d8d8', font: { size: 11 } },
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
                title: { display: true, text: p.label.replace('\n', ' '), color: '#d8d8d8', font: { size: 11 } },
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
    return {
        nv: +document.getElementById('nv').value,
        nc: +document.getElementById('nc').value,
        ntr: +document.getElementById('ntr').value,
        ng: +document.getElementById('ng').value,
        nd: +document.getElementById('nd').value,
        methods: buildMethods({
            Plurality: document.getElementById('m-plurality').checked,
            RCV: document.getElementById('m-irv').checked,
            Borda: document.getElementById('m-borda').checked,
            Score: document.getElementById('m-score').checked,
            Approval: document.getElementById('m-approval').checked,
            STAR: document.getElementById('m-star').checked,
            Condorcet: document.getElementById('m-condorcet').checked,
        }),
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

        const { res, tVals, lVals } = result;

        statusEl.textContent = 'Rendering plots…';
        await sleep(0);

        lastSimResult = { res, tVals, lVals, methods: params.methods };
        renderChartsFromState(lastSimResult);

        document.getElementById('results').style.display = 'block';
        document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });

        progressBar.style.width = '100%';
        progressLabel.textContent = '100%';
        statusEl.textContent = `Done. ${params.ng * params.ng} grid points × ${params.ntr} trials each.`;
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
