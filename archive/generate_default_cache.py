#!/usr/bin/env python3
"""Generate bundled default simulation cache JSON for browser preload.

This script computes honest and strategic results from the same trial draws
and writes them in the shape expected by sim.js loadBundledDefaultBundle().
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import archive.satisficing_sim_strat_1 as sim

METHOD_ORDER = [
    "Plurality",
    "Approval",
    "RCV",
    "STAR",
    "Condorcet",
    "Score",
    "Borda",
]

DEFAULT_ENABLED = {
    "Plurality": True,
    "Approval": True,
    "RCV": True,
    "STAR": True,
    "Condorcet": True,
    "Score": False,
    "Borda": False,
}


def ordered_enabled_methods(enabled: Dict[str, bool]) -> List[str]:
    return [name for name in METHOD_ORDER if enabled.get(name, False)]


def params_signature(nv: int, nc: int, ntr: int, ng: int, nd: int, enabled: Dict[str, bool]) -> str:
    names = ordered_enabled_methods(enabled)
    return f"{nv}|{nc}|{ntr}|{ng}|{nd}|{','.join(names)}"


def cache_key(signature: str, cache_version: int) -> str:
    return f"svs-cache-v{cache_version}:{signature}"


def run_dual_mode(
    nv: int,
    nc: int,
    ntr: int,
    ng: int,
    nd: int,
    enabled_methods: Dict[str, bool],
    strategy_share: float,
    seed: int,
) -> dict:
    method_names = ordered_enabled_methods(enabled_methods)
    if not method_names:
        raise ValueError("At least one voting method must be enabled.")

    sim.rng = np.random.default_rng(seed)

    t_vals = np.linspace(0.0, 1.0, ng)
    l_vals = np.linspace(1.0 / nc, 1.0, ng)

    res_h = {m: np.zeros((ng, ng), dtype=float) for m in method_names}
    trial_h = {m: np.zeros((ng, ng, ntr), dtype=float) for m in method_names}
    res_s = {m: np.zeros((ng, ng), dtype=float) for m in method_names}
    trial_s = {m: np.zeros((ng, ng, ntr), dtype=float) for m in method_names}

    total_cells = ng * ng
    done = 0
    t0 = time.time()

    for ti, t in enumerate(t_vals):
        for li, l in enumerate(l_vals):
            acc_h = {m: 0.0 for m in method_names}
            acc_s = {m: 0.0 for m in method_names}

            for tr in range(ntr):
                u = sim.make_election(nv, nc, nd)
                pu = sim.add_noise(u, float(t))

                cm = u.mean(axis=0)
                sw_rand = float(cm.mean())
                sw_opt = float(cm.max())

                for name in method_names:
                    winner_h = int(sim.METHODS[name](pu, float(l)))
                    vse_h = float(sim.vse(float(cm[winner_h]), sw_rand, sw_opt))
                    acc_h[name] += vse_h
                    trial_h[name][li, ti, tr] = vse_h

                    winner_s = int(
                        sim._method_winner(
                            name,
                            pu,
                            float(l),
                            use_strategy=True,
                            strategy_share=strategy_share,
                        )
                    )
                    vse_s = float(sim.vse(float(cm[winner_s]), sw_rand, sw_opt))
                    acc_s[name] += vse_s
                    trial_s[name][li, ti, tr] = vse_s

            for name in method_names:
                res_h[name][li, ti] = acc_h[name] / ntr
                res_s[name][li, ti] = acc_s[name] / ntr

            done += 1
            if done % 8 == 0 or done == total_cells:
                elapsed = time.time() - t0
                eta = (total_cells - done) * (elapsed / done)
                print(
                    f"{done}/{total_cells} cells "
                    f"| elapsed {elapsed:.1f}s "
                    f"| eta {eta:.1f}s"
                )

    return {
        "method_names": method_names,
        "t_vals": t_vals.tolist(),
        "l_vals": l_vals.tolist(),
        "res_h": {m: res_h[m].tolist() for m in method_names},
        "trial_h": {m: trial_h[m].tolist() for m in method_names},
        "res_s": {m: res_s[m].tolist() for m in method_names},
        "trial_s": {m: trial_s[m].tolist() for m in method_names},
    }


def build_entry(
    cache_version: int,
    cache_key_value: str,
    signature: str,
    created_at_ms: int,
    nv: int,
    nc: int,
    ntr: int,
    ng: int,
    nd: int,
    enabled_methods: Dict[str, bool],
    dual_result: dict,
) -> dict:
    mode_honest = {
        "res": dual_result["res_h"],
        "tVals": dual_result["t_vals"],
        "lVals": dual_result["l_vals"],
        "trialVse": dual_result["trial_h"],
        "methods": enabled_methods,
        "useStrategy": False,
    }
    mode_strategic = {
        "res": dual_result["res_s"],
        "tVals": dual_result["t_vals"],
        "lVals": dual_result["l_vals"],
        "trialVse": dual_result["trial_s"],
        "methods": enabled_methods,
        "useStrategy": True,
    }

    return {
        "version": cache_version,
        "createdAt": created_at_ms,
        "cacheKey": cache_key_value,
        "signature": signature,
        "params": {
            "nv": nv,
            "nc": nc,
            "ntr": ntr,
            "ng": ng,
            "nd": nd,
            "enabledMethods": enabled_methods,
        },
        "modes": {
            "honest": mode_honest,
            "strategic": mode_strategic,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate default browser cache bundle JSON.")
    parser.add_argument("--nv", type=int, default=101)
    parser.add_argument("--nc", type=int, default=8)
    parser.add_argument("--ntr", type=int, default=200)
    parser.add_argument("--ng", type=int, default=8)
    parser.add_argument("--nd", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy-share", type=float, default=1.0)
    parser.add_argument("--cache-version", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/default-sim-cache.json"),
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write pretty JSON (larger file).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    enabled = dict(DEFAULT_ENABLED)
    signature = params_signature(args.nv, args.nc, args.ntr, args.ng, args.nd, enabled)
    key = cache_key(signature, args.cache_version)

    print("Generating dual-mode default cache...")
    dual = run_dual_mode(
        nv=args.nv,
        nc=args.nc,
        ntr=args.ntr,
        ng=args.ng,
        nd=args.nd,
        enabled_methods=enabled,
        strategy_share=args.strategy_share,
        seed=args.seed,
    )

    created_at_ms = int(time.time() * 1000)
    entry = build_entry(
        cache_version=args.cache_version,
        cache_key_value=key,
        signature=signature,
        created_at_ms=created_at_ms,
        nv=args.nv,
        nc=args.nc,
        ntr=args.ntr,
        ng=args.ng,
        nd=args.nd,
        enabled_methods=enabled,
        dual_result=dual,
    )

    payload = {
        "version": args.cache_version,
        "entries": {
            key: entry,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        else:
            json.dump(payload, f, ensure_ascii=True, separators=(",", ":"))

    print(f"Wrote {args.out}")
    print(f"Cache key: {key}")


if __name__ == "__main__":
    main()
