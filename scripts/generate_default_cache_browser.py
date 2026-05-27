#!/usr/bin/env python3
"""Generate bundled default cache by running the browser simulation headlessly.

Workflow:
1. Starts `npx http-server` in the repository root.
2. Opens the app in headless Chromium (Playwright).
3. Clicks the Run Simulation button and waits for completion.
4. Reads the default cache entry from localStorage.
5. Writes `output/default-sim-cache.json` in bundled format:
   {"version": 3, "entries": {"<cacheKey>": <entry>}}

Requirements:
- Node.js + npx
- http-server (auto-fetched by npx)
- Python package: playwright
- Browser install once: `python -m playwright install chromium`
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate output/default-sim-cache.json from browser run.")
    parser.add_argument("--port", type=int, default=4173, help="Local server port.")
    parser.add_argument("--host", default="127.0.0.1", help="Local server host.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/default-sim-cache.json"),
        help="Output cache bundle path.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=420,
        help="Max seconds to wait for simulation completion.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write pretty JSON instead of compact JSON.",
    )
    return parser.parse_args()


def wait_for_http(url: str, timeout_seconds: int = 30) -> None:
    t0 = time.time()
    while True:
        try:
            with urlopen(url, timeout=2) as resp:  # nosec B310: local trusted URL
                if resp.status < 500:
                    return
        except URLError:
            pass

        if time.time() - t0 > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for server at {url}")
        time.sleep(0.3)


def run_browser_export(base_url: str, timeout_seconds: int) -> dict:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Playwright is required. Install with `pip install playwright` and run "
            "`python -m playwright install chromium`."
        ) from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page_errors: list[str] = []
            console_errors: list[str] = []

            page.on("pageerror", lambda exc: page_errors.append(str(exc)))

            def _on_console(msg) -> None:
                try:
                    if msg.type == "error":
                        console_errors.append(msg.text)
                except Exception:
                    pass

            page.on("console", _on_console)
            page.goto(base_url, wait_until="networkidle")

            # Wait for startup auto-run (if any) to finish, then force a fresh
            # uncached run for deterministic bundle generation.
            page.wait_for_function(
                "() => { const btn = document.getElementById('run-btn'); return !!btn && !btn.disabled; }",
                timeout=timeout_seconds * 1000,
            )

            page.evaluate(
                """
                () => {
                    localStorage.clear();
                    window.__genRunDone = false;
                    window.__genRunError = '';
                    if (typeof window.runSimulation !== 'function') {
                        throw new Error('window.runSimulation is not available.');
                    }
                    window.runSimulation({
                        requestedMode: 'honest',
                        background: false,
                        useCache: false,
                        scrollToResults: false,
                        forceRecompute: true,
                    }).then(
                        () => { window.__genRunDone = true; },
                        (err) => { window.__genRunError = String(err); }
                    );
                }
                """
            )

            t0 = time.time()
            last_status = ""
            while True:
                status_text = ""
                if page.locator("#sim-status").count():
                    status_text = page.locator("#sim-status").inner_text().strip()

                if status_text and status_text != last_status:
                    print(f"status: {status_text}")
                    last_status = status_text

                run_state = page.evaluate(
                    """
                    () => ({
                        done: Boolean(window.__genRunDone),
                        err: String(window.__genRunError || ''),
                    })
                    """
                )

                if run_state.get("err"):
                    raise RuntimeError(f"Simulation run failed in browser app: {run_state['err']}")

                if run_state.get("done") and "Done (both modes)" in status_text:
                    break

                if "Simulation failed" in status_text:
                    err_lines = []
                    if page_errors:
                        err_lines.append(f"pageerror: {page_errors[-1]}")
                    if console_errors:
                        err_lines.append(f"console: {console_errors[-1]}")
                    detail = " | ".join(err_lines) if err_lines else "No browser error details captured."
                    raise RuntimeError(f"Simulation failed in browser app. {detail}")

                if (time.time() - t0) > timeout_seconds:
                    raise TimeoutError(
                        f"Simulation did not complete in {timeout_seconds}s. Current status: {status_text!r}"
                    )

                time.sleep(1)

            entry = page.evaluate(
                r"""
                () => {
                    if (typeof window.exportCurrentCacheEntryFromMemory === 'function') {
                        const inMemory = window.exportCurrentCacheEntryFromMemory();
                        if (inMemory && inMemory.cacheKey) return inMemory;
                    }

                    const keys = Object.keys(localStorage);
                    const defaultKey = keys.find(k => /^svs-cache-v\d+:default$/.test(k));
                    if (!defaultKey) {
                        throw new Error('Could not export cache entry from memory and could not find default cache key in localStorage.');
                    }
                    const raw = localStorage.getItem(defaultKey);
                    if (!raw) {
                        throw new Error('Default cache key exists but had no payload.');
                    }
                    const parsed = JSON.parse(raw);
                    if (!parsed || !parsed.cacheKey) {
                        throw new Error('Default cache payload missing cacheKey.');
                    }
                    return parsed;
                }
                """
            )
            return entry
        finally:
            browser.close()


def write_bundle(out_path: Path, entry: dict, pretty: bool) -> None:
    payload = {
        "version": int(entry.get("version", 3)),
        "entries": {
            str(entry["cacheKey"]): entry,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        else:
            json.dump(payload, f, ensure_ascii=True, separators=(",", ":"))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_url = f"http://{args.host}:{args.port}/index.html"

    npx_exe = shutil.which("npx") or shutil.which("npx.cmd")
    if npx_exe is None:
        print("Error: npx not found in PATH. Install Node.js first.", file=sys.stderr)
        return 1

    server_cmd = [
        npx_exe,
        "--yes",
        "http-server",
        str(repo_root),
        "-a",
        args.host,
        "-p",
        str(args.port),
        "-c-1",
        "--silent",
    ]

    print(f"Starting server: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(  # noqa: S603,S607 - controlled local command
        server_cmd,
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        wait_for_http(base_url, timeout_seconds=30)
        print(f"Server ready at {base_url}")

        entry = run_browser_export(base_url, timeout_seconds=args.timeout_seconds)
        write_bundle(args.out, entry, pretty=args.pretty)

        print(f"Wrote {args.out}")
        print(f"Cache key: {entry['cacheKey']}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
