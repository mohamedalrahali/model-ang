"""
Single entrypoint: run demo bootstrap if artifacts are missing, then start the
server with auto-reload.

Usage:
  python serve.py
  python serve.py --port 8080 --no-bootstrap
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _artifacts_ok() -> bool:
    return (
        (ROOT / "artifacts" / "best_model.joblib").is_file()
        and (ROOT / "artifacts" / "engagement" / "best_model_lightgbm.pkl").is_file()
        and (ROOT / "artifacts" / "gmm" / "gmm_model.joblib").is_file()
    )


def _run_bootstrap() -> None:
    script = ROOT / "scripts" / "bootstrap_demo.py"
    print("-> Generating demo artifacts (first run or incomplete folder)…")
    r = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jungle in English — ML Web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Do not run bootstrap_demo even if artifact files are missing.",
    )
    parser.add_argument("--no-reload", action="store_true", help="Disable code auto-reload.")
    args = parser.parse_args()

    if not args.no_bootstrap and not _artifacts_ok():
        _run_bootstrap()

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        app_dir=str(ROOT),
    )


if __name__ == "__main__":
    main()
