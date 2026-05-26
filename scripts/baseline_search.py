from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path


def list_config_names(config_dir: Path) -> list[str]:
    if not config_dir.exists():
        return []
    return sorted(path.stem for path in config_dir.glob("*.json") if path.is_file())


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    optimizer_config_dir = repo_root / "configs" / "optimizers"
    default_optimizers = list_config_names(optimizer_config_dir)

    parser = argparse.ArgumentParser(
        description="Run each optimizer with default config values across seeds."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mnist", "cifar10"],
        help="Dataset config names (-dc).",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=default_optimizers,
        help="Optimizer config names (-oc).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 999],
        help="Seeds to run.",
    )
    args = parser.parse_args()

    if not args.optimizers:
        raise SystemExit(f"No optimizer configs found in {optimizer_config_dir}.")

    script_path = repo_root / "scripts" / "run.py"
    runs = list(itertools.product(args.datasets, args.optimizers, args.seeds))
    total_runs = len(runs)

    print(f"Starting baseline sweep! Total runs: {total_runs}", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    for i, (dataset, optimizer, seed) in enumerate(runs, 1):
        cmd = [
            sys.executable,
            str(script_path),
            "-dc",
            dataset,
            "-oc",
            optimizer,
            "--seed",
            str(seed),
        ]

        print(f"\n--- Run {i}/{total_runs} ---", flush=True)
        print(
            f"Parameters: dataset={dataset}, optimizer={optimizer}, seed={seed}",
            flush=True,
        )
        print(f"Command: {' '.join(cmd)}", flush=True)

        try:
            subprocess.run(cmd, check=True, env=env, cwd=str(repo_root))
        except subprocess.CalledProcessError as exc:
            print(
                f"Run {i} failed with return code {exc.returncode}. Skipping...",
                flush=True,
            )


if __name__ == "__main__":
    main()
