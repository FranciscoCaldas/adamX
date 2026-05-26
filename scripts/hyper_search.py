import argparse
import itertools
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Search")
    parser.add_argument("--dataset", "-d", default="mnist", help="Dataset name (-dc)")
    parser.add_argument("--optimizer", "-o", default="adamx", help="Optimizer name (-oc)")
    
    # Default search space (can be modified here or extended via CLI)
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-4,5e-4,5e-3, 1e-3, 1e-2], help="Learning rates to test")
    parser.add_argument("--batches", nargs="+", type=int, default=[32, 64, 128], help="Batch sizes to test")
    args = parser.parse_args()
    
    # 3 different seeds
    seeds = [42, 123, 999]
    
    # Betas to test
    betas_spaces = [(0.9, 0.999), (0.9, 0.99)]
    
    script_path = Path(__file__).parent / "run.py"
    
    runs = list(itertools.product(args.lrs, args.batches, betas_spaces, seeds))
    total_runs = len(runs)
    
    print(f"Starting hyperparameter search! Total runs: {total_runs}",flush=True)
    
    for i, (lr, batch, betas, seed) in enumerate(runs, 1):

        cmd = [
            sys.executable, str(script_path),
            "-dc", args.dataset,
            "-oc", args.optimizer,
            "--lr", str(lr),
            "--batch-size", str(batch),
            "--betas", str(betas[0]), str(betas[1]),
            "--seed", str(seed)
        ]
        
        print(f"\n--- Run {i}/{total_runs} ---",flush=True)
        print(f"Parameters: lr={lr}, batch_size={batch}, betas={betas}, seed={seed}",flush=True)
        print(f"Command: {' '.join(cmd)}",flush=True)
        
        import os
        env = os.environ.copy()
        root_dir = str(script_path.parent.parent)
        env["PYTHONPATH"] = f"{root_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
        
        try:
            # We use check=True so if a run fails, it throws a traceback.
            # You can change check to False if you want it to continue on error.
            subprocess.run(cmd, check=True, env=env, cwd=root_dir)
            #print(f"Run {i} completed successfully!",flush=True)
        except subprocess.CalledProcessError as e:
            print(f"Run {i} failed with return code {e.returncode}. Skipping...",flush=True)

if __name__ == "__main__":
    main()
