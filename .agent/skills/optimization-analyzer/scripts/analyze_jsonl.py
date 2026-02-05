import json
import argparse
import sys
from pathlib import Path


def analyze_jsonl(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} not found.")
        return

    trials = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    trials.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not trials:
        print("No trials found in the log file.")
        return

    # Filter to COMPLETE trials
    complete_trials = [t for t in trials if t.get("state") == "COMPLETE"]
    if not complete_trials:
        print("No complete trials found.")
        return

    # Find best trial
    best_trial = max(complete_trials, key=lambda x: x.get("target") or 0)

    # Calculate stats
    scores = [t.get("target") for t in complete_trials if t.get("target") is not None]
    avg_score = sum(scores) / len(scores) if scores else 0

    total_time = trials[-1].get("datetime", {}).get("elapsed_seconds", 0)
    avg_trial_time = total_time / len(trials) if trials else 0

    print(f"Analysis of {path.name}:")
    print(f"  Total Trials: {len(trials)} ({len(complete_trials)} complete)")
    print(
        f"  Best Score: {best_trial.get('target'):.4f} (Trial #{best_trial.get('trial_number')})"
    )
    print(f"  Average Score: {avg_score:.4f}")
    print(f"  Total Time: {total_time:.2f}s (Avg: {avg_trial_time:.2f}s/trial)")
    print("\nBest Parameters:")
    for k, v in best_trial.get("params", {}).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze JSONL trial logs.")
    parser.add_argument("file", help="Path to the .jsonl file")
    args = parser.parse_args()
    analyze_jsonl(args.file)
