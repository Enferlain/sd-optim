import json
import argparse
import sys
from pathlib import Path
import sqlite3
import statistics


def get_stats_from_jsonl(file_path):
    trials = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    trials.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    complete = [
        t
        for t in trials
        if t.get("state") == "COMPLETE" and t.get("target") is not None
    ]
    if not complete:
        return None

    scores = [t["target"] for t in complete]
    best_trial = max(complete, key=lambda x: x["target"])

    return {
        "name": Path(file_path).name,
        "total": len(trials),
        "complete": len(complete),
        "best": best_trial["target"],
        "avg": statistics.mean(scores),
        "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "best_params": best_trial["params"],
    }


def get_stats_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT study_name, study_id FROM studies ORDER BY study_id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if not row:
            return None
        study_name, study_id = row

        cursor.execute(
            """
            SELECT t.trial_id, v.value 
            FROM trials t
            JOIN trial_values v ON t.trial_id = v.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE'
        """,
            (study_id,),
        )
        trials = cursor.fetchall()
        if not trials:
            return None

        scores = [t[1] for t in trials]
        best_val = max(scores)
        best_trial_id = trials[scores.index(best_val)][0]

        cursor.execute(
            "SELECT param_name, param_value FROM trial_params WHERE trial_id = ?",
            (best_trial_id,),
        )
        best_params = {r[0]: r[1] for r in cursor.fetchall()}

        return {
            "name": f"{Path(db_path).name} [{study_name}]",
            "total": len(trials),  # For DB we usually just get the complete ones easily
            "complete": len(trials),
            "best": best_val,
            "avg": statistics.mean(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "best_params": best_params,
        }
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Compare multiple optimization runs.")
    parser.add_argument("sources", nargs="+", help="Paths to .jsonl or .db files")
    args = parser.parse_args()

    results = []
    for source in args.sources:
        path = Path(source)
        if not path.exists():
            print(f"Warning: {source} not found. Skipping.")
            continue

        stats = None
        if path.suffix == ".jsonl":
            stats = get_stats_from_jsonl(path)
        elif path.suffix == ".db":
            stats = get_stats_from_db(path)

        if stats:
            results.append(stats)

    if not results:
        print("No valid data found to compare.")
        return

    print("\n" + "=" * 80)
    print(f"{'Run Name':<40} | {'Best':<8} | {'Avg':<8} | {'Stdev':<8}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['name'][:40]:<40} | {r['best']:<8.4f} | {r['avg']:<8.4f} | {r['stdev']:<8.4f}"
        )

    print("=" * 80 + "\n")

    # If exactly two runs, show param diffs for best trials
    if len(results) == 2:
        r1, r2 = results
        print(f"Parameter Differences (Best Trial: {r1['name']} vs {r2['name']}):")
        all_keys = set(r1["best_params"].keys()) | set(r2["best_params"].keys())

        print(f"{'Parameter':<40} | {'Run 1':<12} | {'Run 2':<12} | {'Diff':<12}")
        print("-" * 80)

        diffs = []
        for k in sorted(all_keys):
            v1 = r1["best_params"].get(k, "N/A")
            v2 = r2["best_params"].get(k, "N/A")
            diff = "N/A"
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                diff = abs(v1 - v2)
                diffs.append((k, v1, v2, diff))
            else:
                print(f"{k:<40} | {v1:<12} | {v2:<12} | {diff:<12}")

        # Highlight top 10 biggest differences
        diffs.sort(key=lambda x: x[3], reverse=True)
        for k, v1, v2, d in diffs:
            print(f"{k:<40} | {v1:<12.4f} | {v2:<12.4f} | {d:<12.4f}")


if __name__ == "__main__":
    main()
