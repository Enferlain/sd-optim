import json
import argparse
import sys
from pathlib import Path
import sqlite3
import math


def calculate_pearson(x, y):
    n = len(x)
    if n <= 1:
        return 0.0

    mu_x = sum(x) / n
    mu_y = sum(y) / n

    std_x = math.sqrt(sum((xi - mu_x) ** 2 for xi in x) / n)
    std_y = math.sqrt(sum((yi - mu_y) ** 2 for yi in y) / n)

    if std_x == 0 or std_y == 0:
        return 0.0

    covariance = sum((xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y)) / n
    return covariance / (std_x * std_y)


def get_data_from_jsonl(file_path):
    trials = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    trials.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return [t for t in trials if t.get("state") == "COMPLETE" and t.get("target") is not None]


def get_data_from_db(db_path, study_name=None):
    # We'll use a simple sqlite3 connection to avoids requiring optuna for basic correlation
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        if not study_name:
            cursor.execute("SELECT study_name FROM studies ORDER BY study_id DESC LIMIT 1")
            row = cursor.fetchone()
            if not row:
                return []
            study_name = row[0]
            print(f"Using latest study: {study_name}")

        # Get the study_id
        cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
        study_id = cursor.fetchone()[0]

        # Get trials with their values
        cursor.execute(
            """
            SELECT t.trial_id, v.value 
            FROM trials t
            JOIN trial_values v ON t.trial_id = v.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE'
        """,
            (study_id,),
        )
        trials_meta = cursor.fetchall()

        data = []
        for trial_id, value in trials_meta:
            # Get params for this trial
            cursor.execute(
                """
                SELECT param_name, param_value 
                FROM trial_params 
                WHERE trial_id = ?
            """,
                (trial_id,),
            )
            params = {row[0]: row[1] for row in cursor.fetchall()}
            data.append({"target": value, "params": params})

        return data
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Calculate parameter correlations with the target score.")
    parser.add_argument("source", help="Path to .jsonl or .db file")
    parser.add_argument("--study", help="Study name (for .db files)")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: {source_path} not found.")
        sys.exit(1)

    if source_path.suffix == ".jsonl":
        data = get_data_from_jsonl(source_path)
    elif source_path.suffix == ".db":
        data = get_data_from_db(source_path, args.study)
    else:
        print("Error: Source must be .jsonl or .db")
        sys.exit(1)

    if not data:
        print("No complete trials found.")
        sys.exit(1)

    # Extract score list
    scores = [t["target"] for t in data]
    all_params = set()
    for t in data:
        all_params.update(t["params"].keys())

    correlations = []
    for param in sorted(all_params):
        param_values = []
        valid_scores = []
        for t in data:
            if param in t["params"] and isinstance(t["params"][param], (int, float)):
                param_values.append(t["params"][param])
                valid_scores.append(t["target"])

        if len(param_values) > 5:  # Minimum samples for a meaningful correlation
            corr = calculate_pearson(param_values, valid_scores)
            correlations.append((param, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"Correlation Analysis for {source_path.name} ({len(scores)} trials):")
    print(f"{'Parameter':<40} | {'Correlation':<12}")
    print("-" * 55)
    for param, corr in correlations:
        print(f"{param:<40} | {corr:>11.4f}")


if __name__ == "__main__":
    main()
