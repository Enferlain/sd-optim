import optuna
import argparse
from pathlib import Path


def analyze_optuna(db_path, study_name=None):
    path = Path(db_path)
    if not path.exists():
        print(f"Error: Database {db_path} not found.")
        return

    storage = f"sqlite:///{path.resolve()}"

    try:
        if study_name:
            study = optuna.load_study(study_name=study_name, storage=storage)
        else:
            # Load the latest study if none specified
            summaries = optuna.get_all_study_summaries(storage)
            if not summaries:
                print("No studies found in the database.")
                return
            # Sorting by id usually gets the latest, but let's be safe
            latest_summary = summaries[-1]
            study = optuna.load_study(study_name=latest_summary.study_name, storage=storage)
            print(f"Loading latest study: {study.study_name}")

        print(f"Analysis of study: {study.study_name}")
        print(f"  Total Trials: {len(study.trials)}")

        if study.best_trial:
            print(f"  Best Score: {study.best_value:.4f} (Trial #{study.best_trial.number})")

            # Parameter Importance
            print("\nParameter Importance:")
            try:
                importance = optuna.importance.get_param_importances(study)
                for param, value in importance.items():
                    print(f"  {param}: {value:.4f}")
            except Exception as e_imp:
                print(f"  Could not calculate importance: {e_imp}")
        else:
            print("  No completed trials found in this study.")

    except Exception as e:
        print(f"Error loading study: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Optuna databases.")
    parser.add_argument("db", help="Path to the .db file")
    parser.add_argument("--study", help="Optional name of the study to analyze")
    args = parser.parse_args()
    analyze_optuna(args.db, args.study)
