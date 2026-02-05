# Run Data Specification

## JSONL Trial Logs

Located at: `logs/<run_dir>/run_<timestamp>_<sampler>_trials.jsonl`

Each line is a JSON object representing a single trial:

```json
{
  "trial_number": 0,
  "target": 0.85,
  "params": {
    "UNET_IN01_alpha": 0.1,
    "UNET_MID01_alpha": 0.5
  },
  "state": "COMPLETE",
  "datetime": {
    "datetime": "2026-02-05T02:00:00.000000",
    "elapsed_seconds": 120.5
  }
}
```

## Optuna Database

Located at: `optuna_db/*.db` (SQLite format)

### Core Tables

- `studies`: Stores study metadata and directions.
- `trials`: Stores trial status and values.
- `trial_params`: Stores the actual parameter suggestions for each trial.
- `trial_user_attributes`: Stores custom metadata set during the run.

### Custom User Attributes

- `elapsed_seconds`: Total time since optimization started.
- `timestamp`: Unix timestamp of the trial completion.
- `model_path`: (Optional) Absolute path to the merged model for that trial.
- `iteration`: The iteration number.
