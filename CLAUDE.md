# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This git repository is rooted at the user's home directory (`/home/kali`). The `.gitignore` excludes everything (`*`) except explicitly allow-listed files, so only those are tracked — the rest of the home directory (Desktop, Documents, other project folders, dotfiles, etc.) is local clutter, not part of this project. When making changes, only touch files that are actually tracked or explicitly relevant to the task at hand; do not assume other files/directories in the home directory are part of this codebase.

If you add new source files to this project, remember they must be explicitly un-ignored in `.gitignore` (e.g. `!newfile.py`) or `git add` will refuse to stage them.

## Code architecture

The repository holds two unrelated pieces: the original port scanner, and a generic ML training/serving pipeline added later (PR #1, `add-ml-training-scripts`). They don't share code or data — the ML files are boilerplate for an arbitrary tabular classifier, not specific to port scanning.

- `port_scanner.py` — a standalone, multithreaded TCP port scanner (stdlib only, no third-party dependencies). It resolves the target host, parses a port spec (comma-separated ports and/or ranges, e.g. `22,80,1-1024`), then uses a `ThreadPoolExecutor` to attempt a connection to each port concurrently, reporting open ports and their best-guess service name (via `socket.getservbyport`).

  This tool is for authorized security testing only (scanning hosts you own or have explicit permission to test), per the docstring at the top of the file.

- `feature_engineer.py` — `FeatureEngineer`, a scikit-learn-compatible transformer (`BaseEstimator`/`TransformerMixin`) that imputes missing values, scales numeric columns, and one-hot-encodes categorical columns, auto-detected by dtype. Generic/reusable — not specific to any particular dataset; the `_create_features` method is a stub meant to be customized per use case.
- `advanced_trainer.py` — `AdvancedTrainer`, which runs Optuna hyperparameter search over an XGBoost classifier (cross-validated ROC-AUC objective), fits the tuned pipeline on a train split, evaluates on a held-out test split, and serializes the fitted pipeline with `joblib`. Expects a preprocessor (e.g. `FeatureEngineer` or a `ColumnTransformer`) passed in, not tied to `feature_engineer.py` directly.
- `app.py` — a FastAPI inference service that loads a `joblib`-serialized pipeline (from `artifacts/model_pipeline.pkl`) on startup, and exposes `GET /` (health check reporting model-load state) and `POST /predict` (runs the loaded pipeline against a JSON payload matching the `ModelInput` schema — currently a demo schema of `age`/`salary`/`score_a`/`score_b`/`tenure`/`department`).

These three files are demonstration/boilerplate code (dummy `age`/`salary`/`department`-style example data in their `__main__` blocks) — no dataset, trained model artifact, or task-specific wiring between them exists in this repo yet.

## Commands

Run the scanner directly with Python 3 (no build step, no install required):

```bash
python3 port_scanner.py <target> [-p PORTS] [-t THREADS] [-w TIMEOUT]
```

Examples:
```bash
python3 port_scanner.py 192.168.1.10                 # scan default ports 1-1024
python3 port_scanner.py example.com -p 22,80,443      # scan specific ports
python3 port_scanner.py 10.0.0.5 -p 1-65535 -t 500 -w 0.5   # full scan, more threads, shorter timeout
```

Run tests with:

```bash
python3 -m pytest test_port_scanner.py -v
```

`scan_port` tests mock `socket.socket`/`socket.getservbyport` so they run without real network access. There is no linter or build configuration in this repository.

### ML pipeline files

`feature_engineer.py`, `advanced_trainer.py`, and `app.py` have no `requirements.txt`, tests, or CI coverage yet, and depend on third-party packages not needed by `port_scanner.py`: `pandas`, `numpy`, `scikit-learn`, `optuna`, `xgboost`, `joblib`, `fastapi`, `pydantic`, `uvicorn`. Each file's `__main__`/example block is runnable standalone against synthetic data once those are installed, e.g.:

```bash
python3 feature_engineer.py       # fits/transforms a small dummy DataFrame
python3 advanced_trainer.py       # trains on sklearn's make_classification, saves artifacts/model_pipeline.pkl
python3 app.py                    # serves the saved pipeline via uvicorn on :8000 (needs artifacts/model_pipeline.pkl first)
```
