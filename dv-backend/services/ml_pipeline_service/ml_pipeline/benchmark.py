import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

import requests
from joblib import dump

from .data_loader import CSVLoader
from .pipeline import run_pipeline


@dataclass
class Dataset:
    name: str
    dataset_path: str
    target_field: str
    openml_id: int


BASE_HOST = "https://www.openml.org"
# returns JSON with a 'url' pointing to /data/download/...
META_URL_TPL = BASE_HOST + "/es/data/data/{did}"
BASE_PATH = os.path.join(os.getcwd(), "simulation_datasets")
os.makedirs(BASE_PATH, exist_ok=True)

DATASETS: List[Dataset] = [
    Dataset("diabetes", os.path.join(BASE_PATH, "diabetes.csv"), "class", 37),
    Dataset("blood-transfusion-service-center", os.path.join(BASE_PATH,
            "blood-transfusion-service-center.csv"), "Class", 1464),
    Dataset("dresses-sales", os.path.join(BASE_PATH,
            "dresses-sales.csv"), "Class", 23381),
    Dataset("credit-g", os.path.join(BASE_PATH, "credit-g.csv"), "class", 31),
    Dataset("bank-marketing", os.path.join(BASE_PATH,
            "bank-marketing.csv"), "Class", 1461),
    Dataset("steel-plates-fault", os.path.join(BASE_PATH,
            "steel-plates-fault.csv"), "Class", 1504),
    Dataset("cylinder-bands", os.path.join(BASE_PATH,
            "cylinder-bands.csv"), "band_type", 6332),
]


def download_one(ds: Dataset, session: requests.Session, retries: int = 3, timeout: int = 60) -> None:
    """
    Download a single dataset to ds.dataset_path using:
      metadata -> /data/download/... URL -> build /data/get_csv<suffix>
      where <suffix> is the part AFTER '/download' (including leading slash and query if present).
    """
    os.makedirs(os.path.dirname(ds.dataset_path), exist_ok=True)

    # Resolve download URL from metadata (keep all logic local for readability)
    meta_url = META_URL_TPL.format(did=ds.openml_id)
    get_csv_url: Optional[str] = None

    try:
        r = session.get(meta_url, headers={
                        "Accept": "application/json"}, timeout=30)
        r.raise_for_status()
        meta = r.json()

        # Minimal recursive search for a string URL (prefer key 'url', else any containing '/data/download/')
        def find_url(o) -> Optional[str]:
            if isinstance(o, dict):
                if "url" in o and isinstance(o["url"], str):
                    return o["url"]
                for v in o.values():
                    hit = find_url(v)
                    if hit:
                        return hit
            elif isinstance(o, list):
                for it in o:
                    hit = find_url(it)
                    if hit:
                        return hit
            elif isinstance(o, str) and "/data/download/" in o:
                return o
            return None

        download_like = find_url(meta)
        if download_like:
            # Normalize to absolute
            if download_like.startswith("/"):
                download_like = BASE_HOST + download_like

            p = urlparse(download_like)
            idx = p.path.find("/download")
            if idx != -1:
                # keep leading slash below
                suffix = p.path[idx + len("/download"):]
                if not suffix.startswith("/"):
                    suffix = "/" + suffix
                # preserve query if present
                if p.query:
                    suffix = suffix + "?" + p.query
                get_csv_url = BASE_HOST + "/data/get_csv" + suffix
    except Exception as e:
        print(
            f"⚠️  Metadata resolution failed for {ds.name} (ID {ds.openml_id}): {e}")

    if not get_csv_url:
        print(
            f"✗ Could not build /data/get_csv URL for {ds.name} (ID {ds.openml_id}). Skipping.")
        return

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with session.get(get_csv_url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with open(ds.dataset_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 14):
                        if chunk:
                            f.write(chunk)
            print(
                f"✓ Downloaded: {ds.name} (ID {ds.openml_id}) -> {ds.dataset_path}")
            return
        except Exception as e:
            last_err = e
            print(
                f"Attempt {attempt}/{retries} failed for {ds.name} using {get_csv_url}: {e}")
            if attempt < retries:
                time.sleep(2 * attempt)

    print(
        f"✗ Giving up on {ds.name} (ID {ds.openml_id}). Last error: {last_err}")


def run_benchmark(datasets: List[Dataset]) -> None:
    """
    Run benchmarking on a list of datasets.
    Expects each dataset CSV to exist at dataset.dataset_path.
    """
    import pandas as pd  # defer heavy import to when it’s actually needed
    results_dir = os.path.join(os.getcwd(), "benchmark_results")
    os.makedirs(results_dir, exist_ok=True)

    for ds in datasets:
        if not os.path.exists(ds.dataset_path):
            print(
                f"⚠️  Missing file for {ds.name}: {ds.dataset_path}. Skipping.")
            continue

        df = CSVLoader.load_local(ds.dataset_path)

        metrics, model = run_pipeline(df, target_field=ds.target_field)

        # Metrics as txt (dataset_target.txt)
        metrics_path = os.path.join(
            results_dir, f"{ds.name}_{ds.target_field}.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(metrics, indent=2, sort_keys=True, default=str))

        # Model as joblib (dataset_target_model.joblib)
        model_path = os.path.join(
            results_dir, f"{ds.name}_{ds.target_field}_model.joblib")
        dump(model, model_path)

        print(f"✓ Saved metrics -> {metrics_path}")
        print(f"✓ Saved model   -> {model_path}")


def main():
    # Usage:
    #   python script.py --download [NAME/ID ...]
    #   python script.py --benchmark [NAME/ID ...]
    args = sys.argv[1:]
    if not args:
        print("Nothing to do. Use --download or --benchmark (optionally followed by names/IDs).")
        return

    mode, *filters = args
    session = requests.Session()

    # Inline filtering
    lowered = {str(x).lower() for x in filters}

    def selected(ds: Dataset) -> bool:
        return not lowered or ds.name.lower() in lowered or str(ds.openml_id) in lowered

    targets = [ds for ds in DATASETS if selected(ds)]
    if filters and not targets:
        print("No matching datasets for given filters. Exiting.")
        return

    if mode == "--download":
        for ds in targets:
            download_one(ds, session=session)
    elif mode == "--benchmark":
        run_benchmark(targets)
    else:
        print(f"Unknown mode: {mode}. Use --download or --benchmark.")


if __name__ == "__main__":
    main()
