"""
web.py — single entry point to launch PlanetExplorer.

Automatically downloads and preprocesses the Kepler dataset if missing,
then starts Flask on a single port serving both the API and frontend.

Usage:
    python web.py
    python web.py --port 8080 --no-browser
"""

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "data" / "processed" / "test.csv"


def ensure_dataset():
    if DATASET_PATH.exists():
        return
    print("Dataset not found. Downloading from NASA Exoplanet Archive...")
    from data.download import download_data, preprocess_data
    download_data(base_dir=ROOT)
    preprocess_data(base_dir=ROOT)
    print("Dataset ready.")


def main():
    parser = argparse.ArgumentParser(description="Launch PlanetExplorer")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    ensure_dataset()

    print("=" * 45)
    print("  PlanetExplorer")
    print(f"  http://localhost:{args.port}")
    print("  Press Ctrl+C to stop")
    print("=" * 45)

    if not args.no_browser:
        import threading
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    from backend.app import app
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
