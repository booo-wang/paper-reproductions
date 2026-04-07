from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from reproduce import main as reproduce_main


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "results"
    reproduce_main(output_dir=output_dir)
