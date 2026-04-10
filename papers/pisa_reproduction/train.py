from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from reproduce import DEFAULT_OUTPUT_DIR, run_ablation, run_full, run_random_reconstruction, run_sensor_fusion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PISA reproduction runner")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to store generated figures and metrics.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    random_parser = subparsers.add_parser("random", help="Run 4.1 random set reconstruction")
    random_parser.add_argument("--steps", type=int, default=2000)
    random_parser.add_argument("--trials-per-size", type=int, default=96)

    fusion_parser = subparsers.add_parser("fusion", help="Run 4.2 synthetic sensor fusion")
    fusion_parser.add_argument("--autoencoder-steps", type=int, default=1500)
    fusion_parser.add_argument("--detector-steps", type=int, default=800)
    fusion_parser.add_argument("--evaluation-scenes", type=int, default=32)
    fusion_parser.add_argument("--layers", type=int, default=2)

    ablation_parser = subparsers.add_parser("ablation", help="Run PISA ablation variants")
    ablation_parser.add_argument("--steps", type=int, default=1500)
    ablation_parser.add_argument("--trials-per-size", type=int, default=64)

    full_parser = subparsers.add_parser("full", help="Run the full reproduction suite")
    full_parser.add_argument("--random-steps", type=int, default=2000)
    full_parser.add_argument("--fusion-steps", type=int, default=1500)
    full_parser.add_argument("--detector-steps", type=int, default=800)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    if args.command == "random":
        run_random_reconstruction(output_dir / "random_reconstruction", steps=args.steps, trials_per_size=args.trials_per_size)
        return
    if args.command == "fusion":
        run_sensor_fusion(
            output_dir / "sensor_fusion",
            autoencoder_steps=args.autoencoder_steps,
            detector_steps=args.detector_steps,
            evaluation_scenes=args.evaluation_scenes,
            layers=args.layers,
        )
        return
    if args.command == "ablation":
        run_ablation(output_dir / "ablation", steps=args.steps, trials_per_size=args.trials_per_size)
        return
    if args.command == "full":
        run_full(
            output_dir,
            random_steps=args.random_steps,
            fusion_steps=args.fusion_steps,
            detector_steps=args.detector_steps,
        )
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
