#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

# ------------------------------ USER SETTINGS ------------------------------ #
# Define your default batch jobs here. Each tuple is:
# ("<input video dir or file>", "<output parent dir>")
DEFAULT_JOBS: list[tuple[str, str]] = [
    # Example:
    # ("/absolute/path/to/input_dir_1", "/absolute/path/to/output_dir_1"),
    # ("/absolute/path/to/input_dir_2", "/absolute/path/to/output_dir_2"),
]

# Optional defaults used when running this script with no CLI arguments.
DEFAULT_CONFIG = "config/config.yaml"
DEFAULT_OVERRIDE_OUTPUT = False
DEFAULT_SKIP_EXISTING = False
# -------------------------------------------------------------------------- #


def parse_pair(raw_pair: str) -> tuple[str, str]:
    if ":" not in raw_pair:
        raise ValueError(f"Invalid pair '{raw_pair}'. Expected format INPUT_DIR:OUTPUT_DIR")
    input_dir, output_dir = raw_pair.split(":", 1)
    input_dir = input_dir.strip()
    output_dir = output_dir.strip()
    if not input_dir or not output_dir:
        raise ValueError(f"Invalid pair '{raw_pair}'. Input and output must both be non-empty.")
    return input_dir, output_dir


def load_pairs_from_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            pairs.append(parse_pair(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_batch.py",
        description="Run Polytrack batch jobs using input/output directory pairs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to the Polytrack config YAML file.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Input/output pair in the form INPUT_DIR:OUTPUT_DIR. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pairs-file",
        type=str,
        default=None,
        help="Optional text file with one INPUT_DIR:OUTPUT_DIR pair per line.",
    )
    parser.add_argument(
        "--override-output",
        action="store_true",
        default=DEFAULT_OVERRIDE_OUTPUT,
        help="Pass through --override-output to src/main.py.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=DEFAULT_SKIP_EXISTING,
        help="Pass through --skip-existing to src/main.py.",
    )

    args = parser.parse_args()

    pairs: list[tuple[str, str]] = []
    for raw_pair in args.pair:
        pairs.append(parse_pair(raw_pair))

    if args.pairs_file is not None:
        pairs.extend(load_pairs_from_file(Path(args.pairs_file)))

    if not pairs:
        pairs = list(DEFAULT_JOBS)

    if not pairs:
        parser.error(
            "No batch jobs provided. Define DEFAULT_JOBS in this script, or use --pair/--pairs-file."
        )

    repo_root = Path(__file__).resolve().parent.parent
    main_py = repo_root / "src" / "main.py"
    config_path = str((repo_root / args.config).resolve()) if not Path(args.config).is_absolute() else args.config

    for idx, (input_dir, output_dir) in enumerate(pairs, start=1):
        cmd = [
            sys.executable,
            str(main_py),
            "--config",
            config_path,
            "--input-dir",
            input_dir,
            "--output-dir",
            output_dir,
        ]
        if args.override_output:
            cmd.append("--override-output")
        if args.skip_existing:
            cmd.append("--skip-existing")

        print(f"[{idx}/{len(pairs)}] Running: input={input_dir} output={output_dir}")
        result = subprocess.run(cmd, cwd=str(repo_root))
        if result.returncode != 0:
            print(f"Batch stopped at pair {idx} with exit code {result.returncode}.", file=sys.stderr)
            return result.returncode

    print("Batch processing completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
