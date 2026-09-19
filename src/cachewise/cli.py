import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

from cachewise.config import Settings
from cachewise.dataset import read_dataset, write_dataset
from cachewise.replay import run_replay


def main():
    parser = argparse.ArgumentParser(prog="cachewise")
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset", help="Write the seeded 1,000-request dataset")
    dataset.add_argument("--output", type=Path, default=Path("artifacts/dataset.jsonl"))
    dataset.add_argument("--seed", type=int, default=42)
    replay = commands.add_parser("replay", help="Run a live sequential uncached baseline")
    replay.add_argument("--dataset", type=Path, default=Path("artifacts/dataset.jsonl"))
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--api-url", default="http://127.0.0.1:8000")
    replay.add_argument("--limit", type=int)
    replay.add_argument("--split", choices=["tuning", "evaluation", "all"], default="all")
    replay.add_argument("--warmup-count", type=int, default=2)
    replay.add_argument("--serving-metadata", type=Path)
    args = parser.parse_args()
    if args.command == "dataset":
        rows = write_dataset(args.output, args.seed)
        print(f"Wrote {len(rows)} requests to {args.output}")
        return
    parsed = urlsplit(args.api_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        parser.error("--api-url must be an HTTP(S) URL without embedded credentials")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.warmup_count < 0:
        parser.error("--warmup-count must be nonnegative")
    rows = read_dataset(args.dataset)
    if args.split != "all":
        rows = [row for row in rows if row.split == args.split]
    if args.limit:
        rows = rows[: args.limit]
    metadata = json.loads(args.serving_metadata.read_text()) if args.serving_metadata else None
    aggregate = asyncio.run(
        run_replay(
            rows,
            args.output,
            Settings(),
            args.api_url,
            args.warmup_count,
            metadata,
        )
    )
    print(json.dumps(aggregate, indent=2))
    if aggregate["failures"]:
        raise SystemExit(1)
