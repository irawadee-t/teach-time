"""
Run grading in parallel with multiple API keys.

Splits input into chunks and runs grade_responses.py on each chunk
with a different API key, then combines results.

Usage:
    # Split input into 2 chunks, run with 2 API keys
    python -m evals.synthetic.grade_parallel \
        --input evals/synthetic/pipeline_outputs/20251208_182025/01_generated.jsonl \
        --output evals/synthetic/pipeline_outputs/20251208_182025/02_graded.jsonl \
        --api-keys "key1,key2"

    # Or use environment variables ANTHROPIC_API_KEY_1, ANTHROPIC_API_KEY_2, etc.
    python -m evals.synthetic.grade_parallel \
        --input ... --output ... --num-workers 2
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List


def load_completed_ids(output_path: Path) -> set:
    """Load already graded task IDs from output file."""
    completed = set()
    if output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    task_id = f"{r['sample_id']}_{r['persona_name']}"
                    completed.add(task_id)
                except:
                    continue
    return completed


def split_jsonl(input_path: Path, num_chunks: int, exclude_ids: set = None) -> List[Path]:
    """Split a JSONL file into N chunks, returns paths to chunk files."""
    lines = []
    with open(input_path, 'r') as f:
        for line in f:
            if exclude_ids:
                try:
                    r = json.loads(line)
                    task_id = f"{r['sample_id']}_{r['persona_name']}"
                    if task_id in exclude_ids:
                        continue
                except:
                    pass
            lines.append(line)

    if not lines:
        print("No items to process!")
        return []

    chunk_size = len(lines) // num_chunks
    remainder = len(lines) % num_chunks

    chunk_paths = []
    start = 0

    for i in range(num_chunks):
        # Distribute remainder across first chunks
        size = chunk_size + (1 if i < remainder else 0)
        end = start + size

        chunk_path = input_path.parent / f"{input_path.stem}_chunk{i}.jsonl"
        with open(chunk_path, 'w') as f:
            f.writelines(lines[start:end])

        chunk_paths.append(chunk_path)
        print(f"Chunk {i}: {size} items -> {chunk_path}")
        start = end

    return chunk_paths


def combine_jsonl(chunk_outputs: List[Path], output_path: Path, append: bool = False):
    """Combine multiple JSONL files into one."""
    mode = 'a' if append else 'w'
    count = 0
    with open(output_path, mode) as out:
        for chunk_path in chunk_outputs:
            if chunk_path.exists():
                with open(chunk_path, 'r') as f:
                    for line in f:
                        out.write(line)
                        count += 1
    print(f"{'Appended' if append else 'Combined'} {count} items from {len(chunk_outputs)} chunks -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run grading in parallel with multiple API keys")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with generated responses")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file for combined graded responses")
    parser.add_argument("--api-keys", type=str, default=None,
                        help="Comma-separated API keys (or use ANTHROPIC_API_KEY_1, _2, etc.)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of parallel workers (default: 2)")
    parser.add_argument("--concurrency", type=str, default=None,
                        help="Concurrency per worker, comma-separated (e.g., '150,50,50')")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing chunk outputs")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Get API keys
    if args.api_keys:
        api_keys = [k.strip() for k in args.api_keys.split(",")]
    else:
        # Look for ANTHROPIC_API_KEY_1, ANTHROPIC_API_KEY_2, etc.
        api_keys = []
        for i in range(1, args.num_workers + 1):
            key = os.environ.get(f"ANTHROPIC_API_KEY_{i}")
            if key:
                api_keys.append(key)

        # Fall back to single key duplicated
        if not api_keys:
            base_key = os.environ.get("ANTHROPIC_API_KEY")
            if base_key:
                print(f"Warning: Using same API key for all {args.num_workers} workers")
                api_keys = [base_key] * args.num_workers
            else:
                print("Error: No API keys found. Set ANTHROPIC_API_KEY_1, _2, etc.")
                sys.exit(1)

    num_workers = len(api_keys)
    print(f"Running with {num_workers} parallel workers")

    # Parse per-worker concurrency
    if args.concurrency:
        concurrencies = [int(c.strip()) for c in args.concurrency.split(",")]
        # Extend with last value if not enough
        while len(concurrencies) < num_workers:
            concurrencies.append(concurrencies[-1])
        print(f"Concurrency per worker: {concurrencies[:num_workers]}")
    else:
        concurrencies = [None] * num_workers  # Use default

    # Check for already completed items in main output
    exclude_ids = set()
    if args.resume and output_path.exists():
        exclude_ids = load_completed_ids(output_path)
        print(f"Resume: found {len(exclude_ids)} already graded in {output_path}")

    # Split input (excluding already completed)
    print(f"\nSplitting {input_path} into {num_workers} chunks...")
    chunk_inputs = split_jsonl(input_path, num_workers, exclude_ids)

    if not chunk_inputs:
        print("Nothing to grade!")
        return

    # Define chunk outputs
    chunk_outputs = [
        input_path.parent / f"{input_path.stem}_chunk{i}_graded.jsonl"
        for i in range(num_workers)
    ]

    # Launch parallel processes
    print(f"\nLaunching {num_workers} grading processes...")
    processes = []

    for i, (chunk_in, chunk_out, api_key, conc) in enumerate(zip(chunk_inputs, chunk_outputs, api_keys, concurrencies)):
        env = os.environ.copy()
        env["ANTHROPIC_API_KEY"] = api_key

        cmd = [
            sys.executable, "-m", "evals.synthetic.grade_responses",
            "--input", str(chunk_in),
            "--output", str(chunk_out),
        ]
        if conc is not None:
            cmd.extend(["--concurrency", str(conc)])
        if args.resume:
            cmd.append("--resume")

        conc_str = f" (concurrency={conc})" if conc else ""
        print(f"Worker {i}: grading {chunk_in.name} -> {chunk_out.name}{conc_str}")
        proc = subprocess.Popen(cmd, env=env)
        processes.append((i, proc))

    # Wait for all to complete
    print(f"\nWaiting for {num_workers} workers to complete...")
    for i, proc in processes:
        proc.wait()
        if proc.returncode != 0:
            print(f"Worker {i} failed with code {proc.returncode}")
        else:
            print(f"Worker {i} completed")

    # Combine outputs (append if resuming)
    print(f"\nCombining results...")
    combine_jsonl(chunk_outputs, output_path, append=args.resume)

    # Cleanup chunk files (optional - keep for debugging)
    # for p in chunk_inputs + chunk_outputs:
    #     p.unlink()

    print(f"\nDone! Output: {output_path}")


if __name__ == "__main__":
    main()
