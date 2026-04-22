#!/usr/bin/env python3
"""
Visual coding of AI-generated workshop images using Gemini vision.

Reads gallery_metadata.csv directly. For each consented image, sends the
image to Gemini with the codebook and asks it to apply binary codes.
Runs multiple times for reliability and aggregates by majority vote.

Output matches the format of the prompt coding pipeline so results can be
directly compared image-code vs prompt-code.

Usage:
    export GOOGLE_API_KEY=...

    python code_images.py \\
        --metadata_csv Data/Feb23/gallery_metadata.csv \\
        --image_dir    Data/Feb23/gallery_images \\
        --codebook_md  ImageAnalysis/codebook_images.md \\
        --out_dir      ImageAnalysis/results/images_coded \\
        --runs 3

Outputs (written to --out_dir/TIMESTAMP/):
    final_majority_vote.csv           — one row per image, binary code columns
    final_majority_vote_agreement.csv — agreement rate per image per code
    manifest.json                     — run metadata
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_code_names(codebook_path: Path) -> list[str]:
    text = read_text(codebook_path)
    codes = []
    pattern = re.compile(r"^###\s*(?:Code\s*)?\d+\s*[:\)]\s*(.+?)\s*$")
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            codes.append(m.group(1))
    if not codes:
        raise ValueError(f"No codes found in codebook: {codebook_path}")
    return codes


def to_snake_case(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_code_mapping(code_names: list[str]) -> dict[str, str]:
    mapping = {}
    for name in code_names:
        snake = to_snake_case(name)
        if snake in mapping.values():
            raise ValueError(f"Duplicate snake_case code: {snake}")
        mapping[name] = snake
    return mapping


def encode_image(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, mime


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_images(metadata_csv: Path, image_dir: Path) -> pd.DataFrame:
    """
    Load gallery_metadata.csv, filter to consented rows,
    and return DataFrame with columns: image_id, image_path, prompt.
    """
    df = pd.read_csv(metadata_csv, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    before = len(df)
    df = df[df["consent_all_yes"].isin(["1", "true", "yes"])].copy()
    log(f"Consent filter: {before} -> {len(df)} images")

    df["image_id"] = df["id"].str.strip()
    df["image_path"] = df["image_id"].apply(lambda x: image_dir / f"{int(x):04d}.png")

    missing = df[~df["image_path"].apply(lambda p: Path(p).exists())]
    if not missing.empty:
        log(f"Warning: {len(missing)} image files not found — skipping: {missing['image_id'].tolist()}")
        df = df[df["image_path"].apply(lambda p: Path(p).exists())].copy()

    # Keep id as prompt_id so compare_prompt_image.py can link images back to prompts
    df["prompt_id"] = df["id"].str.strip()
    return df[["image_id", "prompt_id", "image_path", "prompt"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a systematic qualitative researcher coding AI-generated images from \
a workshop on loneliness.

You will be shown one image. Apply every code in the codebook below.
For each code mark 1 if the feature is clearly visible in the image, \
0 if absent or unclear.

Rules:
- Code only what is VISUALLY PRESENT. Do not guess from context.
- When in doubt, code 0.
- Return ONLY a single line of comma-separated integers, one per code, \
  in the exact order the codes appear in the codebook.
- No header, no explanation, no extra text. Just the numbers.

Example output for a 5-code codebook: 1,0,1,0,0
"""


def build_prompt(codebook_text: str, n_codes: int) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Codebook\n\n{codebook_text}\n\n"
        f"Return exactly {n_codes} comma-separated values (0 or 1), "
        f"one per code, in codebook order. Nothing else."
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_model(
    client: genai.Client,
    model: str,
    prompt: str,
    image_data: str,
    mime_type: str,
    temperature: float,
    max_retries: int,
) -> str:
    contents = [
        types.Part.from_bytes(
            data=base64.b64decode(image_data),
            mime_type=mime_type,
        ),
        types.Part.from_text(text=prompt),
    ]
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            return response.text.strip()
        except Exception as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                log(f"  API error (attempt {attempt + 1}): {exc} — retrying in {wait}s")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_response(
    response_text: str,
    image_id: str,
    code_columns: list[str],
) -> dict:
    """Parse model response into {image_id, code1, code2, ...}."""
    text = re.sub(r"^```[^\n]*\n?", "", response_text.strip())
    text = re.sub(r"```\s*$", "", text.strip())

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        # Strip a non-numeric leading token (e.g. image_id)
        if parts and not parts[0].replace(".", "").lstrip("-").isdigit():
            parts = parts[1:]
        if len(parts) >= len(code_columns):
            parts = parts[:len(code_columns)]
            row = {"image_id": image_id}
            for col, val in zip(code_columns, parts):
                try:
                    row[col] = int(float(val))
                except (ValueError, TypeError):
                    row[col] = 0
            return row

    log(f"  Warning: could not parse response for image {image_id} — defaulting to 0")
    log(f"  Response was: {text[:200]}")
    return {"image_id": image_id, **{col: 0 for col in code_columns}}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_majority(
    images_df: pd.DataFrame,
    run_dfs: list[pd.DataFrame],
    code_columns: list[str],
) -> pd.DataFrame:
    base = images_df[["image_id", "prompt_id", "prompt"]].copy()
    for col in code_columns:
        votes = []
        for df in run_dfs:
            vals = (
                df.set_index("image_id")[col]
                .reindex(base["image_id"])
                .fillna(0)
                .astype(int)
            )
            votes.append(vals.values)
        stacked = np.array(votes).T
        base[col] = (stacked.sum(axis=1) > len(run_dfs) / 2).astype(int)
    return base


def compute_agreement(
    images_df: pd.DataFrame,
    run_dfs: list[pd.DataFrame],
    code_columns: list[str],
) -> pd.DataFrame:
    records = []
    ids = images_df["image_id"].tolist()
    for code in code_columns:
        run_votes: dict[str, list[int]] = {iid: [] for iid in ids}
        for df in run_dfs:
            indexed = (
                df.set_index("image_id")[code]
                .reindex(ids).fillna(0).astype(int)
            )
            for iid, val in indexed.items():
                run_votes[iid].append(int(val))
        for iid, votes in run_votes.items():
            records.append({
                "image_id":      iid,
                "code":          code,
                "votes":         json.dumps(votes),
                "majority":      int(sum(votes) > len(votes) / 2),
                "all_agree":     len(set(votes)) == 1,
                "agreement_rate": (
                    sum(v == votes[0] for v in votes) / len(votes)
                    if votes else None
                ),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--metadata_csv", type=Path, required=True,
                   help="Path to gallery_metadata.csv")
    p.add_argument("--image_dir",    type=Path, required=True,
                   help="Folder containing image files named {id}.jpg")
    p.add_argument("--codebook_md",  type=Path, required=True)
    p.add_argument("--out_dir",      type=Path, required=True)
    p.add_argument("--runs",         type=int,  default=3)
    p.add_argument("--model",        default="models/gemini-3.1-pro-preview")
    p.add_argument("--temperature",  type=float, default=0.1)
    p.add_argument("--max_retries",  type=int,  default=3)
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--workers",      type=int,  default=4,
                   help="Concurrent API calls per run (default 4). Raise to speed up, lower if rate-limited.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set")

    client = genai.Client(api_key=api_key)

    run_id  = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    codebook_text = read_text(args.codebook_md)
    code_names    = load_code_names(args.codebook_md)
    code_mapping  = build_code_mapping(code_names)
    code_columns  = list(code_mapping.values())
    log(f"Loaded {len(code_names)} codes")

    images_df = load_images(args.metadata_csv, args.image_dir)
    log(f"Images to code: {len(images_df)}")

    prompt_text = build_prompt(codebook_text, len(code_columns))
    run_outputs: list[pd.DataFrame] = []

    def code_one_image(row: pd.Series, run_idx: int) -> dict:
        """Code a single image; returns a result dict with image_id + code columns."""
        image_id   = str(row["image_id"])
        image_path = Path(row["image_path"])
        full_row   = {"image_id": image_id,
                      "prompt_id": str(row.get("prompt_id", image_id)),
                      **{col: 0 for col in code_columns}}
        try:
            img_data, mime_type = encode_image(image_path)
            thread_client = genai.Client(api_key=api_key)
            response = call_model(
                thread_client, args.model, prompt_text,
                img_data, mime_type,
                args.temperature, args.max_retries,
            )
            parsed = parse_response(response, image_id, code_columns)
            for col in code_columns:
                if col in parsed:
                    full_row[col] = parsed[col]
            log(f"  [run {run_idx}] \u2713 {image_id}")
        except Exception as exc:
            log(f"  [run {run_idx}] Error on {image_id}: {exc} \u2014 using zeros")
        return full_row

    for run_idx in range(1, args.runs + 1):
        log(f"\n\u2500\u2500 Run {run_idx}/{args.runs} \u2500\u2500")
        shuffled = images_df.sample(
            frac=1, random_state=args.seed + run_idx
        ).reset_index(drop=True)

        n_workers = min(args.workers, len(shuffled))
        log(f"  Coding {len(shuffled)} images with {n_workers} concurrent workers...")

        run_rows = [None] * len(shuffled)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(code_one_image, shuffled.iloc[i], run_idx): i
                for i in range(len(shuffled))
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    run_rows[idx] = future.result()
                except Exception as exc:
                    image_id = str(shuffled.iloc[idx]["image_id"])
                    log(f"  Unhandled error for {image_id}: {exc}")
                    run_rows[idx] = {"image_id": image_id,
                                     "prompt_id": str(shuffled.iloc[idx].get("prompt_id", image_id)),
                                     **{col: 0 for col in code_columns}}

        run_df = pd.DataFrame(run_rows)
        log(f"  Run {run_idx} complete: {len(run_df)} images coded")
        run_df.to_csv(out_dir / f"run_{run_idx:02d}_coded.csv", index=False)
        run_outputs.append(run_df)


    log("\nComputing majority vote...")
    for i, df in enumerate(run_outputs):
        log(f"  run_outputs[{i}] columns: {list(df.columns)}")
        log(f"  run_outputs[{i}] shape: {df.shape}")
    final_df = compute_majority(images_df, run_outputs, code_columns)
    final_df.to_csv(out_dir / "final_majority_vote.csv", index=False)

    log("Computing agreement log...")
    agreement_df = compute_agreement(images_df, run_outputs, code_columns)
    agreement_df.to_csv(out_dir / "final_majority_vote_agreement.csv", index=False)

    manifest = {
        "timestamp":    dt.datetime.now().isoformat(),
        "model":        args.model,
        "temperature":  args.temperature,
        "runs":         args.runs,
        "n_images":     len(images_df),
        "codes":        code_names,
        "code_mapping": code_mapping,
        "inputs": {
            "metadata_csv": str(args.metadata_csv),
            "image_dir":    str(args.image_dir),
            "codebook_md":  str(args.codebook_md),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
