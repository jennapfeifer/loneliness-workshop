#!/usr/bin/env python3
"""
Qualitative coding of AI-image generation PROMPTS from the workshop.

Input  : generation_log.csv  (one row per generation attempt)
Output : final_majority_vote.csv             – one row per prompt, binary code columns
         final_majority_vote_agreement.csv   – inter-run agreement per code

Adapted from Renchi_Qual_code.py.
Key changes vs. original:
  - load_responses() reads generation_log.csv, filters by consent + status
  - ID column is a stable key built from gallery_id + attempt_index
  - majority-vote helpers are inlined (no external import needed)
  - Everything else (Gemini client, shuffling, chunking, resume) is unchanged

Usage:
    export GOOGLE_API_KEY=...
    python code_prompts.py \\
        --input_csv   path/to/generation_log.csv \\
        --prompt_md   path/to/coding_prompt_prompts.md \\
        --codebook_md path/to/codebook_prompts.md \\
        --out_dir     results/prompts_coded \\
        --runs 5

    # Optional: filter by status (default: submitted + discarded)
    python code_prompts.py ... --status submitted discarded
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import os
import random
import re
import time
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

# Load GOOGLE_API_KEY from .env file in the project folder
load_dotenv()


# ===========================================================================
# Exceptions
# ===========================================================================

class TokenLimitExceededError(Exception):
    """Raised when the model hits the token limit."""


# ===========================================================================
# Inlined majority-vote helpers
# (originally from majority_vote_from_run.py — no separate file needed)
# ===========================================================================

def parse_model_csv(
    response_text: str,
    chunk_df: pd.DataFrame,
    code_names: list[str],
    code_mapping: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    """
    Parse the model's CSV response and align with the original chunk rows.
    Model must return: prompt_id, <snake_code_1>, <snake_code_2>, ...
    Each code cell should be 1 (present) or 0 (absent).
    """
    code_columns = [code_mapping[c] for c in code_names]

    # Strip markdown fences if present
    text = re.sub(r"^```[^\n]*\n?", "", response_text.strip())
    text = re.sub(r"```$", "", text.strip())

    try:
        model_df = pd.read_csv(StringIO(text), dtype=str, on_bad_lines="skip")
    except Exception:
        # Fallback: parse line-by-line, dropping any row whose field count
        # doesn't match (1 prompt_id + n_codes). Handles stray commas in
        # prompt text that the model echoes back unquoted.
        expected_fields = 1 + len(code_columns)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        clean_lines = [lines[0]]  # keep header unconditionally
        skipped = 0
        for line in lines[1:]:
            if len(line.split(",")) == expected_fields:
                clean_lines.append(line)
            else:
                log(
                    f"  Warning: skipping malformed CSV row "
                    f"({len(line.split(','))} fields, expected {expected_fields}): "
                    f"{line[:80]}"
                )
                skipped += 1
        if skipped:
            log(
                f"  {skipped} row(s) skipped due to CSV parse errors — "
                f"will default to 0 for this run (majority vote absorbs it)"
            )
        try:
            model_df = pd.read_csv(StringIO("\n".join(clean_lines)), dtype=str)
        except Exception as exc:
            raise ValueError(
                f"Could not parse model response as CSV: {exc}\n---\n{text[:500]}"
            ) from exc

    model_df.columns = [c.strip().lower() for c in model_df.columns]

    # Normalise the ID column name
    id_col = None
    for candidate in ("prompt_id", "responseid", "id"):
        if candidate in model_df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(
            f"No ID column in model response. Columns: {list(model_df.columns)}"
        )

    model_df[id_col] = model_df[id_col].astype(str).str.strip()

    for col in code_columns:
        if col not in model_df.columns:
            model_df[col] = "0"

    merged = chunk_df[["prompt_id"]].copy()
    merged = merged.merge(
        model_df[[id_col] + code_columns].rename(columns={id_col: "prompt_id"}),
        on="prompt_id",
        how="left",
    )

    for col in code_columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

    stats = {"expected_rows": len(chunk_df), "parsed_rows": len(model_df)}
    return merged, stats


def compute_majority(
    responses_df: pd.DataFrame,
    run_dfs: list[pd.DataFrame],
    code_columns: list[str],
) -> pd.DataFrame:
    """Majority vote across runs: 1 if >50% of runs coded it as 1, else 0."""
    base = responses_df[["prompt_id", "prompt_text"]].copy()

    for col in code_columns:
        votes = []
        for df in run_dfs:
            col_vals = (
                df.set_index("prompt_id")[col]
                .reindex(base["prompt_id"])
                .fillna(0)
                .astype(int)
            )
            votes.append(col_vals.values)

        stacked = np.array(votes).T  # (n_rows, n_runs)
        base[col] = (stacked.sum(axis=1) > len(run_dfs) / 2).astype(int)

    return base


def compute_agreement_log(
    responses_df: pd.DataFrame,
    run_dfs: list[pd.DataFrame],
    code_columns: list[str],
) -> pd.DataFrame:
    """Per row x code: all run votes + agreement rate. Useful for auditing."""
    records = []
    ids = responses_df["prompt_id"].tolist()

    for code in code_columns:
        run_votes: dict[str, list[int]] = {pid: [] for pid in ids}
        for df in run_dfs:
            indexed = (
                df.set_index("prompt_id")[code]
                .reindex(ids)
                .fillna(0)
                .astype(int)
            )
            for pid, val in indexed.items():
                run_votes[pid].append(int(val))

        for pid, votes in run_votes.items():
            records.append({
                "prompt_id":      pid,
                "code":           code,
                "votes":          json.dumps(votes),
                "majority":       int(sum(votes) > len(votes) / 2),
                "all_agree":      len(set(votes)) == 1,
                "agreement_rate": (
                    sum(v == votes[0] for v in votes) / len(votes) if votes else None
                ),
            })

    return pd.DataFrame(records)


# ===========================================================================
# Utilities (unchanged from original)
# ===========================================================================

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log(message: str) -> None:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {message}", flush=True)


def load_code_names(codebook_path: Path) -> list[str]:
    text = read_text(codebook_path)
    codes: list[str] = []
    pattern = re.compile(r"^###\s*(?:Code\s*)?\d+\s*[:\)]\s*(.+?)\s*$")
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            codes.append(m.group(1))
    if not codes:
        raise ValueError(f"No codes found in codebook: {codebook_path}")
    return codes


def to_snake_case(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return s.strip("_")


def build_code_mapping(code_names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in code_names:
        snake = to_snake_case(name)
        if snake in mapping.values():
            raise ValueError(f"Duplicate snake_case code: {snake}")
        mapping[name] = snake
    return mapping


def shuffle_code_names(code_names: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    shuffled = code_names.copy()
    rng.shuffle(shuffled)
    return shuffled


# ===========================================================================
# Data loading  <- adapted for generation_log.csv
# ===========================================================================

def load_responses(
    input_csv: Path,
    status_filter: list[str],
    consent_col: str,
) -> pd.DataFrame:
    """
    Load generation_log.csv and return a DataFrame with columns:
        prompt_id   - stable string key  (gallery_id_att<attempt_index>)
        prompt_text - the raw prompt string

    Rows are filtered to:
      - consent_col is True/1/yes  (or column absent = keep all)
      - status in status_filter
      - non-empty prompt
    """
    def read_csv_robust(path: Path) -> pd.DataFrame:
        """Try common encodings until one works."""
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return pd.read_csv(path, dtype=str, encoding=enc, on_bad_lines="warn")
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode {path} with any common encoding.")

    suffix = Path(input_csv).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(input_csv, dtype=str, engine="openpyxl")
        except Exception:
            try:
                # Some xlsx files need xlrd engine
                df = pd.read_excel(input_csv, dtype=str, engine="xlrd")
            except Exception as e:
                raise ValueError(
                    f"Could not open {input_csv} as an Excel file.\n"
                    f"Please open it in Excel and re-save as:\n"
                    f"  File → Save As → CSV UTF-8 (Comma delimited) (.csv)\n"
                    f"Then point --input_csv at the .csv file instead.\n"
                    f"Original error: {e}"
                )
    elif suffix == ".csv":
        df = read_csv_robust(input_csv)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .xlsx or .csv")
    df.columns = [c.strip().lower() for c in df.columns]

    # consent filter
    if consent_col in df.columns:
        before = len(df)
        df = df[df[consent_col].str.lower().isin(["true", "1", "yes"])]
        log(f"Consent filter ({consent_col}): {before} -> {len(df)} rows")
    else:
        log(f"Warning: consent column '{consent_col}' not found — keeping all rows")

    # status filter
    if "status" in df.columns:
        before = len(df)
        df = df[df["status"].str.lower().isin([s.lower() for s in status_filter])]
        log(f"Status filter {status_filter}: {before} -> {len(df)} rows")

    # prompt column
    if "prompt" not in df.columns:
        raise ValueError(f"No 'prompt' column found. Available: {list(df.columns)}")
    df = df[df["prompt"].notna() & (df["prompt"].str.strip() != "")]

    # stable prompt_id — includes status so submitted/discarded can be compared later
    status_tag = df["status"].str.lower().str.strip() if "status" in df.columns else pd.Series(["unknown"] * len(df), index=df.index)
    if "gallery_id" in df.columns and "attempt_index" in df.columns:
        df["prompt_id"] = (
            status_tag
            + "_"
            + df["gallery_id"].fillna("unknown").str.strip()
            + "_att"
            + df["attempt_index"].fillna("0").str.strip()
        )
    else:
        df["prompt_id"] = status_tag + "_" + pd.RangeIndex(len(df)).astype(str)

    # safeguard against duplicate IDs
    if df["prompt_id"].duplicated().any():
        df = df.copy()
        df["prompt_id"] = (
            df["prompt_id"] + "_row" + pd.RangeIndex(len(df)).astype(str)
        )

    df = df.rename(columns={"prompt": "prompt_text"})
    return df[["prompt_id", "prompt_text"]].copy().reset_index(drop=True)


# ===========================================================================
# CSV serialisation
# ===========================================================================

def df_to_csv_text(df: pd.DataFrame) -> str:
    out = StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["prompt_id", "prompt_text"])
    for _, row in df.iterrows():
        writer.writerow([row["prompt_id"], row["prompt_text"]])
    return out.getvalue()


def build_prompt(base_prompt: str, codebook_text: str, data_csv: str) -> str:
    return (
        base_prompt.strip()
        + "\n\n# Codebook\n"
        + codebook_text.strip()
        + "\n\n# Prompts CSV\n"
        + data_csv.strip()
        + "\n"
    )


# ===========================================================================
# Gemini API call (unchanged from original)
# ===========================================================================

def call_model(
    client: genai.Client,
    model: str,
    prompt: str,
    temperature: float,
    max_retries: int,
    thinking_level: str = "low",
    debug_path: Optional[Path] = None,
    debug_response: bool = False,
) -> tuple[str, int, dict]:
    attempts = 0
    while True:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                ),
            )

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if getattr(candidate, "finish_reason", None) == "MAX_TOKENS":
                    raise TokenLimitExceededError("Model hit token limit")

            if hasattr(response, "text") and response.text is not None:
                usage = _extract_usage(response)
                return response.text, attempts + 1, usage
            raise ValueError("Model response did not include text")

        except TokenLimitExceededError:
            raise
        except Exception as e:
            attempts += 1
            if attempts > max_retries:
                raise
            time.sleep(1 + attempts)


def _extract_usage(response: object) -> dict:
    usage: dict = {}
    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        for attr in ("model_dump", "to_dict"):
            fn = getattr(meta, attr, None)
            if callable(fn):
                usage["usage_metadata"] = fn()
                break
        else:
            usage["usage_metadata"] = {
                k: v for k, v in vars(meta).items() if not k.startswith("_")
            }
    return usage


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qualitative coding of workshop image-generation PROMPTS via Gemini."
    )
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--thinking_level", default="low")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20250100)
    parser.add_argument(
        "--input_csv",
        default="data/generation_log.csv",
        help="Path to generation_log.csv",
    )
    parser.add_argument(
        "--prompt_md",
        default="prompts/coding_prompt_prompts.md",
        help="Task/system prompt for the coder model",
    )
    parser.add_argument(
        "--codebook_md",
        default="prompts/codebook_prompts.md",
        help="Codebook markdown file",
    )
    parser.add_argument("--out_dir", default="results/prompts_coded")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--status",
        nargs="+",
        default=["submitted", "discarded"],
        help="Status values to include (default: submitted discarded)",
    )
    parser.add_argument(
        "--consent_col",
        default="consent_all_yes",
        help="Column name for the consent flag in the CSV",
    )
    parser.add_argument("--debug_response", action="store_true")
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Use first 10 rows only for a quick test",
    )
    return parser.parse_args()


# ===========================================================================
# Main (same structure as original)
# ===========================================================================

def main() -> None:
    args = parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set")

    input_csv   = Path(args.input_csv)
    prompt_md   = Path(args.prompt_md)
    codebook_md = Path(args.codebook_md)
    out_dir     = Path(args.out_dir)

    log("Loading prompt, codebook, and responses")
    base_prompt   = read_text(prompt_md)
    codebook_text = read_text(codebook_md)
    code_names    = load_code_names(codebook_md)
    code_mapping  = build_code_mapping(code_names)
    log(f"Found {len(code_names)} codes: {code_names}")

    responses_df = load_responses(input_csv, args.status, args.consent_col)
    total_rows   = len(responses_df)
    log(f"Loaded {total_rows} prompts from {input_csv.name}")

    if args.test_mode:
        responses_df = responses_df.head(10).copy()
        args.batch_size = 5
        total_rows = len(responses_df)
        log(f"Test mode: using first {total_rows} rows")

    base_seed = int(args.seed)
    run_id    = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log(f"Writing outputs to {run_dir}")

    run_outputs: list[Optional[pd.DataFrame]]      = [None] * args.runs
    partial_results: dict[int, list[pd.DataFrame]] = {i: [] for i in range(1, args.runs + 1)}
    shuffled_code_names_by_run: dict[int, list[str]] = {}
    job_configs = []

    for run_idx in range(args.runs):
        run_real_idx = run_idx + 1
        run_seed     = base_seed + run_idx
        log(f"Preparing Run {run_real_idx}/{args.runs} (seed {run_seed})")

        shuffled            = responses_df.sample(frac=1, random_state=run_seed).reset_index(drop=True)
        shuffled_code_names = shuffle_code_names(code_names, run_seed)
        shuffled_code_names_by_run[run_real_idx] = shuffled_code_names

        shuffled.to_csv(run_dir / f"run_{run_real_idx:02d}_full_shuffled.csv", index=False, na_rep="")
        (run_dir / f"run_{run_real_idx:02d}_code_shuffle_order.json").write_text(
            json.dumps({"original_order": code_names, "shuffled_order": shuffled_code_names}, indent=2),
            encoding="utf-8",
        )

        for i in range(0, len(shuffled), args.batch_size):
            chunk_idx = i // args.batch_size + 1
            chunk_df  = shuffled.iloc[i: i + args.batch_size].copy()
            job_name  = f"run_{run_real_idx:02d}_chunk_{chunk_idx:02d}"
            job_configs.append({
                "job_name":            job_name,
                "run_index":           run_real_idx,
                "chunk_index":         chunk_idx,
                "chunk_df":            chunk_df,
                "shuffled_code_names": shuffled_code_names,
                "response_path":       run_dir / f"{job_name}_response.txt",
                "debug_path":          run_dir / f"{job_name}_debug.json",
                "parsed_path":         run_dir / f"{job_name}_parsed.csv",
                "usage_path":          run_dir / f"{job_name}_usage.json",
            })

    log(f"Submitting {len(job_configs)} batch jobs ({args.runs} runs x chunks)")
    MAX_WORKERS = min(5, len(job_configs))

    def process_chunk_adaptive(
        chunk_df: pd.DataFrame,
        shuffled_code_names: list[str],
        job_name: str,
        response_path: Path,
        debug_path: Path,
        parsed_path: Path,
        usage_path: Path,
        attempt: int = 0,
        temperature_boost: float = 0.0,
    ) -> pd.DataFrame:
        chunk_csv = df_to_csv_text(chunk_df)
        prompt    = build_prompt(base_prompt, codebook_text, chunk_csv)
        temp      = args.temperature + temperature_boost
        client    = genai.Client(api_key=api_key)
        suffix    = f"_attempt_{attempt}" if attempt > 0 else ""

        try:
            response_text, _, usage = call_model(
                client, args.model, prompt, temp, args.max_retries,
                thinking_level=args.thinking_level,
                debug_path=debug_path,
                debug_response=args.debug_response,
            )
            response_path.with_name(
                f"{response_path.stem}{suffix}{response_path.suffix}"
            ).write_text(response_text, encoding="utf-8")
            usage_path.with_name(
                f"{usage_path.stem}{suffix}{usage_path.suffix}"
            ).write_text(json.dumps(usage, indent=2), encoding="utf-8")

            parsed_df, _ = parse_model_csv(response_text, chunk_df, code_names, code_mapping)
            return parsed_df

        except TokenLimitExceededError:
            log(f"Token limit for {job_name} — splitting in half (attempt {attempt + 1})")
            mid = len(chunk_df) // 2
            if mid == 0:
                raise
            df1 = process_chunk_adaptive(
                chunk_df.iloc[:mid].copy(), shuffled_code_names,
                f"{job_name}_part1", response_path, debug_path, parsed_path, usage_path,
                attempt + 1, temperature_boost + 0.1,
            )
            df2 = process_chunk_adaptive(
                chunk_df.iloc[mid:].copy(), shuffled_code_names,
                f"{job_name}_part2", response_path, debug_path, parsed_path, usage_path,
                attempt + 1, temperature_boost + 0.1,
            )
            return pd.concat([df1, df2], ignore_index=True)

    def process_job(config: dict) -> tuple[int, int, pd.DataFrame]:
        log(f"Processing {config['job_name']}...")
        parsed_df = process_chunk_adaptive(
            config["chunk_df"], config["shuffled_code_names"], config["job_name"],
            config["response_path"], config["debug_path"],
            config["parsed_path"],  config["usage_path"],
        )
        parsed_df.to_csv(config["parsed_path"], index=False, na_rep="")
        return config["run_index"], config["chunk_index"], parsed_df

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_job, cfg): cfg for cfg in job_configs}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(job_configs),
            desc="Processing batches",
        ):
            cfg = futures[future]
            try:
                run_idx, _, parsed_df = future.result()
                partial_results[run_idx].append(parsed_df)
                log(f"Completed {cfg['job_name']}")
            except Exception as exc:
                log(f"Error in {cfg['job_name']}: {exc}")
                raise

    log("Stitching chunks per run...")
    for run_idx in range(1, args.runs + 1):
        chunks = partial_results[run_idx]
        if not chunks:
            log(f"Warning: Run {run_idx} has no completed chunks")
            continue
        full = pd.concat(chunks, ignore_index=True)
        if len(full) != total_rows:
            log(f"Warning: Run {run_idx} has {len(full)} rows, expected {total_rows}")
        run_outputs[run_idx - 1] = full

    valid_outputs = [df for df in run_outputs if df is not None]
    if not valid_outputs:
        raise RuntimeError("No runs completed successfully")

    code_columns = [code_mapping[c] for c in code_names]

    log("Computing majority vote...")
    final_df = compute_majority(responses_df, valid_outputs, code_columns)
    (run_dir / "final_majority_vote.csv").write_bytes(
        final_df.to_csv(index=False, na_rep="").encode()
    )

    log("Computing agreement log...")
    agreement_df = compute_agreement_log(responses_df, valid_outputs, code_columns)
    (run_dir / "final_majority_vote_agreement.csv").write_bytes(
        agreement_df.to_csv(index=False, na_rep="").encode()
    )

    manifest = {
        "timestamp":    dt.datetime.now().isoformat(),
        "model":        args.model,
        "model_params": {"temperature": args.temperature, "thinking_level": args.thinking_level},
        "batch_size":   args.batch_size,
        "status_filter": args.status,
        "consent_col":  args.consent_col,
        "inputs":       {"input_csv": str(input_csv), "prompt_md": str(prompt_md),
                         "codebook_md": str(codebook_md)},
        "input_hashes": {"input_csv": sha256_file(input_csv), "prompt_md": sha256_file(prompt_md),
                         "codebook_md": sha256_file(codebook_md)},
        "code_mapping": code_mapping,
        "code_shuffling": {
            "enabled":   True,
            "base_seed": base_seed,
            "shuffled_orders": {
                f"run_{i}": shuffled_code_names_by_run[i]
                for i in shuffled_code_names_by_run
            },
        },
        "test_mode": bool(args.test_mode),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("Pipeline complete.")


if __name__ == "__main__":
    main()
