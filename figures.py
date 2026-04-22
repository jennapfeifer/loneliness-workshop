#!/usr/bin/env python3
"""
figures.py  —  Analysis and figures for the loneliness workshop paper.


Produces:
  Generation log:
    descriptives.md / descriptives_tidy.csv  —  generation-log statistics

  Main figures:
    fig2_comparison_by_gap.png   —  Figure 2: prompt vs image rate (sorted by gap)
    fig3_image_gallery.png       —  Figure 3: curated image gallery with prompts

  Supplementary figures:
    sup1_likert_grid.png         —  Supplementary Figure 1: Likert item grid
    sup2_prompt_frequency.png    —  Supplementary Figure 2: prompt code prevalence (all prompts)
    sup3_image_frequency.png     —  Supplementary Figure 3: image code prevalence (by section)

All arguments are optional — only outputs whose inputs are available are produced.

Usage:
    python figures.py \\
        --log_csv      Data/generation_log.csv \\
        --prompt_csv   TextAnalysis/results/prompts_coded/TIMESTAMP/final_majority_vote.csv \\
        --image_csv    ImageAnalysis/results/images_coded/TIMESTAMP/final_majority_vote.csv \\
        --image_dir    Data/gallery_images \\
        --survey_xlsx  Data/qualtrics_export.xlsx \\
        --out_dir      figures_pub
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")


# ── Shared style ──────────────────────────────────────────────────────────────

COL_SUBMIT  = "#2C3E50"   # dark navy  — submitted / prompt direct
COL_DISCARD = "#95A5A6"   # muted grey — discarded / image
COL_TRANS   = "#5D8AA8"   # steel blue — translational mapping

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
})


def save(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"  → {path}")
    plt.close(fig)

# ── Likert survey constants ───────────────────────────────────────────────────

LIKERT_PALETTE = {
    "Not at all":   "#d1e5f0",
    "A little":     "#92c5de",
    "Moderately":   "#4393c3",
    "A lot":        "#2166ac",
    "A great deal": "#053061",
}
LIKERT_LEVELS = list(LIKERT_PALETTE.keys())

ITEM_PATTERNS = {
    "easy_generate":        "easy to generate an image",
    "ai_matched":           "translated my prompt into an image that matched",
    "share_ideas":          "workshop format helped me share my ideas",
    "loneliness_diversity": "loneliness can look different across people",
    "bias_understanding":   "limits and biases of using generative ai",
    "engaged":              "how engaged did you feel",
    "safe":                 "how safe did you feel",
    "limitations":          "main limitations or problems you encountered",
    "other_feedback":       "any other feedback on the workshop",
    "consent":              "may we use your anonymous answers",
}

SHORT_LABELS = {
    "easy_generate":        "Easy to generate image",
    "ai_matched":           "AI matched my prompt",
    "share_ideas":          "Format helped share ideas",
    "loneliness_diversity": "Loneliness looks different\nacross people",
    "bias_understanding":   "Understood AI limits\n& biases",
    "engaged":              "Felt engaged",
    "safe":                 "Felt safe",
}

LIKERT_KEYS = list(SHORT_LABELS.keys())


def map_likert(val: str) -> str | None:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    mapping = {
        "1": "Not at all",    "not at all":    "Not at all",
        "2": "A little",      "a little":      "A little",
        "3": "Moderately",    "moderately":    "Moderately",
        "4": "A lot",         "a lot":         "A lot",
        "5": "A great deal",  "a great deal":  "A great deal",
        "prefer not to say": None,
    }
    return mapping.get(s, None)


def load_survey(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load Qualtrics xlsx/csv with two-row header. Returns (consented_df, col_map)."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError("openpyxl is required for .xlsx files: pip install openpyxl")
        raw = pd.read_excel(path, header=None, dtype=str, engine="openpyxl")
    else:
        raw = pd.read_csv(path, header=None, dtype=str)

    q_texts = raw.iloc[1].fillna("").astype(str).tolist()
    data    = raw.iloc[2:].reset_index(drop=True)
    data.columns = [f"col_{i}" for i in range(len(q_texts))]
    q_lookup = {f"col_{i}": q_texts[i].strip().lower() for i in range(len(q_texts))}

    def find_col(pattern: str) -> str | None:
        for col_id, text in q_lookup.items():
            if pattern in text:
                return col_id
        return None

    col_map = {key: find_col(pat) for key, pat in ITEM_PATTERNS.items()}
    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        print(f"  Warning: survey columns not found for: {missing}")

    consent_col = col_map.get("consent")
    if consent_col:
        raw_consent = data[consent_col].str.strip().str.lower()
        mask = raw_consent.isin(["yes", "y", "true", "1"]) |                raw_consent.str.contains("yes", na=False)
        data = data[mask].reset_index(drop=True)

    print(f"  N consenting survey respondents: {len(data)}")
    return data, col_map



# ── Image-codebook section metadata ──────────────────────────────────────────

CODE_META = [
    ("indoor_domestic_setting",                                    "Indoor domestic setting",               "A. Setting & environment"),
    ("public_or_institutional_setting",                            "Public / institutional setting",        "A. Setting & environment"),
    ("outdoor_setting",                                            "Outdoor setting",                       "A. Setting & environment"),
    ("darkness_or_dim_lighting",                                   "Darkness / dim lighting",               "A. Setting & environment"),
    ("bright_or_neutral_lighting",                                 "Bright / neutral lighting",             "A. Setting & environment"),
    ("subject_is_physically_alone",                                "Subject physically alone",              "B. Social configuration"),
    ("subject_is_in_a_group_or_crowd",                             "Subject in group or crowd",             "B. Social configuration"),
    ("others_are_visibly_engaged_with_each_other",                 "Others engaged with each other",        "B. Social configuration"),
    ("subject_is_visibly_excluded_or_peripheral",                  "Subject excluded / peripheral",         "B. Social configuration"),
    ("sad_or_distressed_facial_expression",                        "Sad / distressed expression",           "C. Visual cues"),
    ("withdrawn_body_language",                                    "Withdrawn body language",               "C. Visual cues"),
    ("downward_or_averted_gaze",                                   "Downward / averted gaze",               "C. Visual cues"),
    ("technology_or_screen_use",                                   "Technology / screen use",               "C. Visual cues"),
    ("loneliness_conveyed_through_physical_isolation",             "Loneliness via physical isolation",     "C. Visual cues"),
    ("loneliness_conveyed_through_social_disconnection",           "Loneliness via social disconnection",   "C. Visual cues"),
    ("rain_or_grey_weather",                                       "Rain / grey weather",                   "D. AI default tropes"),
    ("stereotyped_figure_elderly_person",                          "Stereotyped figure (elderly)",          "D. AI default tropes"),
    ("stereotyped_setting_park_bench",                             "Stereotyped setting (park bench)",      "D. AI default tropes"),
    ("exaggerated_or_theatrical_sadness",                          "Exaggerated / theatrical sadness",      "D. AI default tropes"),
    ("racially_or_demographically_homogeneous_crowd",              "Homogeneous crowd",                     "D. AI default tropes"),
    ("image_depicts_a_plausible_everyday_scene",                   "Plausible everyday scene",              "E. Image–prompt adequacy"),
    ("image_appears_to_match_the_social_scenario_in_the_prompt",   "Matches prompt scenario",               "E. Image–prompt adequacy"),
    ("image_reproduces_a_stereotyped_representation_of_loneliness","Stereotyped representation",            "E. Image–prompt adequacy"),
]

SECTION_ORDER = [
    "A. Setting & environment",
    "B. Social configuration",
    "C. Visual cues",
    "D. AI default tropes",
    "E. Image–prompt adequacy",
]

META_DF = pd.DataFrame(CODE_META, columns=["code", "label", "section"])

# ── Codes shared between prompt and image codebooks ───────────────────────────
#
# "direct"       — identical concept in both codebooks; rates are comparable.
# "translational"— prompt names an intent, image shows its expected visual
#                  expression; gaps are indicative rather than exact.

SHARED_MAPPING = {
    # direct mappings
    "physical_isolation":                "subject_is_physically_alone",
    "crowded_but_disconnected":          "subject_is_in_a_group_or_crowd",
    "darkness_or_dim_lighting":          "darkness_or_dim_lighting",
    "technology_or_screen_use":          "technology_or_screen_use",
    "indoor_domestic_setting":           "indoor_domestic_setting",
    "public_or_institutional_setting":   "public_or_institutional_setting",
    "rain_or_grey_weather":              "rain_or_grey_weather",
    "stereotyped_figure_elderly_person": "stereotyped_figure_elderly_person",
    "stereotyped_setting_park_bench":    "stereotyped_setting_park_bench",
    # translational mappings
    "body_language_or_posture_cues":     "withdrawn_body_language",
    "gaze_direction":                    "downward_or_averted_gaze",
    "nature_or_outdoor_setting":         "outdoor_setting",
    "observing_others_connecting":       "others_are_visibly_engaged_with_each_other",
    "explicit_emotional_labelling":      "sad_or_distressed_facial_expression",
    "facial_expression_specified":       "sad_or_distressed_facial_expression",
}

MAPPING_TYPE = {
    "physical_isolation":                "direct",
    "crowded_but_disconnected":          "direct",
    "darkness_or_dim_lighting":          "direct",
    "technology_or_screen_use":          "direct",
    "indoor_domestic_setting":           "direct",
    "public_or_institutional_setting":   "direct",
    "rain_or_grey_weather":              "direct",
    "stereotyped_figure_elderly_person": "direct",
    "stereotyped_setting_park_bench":    "direct",
    "body_language_or_posture_cues":     "translational",
    "gaze_direction":                    "translational",
    "nature_or_outdoor_setting":         "translational",
    "observing_others_connecting":       "translational",
    "explicit_emotional_labelling":      "translational",
    "facial_expression_specified":       "translational",
}


# ── Shared helpers ────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def binary_cols(df: pd.DataFrame) -> list[str]:
    """Return columns that contain only 0/1 values (i.e. code columns)."""
    skip = {"prompt_id", "image_id", "response_id", "prompt_text",
            "prompt", "response_text", "status"}
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
        if set(vals).issubset({0.0, 1.0}):
            cols.append(c)
    return cols


def to_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    return out


def pretty(snake: str) -> str:
    return snake.replace("_", " ").capitalize()


# ══════════════════════════════════════════════════════════════════════════════
# Generation-log descriptives
# ══════════════════════════════════════════════════════════════════════════════

def load_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for col in ["latency_ms", "decision_time_ms", "total_tokens",
                "prompt_tokens", "candidates_tokens", "thoughts_tokens",
                "total_time_ms", "queue_wait_ms", "prompt_words", "image_bytes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "latency_ms" in df.columns:
        df["latency_s"] = df["latency_ms"] / 1000
    if "decision_time_ms" in df.columns:
        df["decision_s"] = df["decision_time_ms"] / 1000
    if "status" in df.columns:
        df["status"] = df["status"].str.strip().str.lower()
    if "consent_all_yes" in df.columns:
        consented = df["consent_all_yes"].str.lower().isin(["true", "1", "yes"])
        df = df[consented | df["consent_all_yes"].isna()]

    return df


def _stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan,
                    q1=np.nan, q3=np.nan, min=np.nan, max=np.nan)
    return dict(
        n=len(s),
        mean=float(s.mean()),
        sd=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        median=float(s.median()),
        q1=float(s.quantile(0.25)),
        q3=float(s.quantile(0.75)),
        min=float(s.min()),
        max=float(s.max()),
    )


def generation_descriptives(log: pd.DataFrame, out_dir: Path) -> None:
    """
    Compute and save descriptive statistics from the generation log.

    Reports attempt counts (overall + by status) and key numeric metrics
    (latency, decision time, token use) for all attempts, submitted only,
    and discarded only.  Results are written as both a tidy CSV and a
    markdown table ready to paste into a paper.
    """

    METRICS = [
        ("latency_s",         "API latency",     "s"),
        ("decision_s",        "Decision time",   "s"),
        ("total_tokens",      "Total tokens",    "tokens"),
        ("prompt_tokens",     "Input tokens",    "tokens"),
        ("candidates_tokens", "Output tokens",   "tokens"),
        ("thoughts_tokens",   "Thinking tokens", "tokens"),
        ("prompt_words",      "Prompt length",   "words"),
    ]

    groups = {"all": log}
    if "status" in log.columns:
        groups["submitted"] = log[log["status"] == "submitted"]
        groups["discarded"] = log[log["status"] == "discarded"]

    # ── Print counts ──────────────────────────────────────────────────────────
    total = len(log)
    print(f"\n  Total (consented) attempts: {total:,}")
    if "status" in log.columns:
        for status in ["submitted", "discarded", "generated", "error"]:
            n = int((log["status"] == status).sum())
            pct = n / total * 100 if total else 0
            print(f"  {status.capitalize():<12}: {n:>4}  ({pct:.1f}%)")

    # ── Build tidy stats rows ─────────────────────────────────────────────────
    tidy_rows = []
    for col, label, unit in METRICS:
        for grp_name, grp_df in groups.items():
            if col not in grp_df.columns:
                continue
            row = {"metric": label, "unit": unit, "group": grp_name,
                   **_stats(grp_df[col])}
            tidy_rows.append(row)

    if not tidy_rows:
        print("  No numeric columns found — check column names in log CSV.")
        return

    tidy_df = pd.DataFrame(tidy_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    tidy_df.to_csv(out_dir / "descriptives_tidy.csv", index=False)

    # ── Markdown summary ──────────────────────────────────────────────────────
    def fmt(val: float, digits: int = 2) -> str:
        return "—" if pd.isna(val) else f"{val:,.{digits}f}"

    def mean_sd(r: pd.Series) -> str:
        if r["n"] == 0:
            return "—"
        return f"{fmt(r['mean'])} ± {fmt(r['sd'])}"

    def med_iqr(r: pd.Series) -> str:
        if r["n"] == 0:
            return "—"
        return f"{fmt(r['median'])}  [{fmt(r['q1'])}, {fmt(r['q3'])}]"

    # Build pivot rows: one row per metric, columns = group stats
    pivot_rows = []
    for col, label, unit in METRICS:
        metric_key = f"{label} ({unit})"
        row = {"Metric": metric_key}
        for grp in ["all", "submitted", "discarded"]:
            sub = tidy_df[(tidy_df["metric"] == label) & (tidy_df["group"] == grp)]
            if sub.empty or sub.iloc[0]["n"] == 0:
                row[f"{grp}_n"]      = "—"
                row[f"{grp}_mean_sd"] = "—"
                row[f"{grp}_med_iqr"] = "—"
            else:
                r = sub.iloc[0]
                row[f"{grp}_n"]       = int(r["n"])
                row[f"{grp}_mean_sd"] = mean_sd(r)
                row[f"{grp}_med_iqr"] = med_iqr(r)
        pivot_rows.append(row)

    # Console print
    print(f"\n  {'Metric':<30} {'All':>25} {'Submitted':>25} {'Discarded':>25}")
    print("  " + "─" * 82)
    for r in pivot_rows:
        print(
            f"  {r['Metric']:<30}"
            f"  {r['all_mean_sd']:>23}"
            f"  {r['submitted_mean_sd']:>23}"
            f"  {r['discarded_mean_sd']:>23}"
        )

    # Markdown file
    md_lines: list[str] = [
        "# Generation-log descriptives\n",
        "## Attempt counts\n",
    ]
    if "status" in log.columns:
        for status in ["submitted", "discarded", "generated", "error"]:
            n = int((log["status"] == status).sum())
            pct = n / total * 100 if total else 0
            md_lines.append(f"- **{status.capitalize()}**: {n:,} ({pct:.1f}%)")
    md_lines.append(f"\nTotal (consented): {total:,}\n")

    md_lines.append("\n## Key metrics  (mean ± SD  |  median [IQR])\n")
    col_heads = ("All", "Submitted", "Discarded")
    md_lines.append(
        "| Metric | N | Mean ± SD | Median [IQR] "
        "| N | Mean ± SD | Median [IQR] "
        "| N | Mean ± SD | Median [IQR] |"
    )
    md_lines.append(
        "|--------|---|-----------|------------- "
        "|---|-----------|------------- "
        "|---|-----------|-------------|"
    )
    for r in pivot_rows:
        md_lines.append(
            f"| {r['Metric']} "
            f"| {r['all_n']} | {r['all_mean_sd']} | {r['all_med_iqr']} "
            f"| {r['submitted_n']} | {r['submitted_mean_sd']} | {r['submitted_med_iqr']} "
            f"| {r['discarded_n']} | {r['discarded_mean_sd']} | {r['discarded_med_iqr']} |"
        )

    md_path = out_dir / "descriptives.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\n  → {md_path}")
    print(f"  → {out_dir / 'descriptives_tidy.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# Supplementary figure 2 — Prompt code frequency
# ══════════════════════════════════════════════════════════════════════════════

def fig_prompt_frequency(prompt_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Supplementary Figure 1: single horizontal bar chart of prompt code prevalence
    across all prompts (submitted and discarded combined), sorted ascending.
    """
    cols      = binary_cols(prompt_df)
    prompt_df = to_int(prompt_df, cols)
    n         = len(prompt_df)

    freqs = prompt_df[cols].mean() * 100
    freqs = freqs[freqs > 0].sort_values(ascending=True)

    if freqs.empty:
        print("  Skipping sup2: all prompt code frequencies are 0%.")
        return

    labels = [pretty(c) for c in freqs.index]
    values = freqs.values

    fig_h = max(4, len(freqs) * 0.35)
    fig, ax = plt.subplots(figsize=(7, fig_h))

    bars = ax.barh(labels, values, color=COL_SUBMIT, height=0.6)
    for bar, val in zip(bars, values):
        count = int(round(val * n / 100))
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}% (n={count})", va="center", ha="left",
                fontsize=8.5, color="#333333")

    ax.set_xlim(0, min(105, values.max() + 18))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel(f"% of all prompts  (N = {n})")
    fig.tight_layout()
    save(fig, out_dir / "sup2_prompt_frequency.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Image code frequency (faceted by codebook section)
# ══════════════════════════════════════════════════════════════════════════════

def fig_image_frequency(image_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Faceted horizontal bar chart of image code prevalence, one panel per
    codebook section (A–E) as defined in CODE_META.

    Any code not listed in CODE_META is plotted in a flat fallback chart.
    """
    cols     = binary_cols(image_df)
    image_df = to_int(image_df, cols)
    n        = len(image_df)

    freq = image_df[cols].mean().reset_index()
    freq.columns = ["code", "prop"]
    freq = freq.merge(META_DF, on="code", how="left")
    freq = freq[freq["prop"] > 0]  # hide zero-prevalence codes

    sections = [s for s in SECTION_ORDER if s in freq["section"].values]

    if not sections:
        # Flat fallback: code names not matched to CODE_META
        freq_s = freq.sort_values("prop")
        fig_h  = max(4, len(freq_s) * 0.35)
        fig, ax = plt.subplots(figsize=(7, fig_h))
        bars = ax.barh(
            freq_s["code"].apply(pretty), freq_s["prop"] * 100,
            color=COL_DISCARD, height=0.6,
        )
        for bar, val in zip(bars, freq_s["prop"] * 100):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", va="center", ha="left", fontsize=8.5)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_xlabel("% of all images")
        fig.tight_layout()
        save(fig, out_dir / "sup3_image_frequency.png")
        return

    n_rows = (len(sections) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, max(6, len(sections) * 2.4)))
    axes = axes.flatten()

    for ax_idx, section in enumerate(sections):
        ax  = axes[ax_idx]
        sub = freq[freq["section"] == section].sort_values("prop")
        if sub.empty:
            ax.set_visible(False)
            continue

        labels = sub["label"].fillna(sub["code"].apply(pretty))
        bars   = ax.barh(labels, sub["prop"] * 100, color=COL_DISCARD, height=0.65)
        for bar, val in zip(bars, sub["prop"] * 100):
            ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", va="center", ha="left", fontsize=8.5,
                    color="#444444")

        ax.set_xlim(0, min(110, sub["prop"].max() * 100 + 18))
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_xlabel("% of all images", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)

    for i in range(len(sections), len(axes)):
        axes[i].set_visible(False)


    fig.tight_layout()
    save(fig, out_dir / "sup3_image_frequency.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Prompt vs image rate comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_comparison(
    prompt_df: pd.DataFrame,
    image_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Side-by-side horizontal bars showing, for each shared concept, the rate at
    which it appeared in submitted prompts (intent) vs generated images (output).

    Bar colour encodes mapping type:
      Dark navy  [D] = direct mapping — same concept in both codebooks.
      Steel blue [T] = translational mapping — prompt states intent, image
                       shows expected visual expression.  Gaps are indicative.
    Image bars are always muted grey.
    """
    p_cols    = binary_cols(prompt_df)
    i_cols    = binary_cols(image_df)
    prompt_df = to_int(prompt_df, p_cols)
    image_df  = to_int(image_df,  i_cols)

    # Comparison uses submitted prompts only
    if "prompt_id" in prompt_df.columns:
        submitted = prompt_df[
            prompt_df["prompt_id"].str.startswith("submitted", na=False)
        ]
        if submitted.empty:
            submitted = prompt_df  # fall back if no prefix
    else:
        submitted = prompt_df

    n_p = len(submitted)
    n_i = len(image_df)

    rows = []
    for p_col, i_col in SHARED_MAPPING.items():
        if p_col not in submitted.columns or i_col not in image_df.columns:
            continue
        rows.append({
            "label":        pretty(p_col),
            "prompt_rate":  submitted[p_col].mean() * 100,
            "image_rate":   image_df[i_col].mean()  * 100,
            "mapping_type": MAPPING_TYPE.get(p_col, "direct"),
        })

    if not rows:
        print("  Skipping fig3: no shared code columns found in both CSVs.")
        return

    df = pd.DataFrame(rows)
    df["gap"] = df["prompt_rate"] - df["image_rate"]
    df = df.sort_values("prompt_rate", ascending=True)

    def _draw(sorted_df: pd.DataFrame, path: Path,
              divider_after: int | None = None) -> None:
        """Shared drawing logic for both variants."""
        x     = np.arange(len(sorted_df))
        width = 0.38
        fig_h = max(5, len(sorted_df) * 0.52)
        fig, ax = plt.subplots(figsize=(9, fig_h))

        ax.barh(x - width / 2, sorted_df["prompt_rate"].values, width,
                color=COL_SUBMIT,  label="In prompt")
        ax.barh(x + width / 2, sorted_df["image_rate"].values,  width,
                color=COL_DISCARD, label="In image")



        ax.set_yticks(x)
        ax.set_yticklabels(sorted_df["label"].values, fontsize=9)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_xlabel("% of items")
        ax.invert_yaxis()
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        fig.tight_layout()
        save(fig, path)

    # ── Fig 3a: sorted by gap, largest to smallest ────────────────────────────
    df_a = df.sort_values("gap", ascending=False)
    _draw(df_a, out_dir / "fig2_comparison_by_gap.png")

    # ── Fig 3b: grouped by direction, sorted by magnitude within each group ──
    over  = df[df["gap"] <= 0].sort_values("gap", ascending=True)   # image ≥ prompt
    under = df[df["gap"] >  0].sort_values("gap", ascending=False)  # prompt > image
    df_b  = pd.concat([over, under], ignore_index=True)
    _draw(df_b, out_dir / "fig2_comparison_grouped.png",
          divider_after=len(over))

    # Also print a quick summary to the console
    print(f"  Mean gap (prompt − image): {df['gap'].mean():.1f} pp")
    top = df.nlargest(5, "gap")[["label", "prompt_rate", "image_rate", "gap"]]
    print("  Largest gaps:")
    print(top.to_string(index=False))



# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Image gallery with prompts
# ══════════════════════════════════════════════════════════════════════════════

def fig_image_gallery(image_dir: Path, out_dir: Path) -> None:
    """
    Display a curated set of generated images with their prompts as captions.
    """
    import textwrap

    items = [
        ("0011.png", "\u2026the person is like stuck in place still alone in like the original person thinking of being lonely."),
        ("0013.png", "\u2026they are really sad they can't go out like everyone else, there need to be a lot of people outside leaving the library laughing having fun."),
        ("0014.png", "\u2026alone in a group of people\u2026"),
        ("0015.png", "A lonely sober person in a room full of drunk people."),
    ]

    n     = len(items)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 6.5))
    axes = axes.flatten()

    for ax, (filename, prompt) in zip(axes, items):
        img_path = image_dir / filename
        if not img_path.exists():
            ax.text(0.5, 0.5, f"Image not found:\n{filename}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="grey")
            ax.set_axis_off()
            continue

        img = plt.imread(str(img_path))
        ax.imshow(img)
        ax.set_axis_off()

        wrapped = "\n".join(textwrap.wrap(f'"{prompt}"', width=50))
        ax.text(0.5, -0.02, wrapped,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=8.5, color="#333333",
                style="italic",
                wrap=False)

    for ax in axes[n:]:
        ax.set_axis_off()

    fig.tight_layout(h_pad=3.5, w_pad=1.5)
    save(fig, out_dir / "fig3_image_gallery.png")


# ══════════════════════════════════════════════════════════════════════════════
# Supplementary Figure 3 — Likert survey results
# ══════════════════════════════════════════════════════════════════════════════

_QUESTION_ORDER = [
    "Felt safe",
    "Felt engaged",
    "Understood AI limits\n& biases",
    "Loneliness looks different\nacross people",
    "Format helped share ideas",
    "AI matched my prompt",
    "Easy to generate image",
]


def _build_likert_summary(survey: pd.DataFrame, col_map: dict) -> pd.DataFrame | None:
    rows = []
    for key in LIKERT_KEYS:
        col = col_map.get(key)
        if col is None or col not in survey.columns:
            continue
        for val in survey[col]:
            mapped = map_likert(val)
            if mapped:
                rows.append({"key": key, "response": mapped})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    summary = df.groupby(["key", "response"]).size().reset_index(name="n")
    summary["response"] = pd.Categorical(summary["response"],
                                          categories=LIKERT_LEVELS, ordered=True)
    totals = summary.groupby("key")["n"].sum().rename("total")
    summary = summary.join(totals, on="key")
    summary["prop"] = summary["n"] / summary["total"]
    summary["label"] = summary["key"].map(SHORT_LABELS)
    return summary


def fig_likert_stacked(survey: pd.DataFrame, col_map: dict, out_dir: Path) -> None:
    """Supplementary Figure 2a: all Likert items as a single stacked bar chart."""
    summary = _build_likert_summary(survey, col_map)
    if summary is None:
        print("  Skipping sup1a (likert stacked): no Likert data found")
        return

    summary["label"] = pd.Categorical(
        summary["label"], categories=_QUESTION_ORDER, ordered=True
    )
    summary = summary.dropna(subset=["label"]).sort_values(["label", "response"])
    labels_present = [l for l in _QUESTION_ORDER if l in summary["label"].values]

    fig, ax = plt.subplots(figsize=(9, 5))
    for q_label in labels_present:
        q_data = summary[summary["label"] == q_label].sort_values("response")
        left = 0
        for _, row in q_data.iterrows():
            colour = LIKERT_PALETTE.get(str(row["response"]), "#CCCCCC")
            ax.barh(q_label, row["prop"], left=left,
                    color=colour, height=0.65, edgecolor="white", linewidth=0.4)
            if row["prop"] >= 0.07:
                ax.text(left + row["prop"] / 2,
                        labels_present.index(q_label),
                        f"{row['prop']:.0%}",
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            left += row["prop"]

    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Proportion of responses")
    ax.set_yticks(range(len(labels_present)))
    ax.set_yticklabels(labels_present)
    ax.legend(
        handles=[Patch(facecolor=LIKERT_PALETTE[l], label=l) for l in LIKERT_LEVELS],
        loc="lower right", ncol=5, frameon=False, fontsize=8,
        bbox_to_anchor=(1.0, -0.18),
    )
    save(fig, out_dir / "sup1a_likert_stacked.png")


def fig_likert_grid(survey: pd.DataFrame, col_map: dict, out_dir: Path) -> None:
    """Supplementary Figure 2b: one panel per Likert item."""
    summary = _build_likert_summary(survey, col_map)
    if summary is None:
        print("  Skipping sup1 (likert grid): no Likert data found")
        return

    items   = [k for k in LIKERT_KEYS if col_map.get(k) is not None
               and col_map[k] in survey.columns]
    n_items = len(items)
    n_cols  = 2
    n_rows  = (n_items + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(11, n_rows * 2.6), sharex=True)
    axes_flat = axes.flatten() if n_items > 1 else [axes]

    for ax_idx, key in enumerate(items):
        ax  = axes_flat[ax_idx]
        sub = summary[summary["key"] == key].sort_values("response", ascending=False)
        n   = int(sub["total"].iloc[0]) if not sub.empty else 0
        label = SHORT_LABELS.get(key, key).replace("\n", " ")

        bars = ax.barh(
            sub["response"].astype(str),
            sub["prop"] * 100,
            color=[LIKERT_PALETTE.get(str(r), "#CCCCCC") for r in sub["response"]],
            height=0.65, edgecolor="white", linewidth=0.3,
        )
        for bar, (_, row) in zip(bars, sub.iterrows()):
            pct = row["prop"] * 100
            if pct >= 5:
                ax.text(pct + 1, bar.get_y() + bar.get_height() / 2,
                        f"{pct:.0f}%", va="center", ha="left",
                        fontsize=8, color="#333333")

        ax.set_xlim(0, 115)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title(f"{label}  (n={n})", fontsize=9.5, fontweight="bold", pad=4)
        ax.set_xlabel("% of responses", fontsize=8)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for i in range(n_items, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.tight_layout()
    save(fig, out_dir / "sup1_likert_grid.png")

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--log_csv",    type=Path, default=None,
                   help="Path to generation_log.csv")
    p.add_argument("--prompt_csv", type=Path, default=None,
                   help="final_majority_vote.csv from code_prompts.py")
    p.add_argument("--image_csv",  type=Path, default=None,
                   help="final_majority_vote.csv from code_images.py")
    p.add_argument("--survey_xlsx", type=Path, default=None,
                   help="Qualtrics export (.xlsx or .csv) for Supplementary Figure 2")
    p.add_argument("--image_dir",  type=Path, default=None,
                   help="Folder containing gallery images (for fig4)")
    p.add_argument("--out_dir",    type=Path, default=Path("figures_pub"),
                   help="Output directory (default: figures_pub/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prompt_df = image_df = None

    # ── Descriptives ──────────────────────────────────────────────────────────
    if args.log_csv:
        if args.log_csv.exists():
            print("\n── Generation descriptives ──")
            log = load_log(args.log_csv)
            print(f"  Loaded {len(log):,} consented attempts")
            generation_descriptives(log, args.out_dir)
        else:
            print(f"  Warning: {args.log_csv} not found — skipping descriptives")

    # ── Fig 1: Prompt code frequency ─────────────────────────────────────────
    if args.prompt_csv:
        if args.prompt_csv.exists():
            print("\n── Fig 1: Prompt code frequency ──")
            prompt_df = load_csv(args.prompt_csv)
            fig_prompt_frequency(prompt_df, args.out_dir)
        else:
            print(f"  Warning: {args.prompt_csv} not found — skipping Fig 1")

    # ── Fig 2: Image code frequency ───────────────────────────────────────────
    if args.image_csv:
        if args.image_csv.exists():
            print("\n── Fig 2: Image code frequency ──")
            image_df = load_csv(args.image_csv)
            fig_image_frequency(image_df, args.out_dir)
        else:
            print(f"  Warning: {args.image_csv} not found — skipping Fig 2")

    # ── Fig 3: Prompt vs image comparison ────────────────────────────────────
    if prompt_df is not None and image_df is not None:
        print("\n── Fig 3: Prompt vs image comparison ──")
        fig_comparison(prompt_df, image_df, args.out_dir)
    elif args.prompt_csv and args.image_csv:
        print("  Skipping Fig 3: need both --prompt_csv and --image_csv")

    # ── Fig 4: Image gallery ─────────────────────────────────────────────────
    if args.image_dir:
        if args.image_dir.exists():
            print("\n\u2500\u2500 Fig 4: Image gallery \u2500\u2500")
            fig_image_gallery(args.image_dir, args.out_dir)
        else:
            print(f"  Warning: {args.image_dir} not found \u2014 skipping Fig 4")

    # ── Supplementary Figure 3: Likert ────────────────────────────────────────
    if args.survey_xlsx:
        if args.survey_xlsx.exists():
            print("\n── Supplementary Figure 2: Likert ──")
            survey, col_map = load_survey(args.survey_xlsx)
            fig_likert_stacked(survey, col_map, args.out_dir)
            fig_likert_grid(survey, col_map, args.out_dir)
        else:
            print(f"  Warning: {args.survey_xlsx} not found — skipping Supplementary Figure 1")

    if not any([args.log_csv, args.prompt_csv, args.image_csv, args.image_dir, args.survey_xlsx]):
        print("Pass at least one of --log_csv, --prompt_csv, or --image_csv")
        return

    print(f"\nDone. Outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
