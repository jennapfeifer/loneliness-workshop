# Research repository for "Synthetic photo-elicitation with text-to-image AI: A workshop method for discussing youth loneliness"

Code and analysis for the paper *Synthetic photo-elicitation with text-to-image AI: A workshop method for discussing youth loneliness*. The repository covers the full pipeline from the live Streamlit workshop application through qualitative coding of prompts and images to the final publication figures.

---

## Repository structure

```
.
├── workshop.py                           # Streamlit workshop application
├── code_prompts.py                       # LLM-based qualitative coding of text prompts
├── code_images.py                        # LLM-based visual coding of generated images
├── validate_loneliness.py                # Sanity check: do prompts/images depict loneliness?
├── figures.py                            # All publication figures + generation-log descriptives
├── codebook_images.md                    # Visual codebook applied to generated images (23 binary codes)
├── codebook_prompts.md                   # Thematic codebook applied to text prompts (25 binary codes)
├── coding_prompt_prompts.md              # System prompt instructing the model how to apply the prompt codebook
```

### Codebook files

**`codebook_images.md`** defines what to code in the *generated images*. It contains 23 binary codes organised into five sections: setting and environment (A), social configuration (B), visual cues for loneliness (C), AI default tropes (D), and image–prompt adequacy (E). The coder (Gemini vision) looks at each image and marks what is literally, visually present — it is about what the model *produced*.

**`codebook_prompts.md`** defines what to code in the *text prompts*. It contains 25 binary codes covering visual and environmental cues (A), social and relational cues (B), AI stereotypes and tropes (C), prompt characteristics (D), and researcher-added inductive codes (E). The coder (Gemini text) reads each participant's written prompt and marks which themes and cues are present — it is about what participants *intended*.

**`coding_prompt_prompts.md`** is not a codebook. It is the system prompt passed to Gemini when running `code_prompts.py` — the instructions that tell the model how to apply `codebook_prompts.md`: code only what is explicitly stated, when in doubt code 0, return a CSV with one row per prompt. The equivalent instructions for image coding are embedded directly in `code_images.py`.

---

## Requirements

```
Python >= 3.10
```

Install dependencies:

```bash
pip install streamlit Pillow google-genai python-dotenv \
            pandas matplotlib numpy openpyxl tqdm rapidfuzz
```

API keys are required for the coding pipelines and the workshop app:

- **Google Gemini API key** — for the workshop app (`workshop.py`) and both coding scripts (`code_prompts.py`, `code_images.py`, `validate_loneliness.py`).

---

## 1. Workshop application — `workshop.py`

A Streamlit app used during the live workshop. Participants write prompts, generate photorealistic images via the Gemini image API, and choose to submit or discard each image. Submitted images appear in a shared gallery. All generation attempts (including discards) are logged to a local SQLite database.

**Setup**

Add your Google API key to `.streamlit/secrets.toml`:

```toml
[google_api]
key = "YOUR_GOOGLE_API_KEY"
```

**Run**

```bash
streamlit run workshop.py
```

Open `?host=1` in the URL for the facilitator/host view.

**Outputs**

- `workshop.db` — SQLite database containing two tables:
  - `generation_log` — every generation attempt with status (`submitted`, `discarded`, `generated`, `error`), timing, token counts, and prompt text
  - `gallery` — submitted images with metadata

**Export to CSV for analysis**

```bash
sqlite3 -header -csv workshop.db "SELECT * FROM generation_log;" > Data/generation_log.csv
sqlite3 -header -csv workshop.db "SELECT id, prompt FROM gallery WHERE consent_all_yes=1;" > Data/gallery_metadata.csv
```

---

## 2. Qualitative coding — `code_prompts.py`

Codes each text prompt against a qualitative codebook using Gemini. Runs multiple independent coding passes and aggregates by majority vote to improve reliability.

**Requires**

| File | Description |
|------|-------------|
| `Data/generation_log.csv` | Exported from `workshop.db` |
| `codebook_prompts.md` | Binary codebook for prompt themes |
| `coding_prompt_prompts.md` | System prompt sent to the model |

**Run**

```bash
export GOOGLE_API_KEY=...

python code_prompts.py \
    --input_csv   Data/generation_log.csv \
    --prompt_md   coding_prompt_prompts.md \
    --codebook_md codebook_prompts.md \
    --out_dir     results/prompts_coded \
    --runs        5
```

**Outputs** (written to `--out_dir/TIMESTAMP/`):

- `final_majority_vote.csv` — one row per prompt, one column per binary code
- `final_majority_vote_agreement.csv` — inter-run agreement rates
- `manifest.json` — run metadata and code mappings

---

## 3. Visual coding of images — `code_images.py`

Codes each generated image against the visual codebook (23 binary codes across 5 sections) using Gemini vision. Runs multiple independent coding passes and aggregates by majority vote.

**Requires**

| File | Description |
|------|-------------|
| `Data/gallery_metadata.csv` | Exported from `workshop.db` (must include `id`, `consent_all_yes`) |
| `Data/gallery_images/` | Folder of image files named `{id}.png` |
| `codebook_images.md` | Visual codebook (included in this repo) |

**Run**

```bash
export GOOGLE_API_KEY=...

python code_images.py \
    --metadata_csv Data/gallery_metadata.csv \
    --image_dir    Data/gallery_images \
    --codebook_md  codebook_images.md \
    --out_dir      ImageAnalysis/results/images_coded \
    --runs         3
```

**Outputs** (written to `--out_dir/TIMESTAMP/`):

- `final_majority_vote.csv` — one row per image, one column per binary code
- `final_majority_vote_agreement.csv` — inter-run agreement rates
- `manifest.json` — run metadata

---


## 4. Figures and descriptives — `figures.py`

Produces all publication figures and the generation-log descriptive statistics table. All arguments are optional — only outputs whose inputs are present are produced.

**Requires**

| Argument | File | Description |
|----------|------|-------------|
| `--log_csv` | `Data/generation_log.csv` | Exported from `workshop.db` |
| `--prompt_csv` | `results/prompts_coded/TIMESTAMP/final_majority_vote.csv` | From `code_prompts.py` |
| `--image_csv` | `results/images_coded/TIMESTAMP/final_majority_vote.csv` | From `code_images.py` |
| `--image_dir` | `Data/gallery_images/` | Folder of image files |
| `--survey_xlsx` | `Data/qualtrics_export.xlsx` | Qualtrics survey export |

**Run**

```bash
python figures.py \
    --log_csv      Data/generation_log.csv \
    --prompt_csv   results/prompts_coded/TIMESTAMP/final_majority_vote.csv \
    --image_csv    results/images_coded/TIMESTAMP/final_majority_vote.csv \
    --image_dir    Data/gallery_images \
    --survey_xlsx  Data/qualtrics_export.xlsx \
    --out_dir      figures_pub
```

**Outputs**

| File | Description |
|------|-------------|
| `descriptives.md` | Generation-log statistics table (all / submitted / discarded) |
| `descriptives_tidy.csv` | Same in tidy long format |
| `fig2_comparison_by_gap.png` | **Figure 2**: prompt vs image rate, sorted by intent–output gap |
| `fig2_comparison_grouped.png` | **Figure 2 (alt)**: same, grouped by over- vs under-delivery |
| `fig3_image_gallery.png` | **Figure 3**: curated image gallery with participant prompts |
| `sup1_likert_grid.png` | **Supplementary Figure 1**: Likert survey items, per-item grid |
| `sup2_prompt_frequency.png` | **Supplementary Figure 2**: prompt code prevalence across all prompts |
| `sup3_image_frequency.png` | **Supplementary Figure 3**: image code prevalence, faceted by codebook section |

---

## Data

**To set up the data folder:**

1. Download the shared `Data/` folder from [Dropbox link]
2. Place it in the root of this repository so the structure looks like:

```
.
├── Data/
│   ├── generation_log.csv
│   ├── gallery_metadata.csv
│   ├── gallery_images/
│   │   ├── 0001.png
│   │   └── ...
│   ├── qualtrics_export.csv
│   ├── prompts_coded_final.csv        ← final_majority_vote.csv from code_prompts.py
│   └── images_coded_final.csv         ← final_majority_vote.csv from code_images.py
├── figures.py
├── code_prompts.py
└── ...
```

When running `figures.py`, point `--prompt_csv` and `--image_csv` at these files:

```bash
python figures.py \
    --log_csv     Data/generation_log.csv \
    --prompt_csv  Data/prompts_coded_final.csv \
    --image_csv   Data/images_coded_final.csv \
    --image_dir   Data/gallery_images \
    --survey_xlsx Data/qualtrics_export.csv \
    --out_dir     figures_pub
```

If you do not have access to the Dropbox folder, contact j.pfeifer@tudelft.nl.

Only consented data is used in analysis; `code_images.py` and `figures.py` filter automatically on `consent_all_yes`.


---

## Contact

Jenna Pfeifer — j.pfeifer@tudelft.nl
