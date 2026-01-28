from __future__ import annotations

import time
import threading
import concurrent.futures  # kept (even though generation is synchronous)
import sqlite3
import re
import zipfile
import uuid
import json
from io import BytesIO, StringIO
from typing import Optional, Dict, Any, List, Tuple

import streamlit as st
from PIL import Image

from google import genai
from google.genai import types

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Engineering loneliness with GenAI",
    layout="wide",
)

DB_PATH = "workshop.db"
HOST_PASSWORD = "admin123"

MAX_CONCURRENT_GEN = 8
IMAGE_MODEL = "gemini-3-pro-image-preview"

# Show task instructions ONLY on the prompt page (not during consent/setup)
TASK_INSTRUCTIONS = """
Create a few photorealistic, everyday student-life scenes showing a student who might be lonely.  
Vary the situation across prompts.

Write your prompt in third person (e.g., “A student…”). Include:
- The student’s experience (socially, emotionally, or otherwise)
- Context (who is involved, where does it take place, what is happening)
- 2–3 visual cues (posture, gaze, distance to others, objects, lighting)
- Constraints (photorealistic, candid documentary style, natural colors; **no text/watermark**)

You can choose to submit or discard each image. Only submitted images appear in the gallery.  
Submit up to **2** images per group.
"""

DEFAULT_BUCKETS = "Unsorted, Interesting, Maybe, Other"

CONSENT_TEXT = """**Before you start**

In this activity, you will enter prompts that are sent to an AI provider to generate images.  
The AI provider will not use your prompts for model training.

This is a voluntary workshop activity; you can stop at any time without negative consequences.

Your prompts will be collected anonymously. Please do not include real names, emails, faces, or identifiable locations in your prompts.

The workshop includes a research component conducted by Jenna Pfeifer, PhD candidate, Dr.ir. Yke Bauke Eisma, Dr. D. Dodou, and Prof.dr.ir. Joost de Winter.  
The aim of the research is to investigate how young people conceptualise and recognise loneliness.  
For questions, contact: j.pfeifer@tudelft.nl.

**May the research team store and use your prompts and corresponding AI-generated images for future research and publications?**
"""

# ============================================================
# QUERY PARAMS (host mode via ?host=1)
# ============================================================
def qp_get(name: str, default: str = "") -> str:
    try:
        v = st.query_params.get(name, default)
        if isinstance(v, (list, tuple)):
            return str(v[0]) if v else default
        return str(v)
    except Exception:
        v = st.experimental_get_query_params().get(name, [default])
        return str(v[0]) if v else default


HOST_FLAG = qp_get("host", "0").strip().lower() in ("1", "true", "yes")

# ============================================================
# CONCURRENCY
# ============================================================
@st.cache_resource(show_spinner=False)
def global_gen_semaphore():
    return threading.BoundedSemaphore(MAX_CONCURRENT_GEN)

# ============================================================
# DATABASE
# ============================================================
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


@st.cache_resource(show_spinner=False)
def init_db_once() -> bool:
    with get_conn() as conn:
        # --- gallery (submitted images only)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gallery (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT,
                prompt TEXT,
                image_blob BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )

        # host curation migrations
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(gallery)")]
        added_host_cluster = False
        added_host_rank = False
        if "host_cluster" not in cols:
            conn.execute("ALTER TABLE gallery ADD COLUMN host_cluster TEXT DEFAULT 'Unsorted'")
            added_host_cluster = True
        if "host_rank" not in cols:
            conn.execute("ALTER TABLE gallery ADD COLUMN host_rank INTEGER")
            added_host_rank = True

        # gallery: session/consent + metrics migrations
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(gallery)")]

        def add_gallery_col(name: str, ddl: str):
            if name not in cols:
                conn.execute(f"ALTER TABLE gallery ADD COLUMN {ddl}")

        add_gallery_col("session_id", "session_id TEXT")
        add_gallery_col("group_size", "group_size INTEGER")
        add_gallery_col("consent_all_yes", "consent_all_yes INTEGER")

        add_gallery_col("model_name", "model_name TEXT")
        add_gallery_col("latency_ms", "latency_ms REAL")
        add_gallery_col("queue_wait_ms", "queue_wait_ms REAL")
        add_gallery_col("total_time_ms", "total_time_ms REAL")
        add_gallery_col("prompt_tokens", "prompt_tokens INTEGER")
        add_gallery_col("candidates_tokens", "candidates_tokens INTEGER")
        add_gallery_col("total_tokens", "total_tokens INTEGER")
        add_gallery_col("prompt_chars", "prompt_chars INTEGER")
        add_gallery_col("prompt_words", "prompt_words INTEGER")
        add_gallery_col("image_bytes", "image_bytes INTEGER")

        if added_host_cluster:
            conn.execute("UPDATE gallery SET host_cluster='Unsorted' WHERE host_cluster IS NULL OR host_cluster=''")
        if added_host_rank:
            conn.execute("UPDATE gallery SET host_rank=id WHERE host_rank IS NULL")

        # --- generation_log (all attempts)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                team_name TEXT,
                attempt_index INTEGER,
                prompt TEXT,
                status TEXT,  -- generated | submitted | discarded | error
                created_at TEXT DEFAULT (datetime('now')),

                model_name TEXT,
                latency_ms REAL,
                queue_wait_ms REAL,
                total_time_ms REAL,
                decision_time_ms REAL,

                prompt_tokens INTEGER,
                candidates_tokens INTEGER,
                total_tokens INTEGER,
                prompt_chars INTEGER,
                prompt_words INTEGER,
                image_bytes INTEGER,

                error_message TEXT,
                gallery_id INTEGER
            )
            """
        )

        # generation_log: consent migrations
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(generation_log)")]

        def add_log_col(name: str, ddl: str):
            if name not in cols:
                conn.execute(f"ALTER TABLE generation_log ADD COLUMN {ddl}")

        add_log_col("group_size", "group_size INTEGER")
        add_log_col("consent_all_yes", "consent_all_yes INTEGER")

        # --- session_meta (per-person consent)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_meta (
                session_id TEXT PRIMARY KEY,
                team_name TEXT,
                group_size INTEGER,
                consent_all_yes INTEGER,
                consent_choices_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )

        conn.commit()
    return True


init_db_once()


def upsert_session_meta(
    session_id: str,
    team_name: str,
    group_size: int,
    consent_all_yes: bool,
    consent_choices: List[str],
):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO session_meta (session_id, team_name, group_size, consent_all_yes, consent_choices_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                team_name=excluded.team_name,
                group_size=excluded.group_size,
                consent_all_yes=excluded.consent_all_yes,
                consent_choices_json=excluded.consent_choices_json
            """,
            (
                session_id,
                team_name,
                int(group_size),
                1 if consent_all_yes else 0,
                json.dumps(consent_choices),
            ),
        )
        conn.commit()

# ============================================================
# IMAGE HELPERS
# ============================================================
def image_to_blob(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def safe_filename(s: str, maxlen: int = 60) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:maxlen] or "item")

# ============================================================
# GEMINI HELPERS
# ============================================================
_thread_local = threading.local()


def _get_thread_client(api_key: str):
    if not hasattr(_thread_local, "client"):
        _thread_local.client = genai.Client(api_key=api_key)
    return _thread_local.client


def _extract_usage_tokens(response) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
    if usage is None:
        return {"prompt_tokens": None, "candidates_tokens": None, "total_tokens": None}

    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("prompt_token_count") or usage.get("promptTokens") or usage.get("prompt_tokens"),
            "candidates_tokens": usage.get("candidates_token_count")
            or usage.get("candidatesTokens")
            or usage.get("candidates_tokens"),
            "total_tokens": usage.get("total_token_count") or usage.get("totalTokens") or usage.get("total_tokens"),
        }

    return {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    }


def _generate_image_bytes_with_metrics(prompt: str, api_key: str) -> Dict[str, Any]:
    sem = global_gen_semaphore()

    t0 = time.perf_counter()
    sem.acquire()
    t1 = time.perf_counter()
    queue_wait_ms = (t1 - t0) * 1000.0

    client = _get_thread_client(api_key)

    api_start = time.perf_counter()
    response = None
    data = None
    err = None

    try:
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="3:2"),
            ),
        )

        img_part = next((p for p in getattr(response, "parts", []) if getattr(p, "inline_data", None)), None)
        if img_part is None:
            raise RuntimeError("No image returned. Try a simpler, more literal prompt.")

        data = img_part.inline_data.data
        if isinstance(data, str):
            raise RuntimeError("Gemini returned image data as str (expected bytes). Check SDK/response parsing.")
    except Exception as e:
        err = str(e)

    api_end = time.perf_counter()
    sem.release()

    usage = _extract_usage_tokens(response) if response is not None else {"prompt_tokens": None, "candidates_tokens": None, "total_tokens": None}
    prompt_chars = len(prompt or "")
    prompt_words = len((prompt or "").split())

    metrics = {
        "model_name": IMAGE_MODEL,
        "queue_wait_ms": queue_wait_ms,
        "latency_ms": (api_end - api_start) * 1000.0,
        "total_time_ms": (api_end - t0) * 1000.0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "candidates_tokens": usage.get("candidates_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "prompt_chars": prompt_chars,
        "prompt_words": prompt_words,
        "image_bytes": len(data) if data else None,
    }

    return {
        "ok": (err is None and data is not None),
        "image_bytes": data,
        "error": err,
        "metrics": metrics,
    }

# ============================================================
# GENERATION LOG
# ============================================================
def insert_generation_log(
    session_id: str,
    team: str,
    attempt_index: int,
    prompt: str,
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
    group_size: Optional[int] = None,
    consent_all_yes: Optional[bool] = None,
) -> int:
    metrics = metrics or {}

    consent_val = None
    if consent_all_yes is True:
        consent_val = 1
    elif consent_all_yes is False:
        consent_val = 0

    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO generation_log (
                session_id, team_name, attempt_index, prompt, status,
                model_name, latency_ms, queue_wait_ms, total_time_ms, decision_time_ms,
                prompt_tokens, candidates_tokens, total_tokens,
                prompt_chars, prompt_words, image_bytes,
                error_message, gallery_id,
                group_size, consent_all_yes
            )
            VALUES (?, ?, ?, ?, ?,
                    ?, ?, ?, ?, NULL,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, NULL,
                    ?, ?)
            """,
            (
                session_id,
                team,
                attempt_index,
                prompt,
                status,
                metrics.get("model_name"),
                metrics.get("latency_ms"),
                metrics.get("queue_wait_ms"),
                metrics.get("total_time_ms"),
                metrics.get("prompt_tokens"),
                metrics.get("candidates_tokens"),
                metrics.get("total_tokens"),
                metrics.get("prompt_chars"),
                metrics.get("prompt_words"),
                metrics.get("image_bytes"),
                error_message,
                group_size,
                consent_val,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def finalize_generation_log(
    log_id: int,
    status: str,
    decision_time_ms: Optional[float] = None,
    gallery_id: Optional[int] = None,
):
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE generation_log
            SET status = ?,
                decision_time_ms = COALESCE(?, decision_time_ms),
                gallery_id = COALESCE(?, gallery_id)
            WHERE id = ?
            """,
            (status, decision_time_ms, gallery_id, log_id),
        )
        conn.commit()

# ============================================================
# DB OPERATIONS (gallery = submitted only)
# ============================================================
def save_submission(
    team: str,
    prompt: str,
    img: Image.Image,
    session_id: str,
    group_size: Optional[int],
    consent_all_yes: Optional[bool],
    metrics: Optional[Dict[str, Any]] = None,
) -> int:
    metrics = metrics or {}

    consent_val = None
    if consent_all_yes is True:
        consent_val = 1
    elif consent_all_yes is False:
        consent_val = 0

    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO gallery (
                team_name, prompt, image_blob,
                host_cluster, host_rank,
                session_id, group_size, consent_all_yes,
                model_name, latency_ms, queue_wait_ms, total_time_ms,
                prompt_tokens, candidates_tokens, total_tokens,
                prompt_chars, prompt_words, image_bytes
            )
            VALUES (
                ?, ?, ?,
                'Unsorted', NULL,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
            """,
            (
                team,
                prompt,
                image_to_blob(img),
                session_id,
                group_size,
                consent_val,
                metrics.get("model_name"),
                metrics.get("latency_ms"),
                metrics.get("queue_wait_ms"),
                metrics.get("total_time_ms"),
                metrics.get("prompt_tokens"),
                metrics.get("candidates_tokens"),
                metrics.get("total_tokens"),
                metrics.get("prompt_chars"),
                metrics.get("prompt_words"),
                metrics.get("image_bytes"),
            ),
        )
        conn.execute("UPDATE gallery SET host_rank=id WHERE host_rank IS NULL")
        conn.commit()
        return int(cur.lastrowid)


def get_gallery_meta(order_by: str = "curated") -> List[Dict[str, Any]]:
    with get_conn() as conn:
        if order_by == "curated":
            rows = conn.execute(
                """
                SELECT
                    id,
                    COALESCE(team_name, '') AS team_name,
                    COALESCE(prompt, '') AS prompt,
                    COALESCE(created_at, '') AS created_at,
                    COALESCE(host_cluster, 'Unsorted') AS host_cluster,
                    COALESCE(host_rank, id) AS host_rank,

                    COALESCE(session_id, '') AS session_id,
                    group_size,
                    consent_all_yes,

                    COALESCE(model_name, '') AS model_name,
                    latency_ms,
                    queue_wait_ms,
                    total_time_ms,
                    prompt_tokens,
                    candidates_tokens,
                    total_tokens,
                    prompt_chars,
                    prompt_words,
                    image_bytes
                FROM gallery
                ORDER BY host_cluster, COALESCE(host_rank, id), id
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                    id,
                    COALESCE(team_name, '') AS team_name,
                    COALESCE(prompt, '') AS prompt,
                    COALESCE(created_at, '') AS created_at,
                    COALESCE(host_cluster, 'Unsorted') AS host_cluster,
                    COALESCE(host_rank, id) AS host_rank,

                    COALESCE(session_id, '') AS session_id,
                    group_size,
                    consent_all_yes,

                    COALESCE(model_name, '') AS model_name,
                    latency_ms,
                    queue_wait_ms,
                    total_time_ms,
                    prompt_tokens,
                    candidates_tokens,
                    total_tokens,
                    prompt_chars,
                    prompt_words,
                    image_bytes
                FROM gallery
                ORDER BY id DESC
                """
            ).fetchall()

    return [dict(r) for r in rows]


def get_gallery_blobs() -> Dict[int, bytes]:
    with get_conn() as conn:
        rows = conn.execute("SELECT id, image_blob FROM gallery").fetchall()
    return {int(r["id"]): r["image_blob"] for r in rows}


def get_generation_log_rows() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM generation_log ORDER BY id ASC").fetchall()
    return [dict(r) for r in rows]


def get_session_meta_rows() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM session_meta ORDER BY created_at ASC").fetchall()
    return [dict(r) for r in rows]


def update_host_layout(layout_containers: List[Dict[str, Any]]):
    seen = set()
    updates: List[Tuple[str, int, int]] = []

    for c in layout_containers:
        cluster = c["header"]
        for rank, label in enumerate(c["items"]):
            try:
                image_id = int(str(label).split("|", 1)[0].strip())
            except Exception:
                continue
            if image_id in seen:
                continue
            seen.add(image_id)
            updates.append((cluster, rank, image_id))

    if not updates:
        return

    with get_conn() as conn:
        conn.executemany("UPDATE gallery SET host_cluster=?, host_rank=? WHERE id=?", updates)
        conn.commit()


def normalize_layout(layout_containers: List[Dict[str, Any]]) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    out = []
    for c in layout_containers:
        ids = []
        for label in c["items"]:
            try:
                ids.append(int(str(label).split("|", 1)[0].strip()))
            except Exception:
                pass
        out.append((c["header"], tuple(ids)))
    return tuple(out)

# ============================================================
# DOWNLOAD HELPERS
# ============================================================
def export_gallery_csv_bytes() -> bytes:
    import pandas as pd

    meta = get_gallery_meta(order_by="curated")
    cols = [
        "id",
        "team_name",
        "prompt",
        "created_at",
        "host_cluster",
        "host_rank",
        "session_id",
        "group_size",
        "consent_all_yes",
        "model_name",
        "latency_ms",
        "queue_wait_ms",
        "total_time_ms",
        "prompt_tokens",
        "candidates_tokens",
        "total_tokens",
        "prompt_chars",
        "prompt_words",
        "image_bytes",
    ]
    df = pd.DataFrame(meta, columns=cols)
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def export_generation_log_csv_bytes() -> bytes:
    import pandas as pd

    rows = get_generation_log_rows()
    df = pd.DataFrame(rows)
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def export_session_meta_csv_bytes() -> bytes:
    import pandas as pd

    rows = get_session_meta_rows()
    df = pd.DataFrame(rows)
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def export_zip_bytes(include_csv: bool = True, include_log: bool = True, include_session_meta: bool = True) -> bytes:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, host_cluster, image_blob
            FROM gallery
            ORDER BY COALESCE(host_cluster,'Unsorted'), COALESCE(host_rank,id), id
            """
        ).fetchall()

    gallery_csv = export_gallery_csv_bytes() if include_csv else None
    genlog_csv = export_generation_log_csv_bytes() if include_log else None
    session_meta_csv = export_session_meta_csv_bytes() if include_session_meta else None

    zbuf = BytesIO()
    with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        if include_csv and gallery_csv:
            z.writestr("gallery_metadata.csv", gallery_csv)
        if include_log and genlog_csv:
            z.writestr("generation_log.csv", genlog_csv)
        if include_session_meta and session_meta_csv:
            z.writestr("session_meta.csv", session_meta_csv)

        for r in rows:
            rid = int(r["id"])
            cluster = safe_filename(r["host_cluster"] or "Unsorted")
            fname = f"images/{cluster}/{rid:04d}.png"
            z.writestr(fname, r["image_blob"])

    zbuf.seek(0)
    return zbuf.getvalue()

# ============================================================
# UI HEADER (TITLE ONLY — no task text during consent/setup)
# ============================================================
st.title("Engineering loneliness with GenAI")

# ============================================================
# HOST MODE
# ============================================================
if HOST_FLAG:
    try:
        from streamlit_sortables import sort_items
    except Exception:
        st.error("Missing dependency: streamlit-sortables. Add it to requirements.txt or pip install streamlit-sortables.")
        st.stop()

    with st.sidebar:
        st.header("Host login")
        pw = st.text_input("Admin password", type="password")
        is_host = (pw == HOST_PASSWORD)

        st.divider()
        view = st.radio("Host view", ["Gallery wall", "Curate (drag & drop)", "Download"], index=1)

        bucket_names = st.text_input("Buckets (comma-separated)", value=DEFAULT_BUCKETS)
        buckets = [b.strip() for b in bucket_names.split(",") if b.strip()]
        if not buckets:
            buckets = ["Unsorted"]
        if "Unsorted" not in buckets:
            buckets = ["Unsorted"] + buckets

        n_cols = st.slider("Gallery columns", 2, 6, 4)
        compact = st.toggle("Compact captions", value=True)

        st.divider()
        st.subheader("Privacy")
        reveal_team = st.toggle("Reveal team names (admin only)", value=False)
        reveal_prompt = st.toggle("Reveal prompts (admin only)", value=False)
        show_ids = st.toggle("Show image #", value=True)

        st.divider()
        if st.button("Refresh"):
            st.rerun()

    if not is_host:
        st.warning("Host mode is enabled via the URL. Enter the admin password in the sidebar.")
        st.stop()

    meta = get_gallery_meta(order_by="curated")
    blobs_by_id = get_gallery_blobs()
    meta_by_id = {int(m["id"]): m for m in meta}

    with st.sidebar.expander("Lookup (image # → team)", expanded=False):
        st.dataframe(
            [{"id": m["id"], "team_name": m["team_name"], "consent_all_yes": m.get("consent_all_yes")} for m in meta],
            use_container_width=True,
        )

    by_cluster = {b: [] for b in buckets}
    for m in meta:
        image_id = int(m["id"])
        label = f"{image_id}"
        cluster = m["host_cluster"] if m["host_cluster"] in by_cluster else "Unsorted"
        by_cluster[cluster].append((m["host_rank"], image_id, label))

    containers = []
    for b in buckets:
        items = [lbl for _, _, lbl in sorted(by_cluster[b], key=lambda x: (x[0], x[1]))]
        containers.append({"header": b, "items": items})

    def render_bucket_images(layout_containers):
        for c in layout_containers:
            st.markdown(f"### {c['header']}")
            ids: List[int] = []
            for label in c["items"]:
                try:
                    ids.append(int(str(label).split("|", 1)[0].strip()))
                except Exception:
                    pass

            if not ids:
                st.caption("—")
                continue

            cols = st.columns(n_cols)
            for i, image_id in enumerate(ids):
                blob = blobs_by_id.get(image_id)
                if not blob:
                    continue

                m = meta_by_id.get(image_id, {})
                with cols[i % n_cols]:
                    st.image(blob, use_container_width=True)

                    if compact:
                        if show_ids:
                            cap = f"#{image_id}"
                            if reveal_team:
                                cap += f" • {m.get('team_name','')}"
                            st.caption(cap)
                    else:
                        header = f"Image #{image_id}" if show_ids else "Image"
                        with st.expander(header):
                            if reveal_team:
                                st.write(f"**Team:** {m.get('team_name','')}")
                            if reveal_prompt:
                                st.write(f"**Prompt:** {m.get('prompt','')}")
                            st.write(f"**Consent all yes:** {m.get('consent_all_yes')}")

    if view == "Curate (drag & drop)":
        st.subheader("Drag & drop to cluster and reorder")
        new_containers = sort_items(containers, multi_containers=True)
        if normalize_layout(new_containers) != normalize_layout(containers):
            update_host_layout(new_containers)
            containers = new_containers

        st.divider()
        st.subheader("Preview (images follow your bucket order)")
        render_bucket_images(containers)
        st.stop()

    if view == "Gallery wall":
        render_bucket_images(containers)
        st.stop()

    if view == "Download":
        st.subheader("Download gallery data")

        st.download_button(
            "Download gallery_metadata.csv (submitted images + consent flags)",
            data=export_gallery_csv_bytes(),
            file_name="gallery_metadata.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download generation_log.csv (all attempts incl. discarded)",
            data=export_generation_log_csv_bytes(),
            file_name="generation_log.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download session_meta.csv (per-person consent choices)",
            data=export_session_meta_csv_bytes(),
            file_name="session_meta.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download ZIP (images + CSVs)",
            data=export_zip_bytes(include_csv=True, include_log=True, include_session_meta=True),
            file_name="gallery_images_and_metadata.zip",
            mime="application/zip",
        )

        st.info("ZIP includes: gallery_metadata.csv, generation_log.csv, session_meta.csv, and images/<bucket>/<id>.png")
        st.stop()

# ============================================================
# PARTICIPANT MODE
# ============================================================
def reset_participant_state():
    keys = [
        "setup_complete",
        "setup_phase",
        "consent_index",
        "group_size",
        "team_name",
        "session_id",
        "attempt_index",
        "draft_bytes",
        "draft_metrics",
        "draft_log_id",
        "draft_ready_at",
        "last_prompt_used",
        "consent_all_yes",
        "consent_choices",
        "submitted_count",
    ]
    for k in keys:
        st.session_state.pop(k, None)


with st.sidebar:
    st.header("Setup")
    if st.button("Restart setup"):
        reset_participant_state()
        st.rerun()
    st.divider()
    st.caption("If the app feels stuck, restart setup to clear in-progress state.")

# defaults
st.session_state.setdefault("setup_complete", False)
st.session_state.setdefault("setup_phase", "group_size")  # group_size -> consent -> group_name
st.session_state.setdefault("consent_index", 0)
st.session_state.setdefault("group_size", 1)
st.session_state.setdefault("team_name", "")
st.session_state.setdefault("session_id", str(uuid.uuid4()))
st.session_state.setdefault("attempt_index", 0)
st.session_state.setdefault("consent_choices", None)
st.session_state.setdefault("consent_all_yes", None)
st.session_state.setdefault("submitted_count", 0)

# ---------- SETUP FLOW (multi-page, one person at a time) ----------
if not st.session_state["setup_complete"]:
    phase = st.session_state["setup_phase"]

    if phase == "group_size":
        st.subheader("Welcome")
        n = int(
            st.number_input(
                "How many people are in your group?",
                min_value=1,
                max_value=20,
                value=max(1, int(st.session_state.get("group_size") or 1)),
                step=1,
            )
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Next"):
                st.session_state["group_size"] = n
                st.session_state["consent_index"] = 0
                st.session_state["consent_choices"] = [""] * n
                st.session_state["consent_all_yes"] = None
                st.session_state["submitted_count"] = 0

                st.session_state["session_id"] = str(uuid.uuid4())
                st.session_state["setup_phase"] = "consent"
                st.rerun()
        with col2:
            st.caption("Next you will complete consent privately, one person at a time.")

        st.stop()

    if phase == "consent":
        n = int(st.session_state["group_size"])
        idx = int(st.session_state["consent_index"])
        person_num = idx + 1

        st.subheader(f"Consent (Person {person_num} of {n})")
        st.markdown(CONSENT_TEXT)

        choice = st.selectbox(
            "Select your answer",
            options=["Select…", "No", "Yes"],
            index=0,
            key=f"consent_choice_person_{person_num}",
        )

        col_a, col_b, col_c = st.columns([1, 1, 3])

        with col_a:
            if st.button("Back") and idx > 0:
                st.session_state["consent_index"] = idx - 1
                st.rerun()

        with col_b:
            if st.button("Next"):
                if choice == "Select…":
                    st.warning("Please select Yes or No to continue.")
                else:
                    choices = st.session_state["consent_choices"] or [""] * n
                    choices[idx] = choice
                    st.session_state["consent_choices"] = choices

                    filled = all(c in ("Yes", "No") for c in choices)
                    all_yes = filled and all(c == "Yes" for c in choices)
                    st.session_state["consent_all_yes"] = all_yes if filled else None

                    upsert_session_meta(
                        session_id=st.session_state["session_id"],
                        team_name=st.session_state.get("team_name", "") or "",
                        group_size=n,
                        consent_all_yes=all_yes if filled else False,
                        consent_choices=choices,
                    )

                    if idx + 1 < n:
                        st.session_state["consent_index"] = idx + 1
                        st.rerun()
                    else:
                        st.session_state["setup_phase"] = "group_name"
                        st.rerun()

        with col_c:
            st.caption("This screen is intended to be completed by one person at a time.")

        st.stop()

    if phase == "group_name":
        st.subheader("Group name")
        group = st.text_input("Enter your group name", placeholder="e.g., Team Blue", value=st.session_state.get("team_name", ""))

        if st.button("Continue"):
            if not group.strip():
                st.error("Please enter a group name.")
            else:
                st.session_state["team_name"] = group.strip()
                st.session_state["attempt_index"] = 0
                st.session_state["submitted_count"] = 0

                n = int(st.session_state["group_size"])
                choices = st.session_state["consent_choices"] or [""] * n
                safe_choices = [(c if c in ("Yes", "No") else "No") for c in choices]
                all_yes = all(c == "Yes" for c in safe_choices)

                st.session_state["consent_choices"] = safe_choices
                st.session_state["consent_all_yes"] = all_yes

                upsert_session_meta(
                    session_id=st.session_state["session_id"],
                    team_name=st.session_state["team_name"],
                    group_size=n,
                    consent_all_yes=all_yes,
                    consent_choices=safe_choices,
                )

                st.session_state["setup_complete"] = True
                st.rerun()

        st.stop()

# ---------- MAIN APP ----------
team_name = st.session_state["team_name"]
session_id = st.session_state["session_id"]
group_size = st.session_state.get("group_size")
consent_all_yes = st.session_state.get("consent_all_yes")

with st.sidebar:
    st.header("Your info")
    st.success(f"Group: {team_name}")
    # no consent status shown to participants

# Secrets check
if "google_api" not in st.secrets or "key" not in st.secrets["google_api"]:
    st.error("Missing Google API key. Set st.secrets['google_api']['key'] in your Streamlit secrets.")
    st.stop()

api_key = st.secrets["google_api"]["key"]

# Task instructions ONLY here (prompting page)
st.markdown("---")
st.markdown(TASK_INSTRUCTIONS)

# Soft counter only (NO enforcement)
submitted_count = int(st.session_state.get("submitted_count", 0))
st.caption(f"Submitted so far (this session): **{submitted_count}**")
st.caption("Tip: aim for 1–2 submissions at a time; you can submit more later after discussion/refinement.")

prompt = st.text_area("Prompt", height=220, placeholder="Photorealistic documentary photograph of...")

# Generate (synchronous)
if st.button("Generate image", key="gen_btn"):
    if not prompt.strip():
        st.warning("Please write a prompt.")
    else:
        # discard prior draft if any
        if st.session_state.get("draft_log_id") is not None:
            dt_ms = None
            if st.session_state.get("draft_ready_at") is not None:
                dt_ms = (time.time() - st.session_state["draft_ready_at"]) * 1000.0
            finalize_generation_log(st.session_state["draft_log_id"], status="discarded", decision_time_ms=dt_ms)

            st.session_state.pop("draft_bytes", None)
            st.session_state.pop("draft_metrics", None)
            st.session_state.pop("draft_log_id", None)
            st.session_state.pop("draft_ready_at", None)

        with st.spinner("Generating…"):
            result = _generate_image_bytes_with_metrics(prompt.strip(), api_key)

        st.session_state["attempt_index"] += 1
        attempt_index = st.session_state["attempt_index"]
        metrics = result.get("metrics", {}) or {}

        if result.get("ok"):
            st.session_state["draft_bytes"] = result["image_bytes"]
            st.session_state["draft_metrics"] = metrics
            st.session_state["draft_ready_at"] = time.time()
            st.session_state["last_prompt_used"] = prompt.strip()

            log_id = insert_generation_log(
                session_id=session_id,
                team=team_name,
                attempt_index=attempt_index,
                prompt=prompt.strip(),
                status="generated",
                metrics=metrics,
                error_message=None,
                group_size=group_size,
                consent_all_yes=consent_all_yes,
            )
            st.session_state["draft_log_id"] = log_id
        else:
            err = result.get("error") or "Unknown error"
            insert_generation_log(
                session_id=session_id,
                team=team_name,
                attempt_index=attempt_index,
                prompt=prompt.strip(),
                status="error",
                metrics=metrics,
                error_message=err,
                group_size=group_size,
                consent_all_yes=consent_all_yes,
            )
            st.error(f"Generation failed: {err}")

# Preview + submit/discard
if "draft_bytes" in st.session_state:
    img = Image.open(BytesIO(st.session_state["draft_bytes"]))
    st.image(img, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Submit to gallery", key="submit_btn"):
            used_prompt = st.session_state.get("last_prompt_used", prompt).strip() or prompt.strip()
            metrics = st.session_state.get("draft_metrics", {}) or {}

            gallery_id = save_submission(
                team=team_name or "Anonymous",
                prompt=used_prompt,
                img=img,
                session_id=session_id,
                group_size=group_size,
                consent_all_yes=consent_all_yes,
                metrics=metrics,
            )

            if st.session_state.get("draft_log_id") is not None:
                dt_ms = None
                if st.session_state.get("draft_ready_at") is not None:
                    dt_ms = (time.time() - st.session_state["draft_ready_at"]) * 1000.0
                finalize_generation_log(
                    st.session_state["draft_log_id"],
                    status="submitted",
                    decision_time_ms=dt_ms,
                    gallery_id=gallery_id,
                )

            st.session_state["submitted_count"] = int(st.session_state.get("submitted_count", 0)) + 1

            st.session_state.pop("draft_bytes", None)
            st.session_state.pop("draft_metrics", None)
            st.session_state.pop("draft_log_id", None)
            st.session_state.pop("draft_ready_at", None)

            st.success("Saved to gallery.")
            st.rerun()

    with col_b:
        if st.button("Discard (don’t submit)", key="discard_btn"):
            if st.session_state.get("draft_log_id") is not None:
                dt_ms = None
                if st.session_state.get("draft_ready_at") is not None:
                    dt_ms = (time.time() - st.session_state["draft_ready_at"]) * 1000.0
                finalize_generation_log(st.session_state["draft_log_id"], status="discarded", decision_time_ms=dt_ms)

            st.session_state.pop("draft_bytes", None)
            st.session_state.pop("draft_metrics", None)
            st.session_state.pop("draft_log_id", None)
            st.session_state.pop("draft_ready_at", None)

            st.info("Discarded. You can generate again.")
            st.rerun()



