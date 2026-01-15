import time
import threading
import concurrent.futures
import sqlite3
import random
import re
import zipfile
from io import BytesIO, StringIO

import streamlit as st
from PIL import Image

from google import genai
from google.genai import types

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Engineering loneliness with GenAI",
    layout="wide"
)

DB_PATH = "workshop.db"
HOST_PASSWORD = "admin123"

MAX_CONCURRENT_GEN = 8  # across all users on this Streamlit instance

TASK_BYLINE = (
    "Create a few photorealistic, everyday student-life scenes showing a student who might be lonely. "
    "Try to inject context (e.g., home, study location, otherwise), and try to inject the student's experience "
    "(socially, emotionally, or otherwise). Try to vary the situations as much as possible."
)

DEFAULT_BUCKETS = "Unsorted, Interesting, Maybe, Other"

# ============================================================
# QUERY PARAMS (host mode via ?host=1)
# ============================================================
def qp_get(name: str, default: str = "") -> str:
    try:
        v = st.query_params.get(name, default)  # new API
        if isinstance(v, (list, tuple)):
            return str(v[0]) if v else default
        return str(v)
    except Exception:
        # fallback for older Streamlit
        v = st.experimental_get_query_params().get(name, [default])
        return str(v[0]) if v else default

HOST_FLAG = qp_get("host", "0").strip().lower() in ("1", "true", "yes")

# ============================================================
# CONCURRENCY
# ============================================================
@st.cache_resource
def global_gen_semaphore():
    return threading.BoundedSemaphore(MAX_CONCURRENT_GEN)

# ============================================================
# DATABASE
# ============================================================
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS gallery (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT,
            prompt TEXT,
            image_blob BLOB,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """)

        # --- MIGRATION: add host curation fields if missing
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(gallery)")]
        if "host_cluster" not in cols:
            conn.execute("ALTER TABLE gallery ADD COLUMN host_cluster TEXT DEFAULT 'Unsorted'")
        if "host_rank" not in cols:
            conn.execute("ALTER TABLE gallery ADD COLUMN host_rank INTEGER")

        # normalize defaults (helps existing rows)
        conn.execute("UPDATE gallery SET host_cluster='Unsorted' WHERE host_cluster IS NULL OR host_cluster=''")
        conn.execute("UPDATE gallery SET host_rank=id WHERE host_rank IS NULL")
        conn.commit()

init_db()

# ============================================================
# IMAGE HELPERS
# ============================================================
def image_to_blob(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def blob_to_image(blob: bytes) -> Image.Image:
    return Image.open(BytesIO(blob))

def safe_filename(s: str, maxlen: int = 60) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:maxlen] or "team")

# ============================================================
# GEMINI HELPERS
# ============================================================
_thread_local = threading.local()

def _get_thread_client(api_key: str):
    # One client per worker thread.
    if not hasattr(_thread_local, "client"):
        _thread_local.client = genai.Client(api_key=api_key)
    return _thread_local.client

def _generate_image_bytes(prompt: str, api_key: str) -> bytes:
    sem = global_gen_semaphore()
    sem.acquire()
    try:
        client = _get_thread_client(api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="3:2")
            )
        )
        img_part = next((p for p in response.parts if getattr(p, "inline_data", None)), None)
        if img_part is None:
            raise RuntimeError("No image returned. Try a simpler, more literal prompt.")
        data = img_part.inline_data.data
        if isinstance(data, str):
            raise RuntimeError("Gemini returned image data as str (expected bytes). Check SDK/response parsing.")
        return data
    finally:
        sem.release()

# ============================================================
# DB OPERATIONS
# ============================================================
def save_submission(team: str, prompt: str, img: Image.Image):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO gallery (team_name, prompt, image_blob, host_cluster, host_rank)
            VALUES (?, ?, ?, 'Unsorted', NULL)
            """,
            (team, prompt, image_to_blob(img))
        )
        # give it a stable default rank = id
        conn.execute("UPDATE gallery SET host_rank=id WHERE host_rank IS NULL")
        conn.commit()

def get_submissions_full(order_by: str = "newest"):
    """
    Returns list of dicts including decoded PIL images.
    order_by: 'newest' or 'curated'
    """
    with get_conn() as conn:
        if order_by == "curated":
            rows = conn.execute("""
                SELECT * FROM gallery
                ORDER BY host_cluster, host_rank, id
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM gallery
                ORDER BY id DESC
            """).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        d["image"] = blob_to_image(d["image_blob"])
        out.append(d)
    return out

def get_gallery_meta():
    """
    Lightweight: no blobs.
    """
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                id,
                COALESCE(team_name, '') AS team_name,
                COALESCE(prompt, '') AS prompt,
                COALESCE(created_at, '') AS created_at,
                COALESCE(host_cluster, 'Unsorted') AS host_cluster,
                COALESCE(host_rank, id) AS host_rank
            FROM gallery
            ORDER BY host_cluster, COALESCE(host_rank, id), id
        """).fetchall()
    return [dict(r) for r in rows]

def update_host_layout(layout_containers):
    """
    layout_containers: [{'header': 'Interesting', 'items': ['12 | Team A', ...]}, ...]
    Stores host_cluster and host_rank based on order in each bucket.
    """
    seen = set()
    updates = []

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
        conn.executemany(
            "UPDATE gallery SET host_cluster=?, host_rank=? WHERE id=?",
            updates
        )
        conn.commit()

def normalize_layout(layout_containers):
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
def export_csv_bytes() -> bytes:
    meta = get_gallery_meta()
    import pandas as pd
    df = pd.DataFrame(meta, columns=["id", "team_name", "prompt", "created_at", "host_cluster", "host_rank"])
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def export_zip_bytes(include_csv: bool = True) -> bytes:
    # Pull raw blobs (faster than re-encoding PIL)
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT id, team_name, prompt, created_at, host_cluster, host_rank, image_blob
            FROM gallery
            ORDER BY host_cluster, host_rank, id
        """).fetchall()

    csv_bytes = export_csv_bytes() if include_csv else None

    zbuf = BytesIO()
    with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        if include_csv and csv_bytes:
            z.writestr("gallery_metadata.csv", csv_bytes)

        for r in rows:
            rid = int(r["id"])
            team = safe_filename(r["team_name"])
            cluster = safe_filename(r["host_cluster"])
            fname = f"images/{cluster}/{rid:04d}_{team}.png"
            z.writestr(fname, r["image_blob"])

    zbuf.seek(0)
    return zbuf.getvalue()

# ============================================================
# UI HEADER
# ============================================================
st.title("Engineering loneliness with GenAI")
st.markdown(f"**Task:** {TASK_BYLINE}")

# ============================================================
# HOST MODE (hidden unless ?host=1)
# ============================================================
if HOST_FLAG:
    # component import only needed for host curation view
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
        if st.button("Refresh"):
            st.rerun()

    if not is_host:
        st.warning("Host mode is enabled via the URL. Enter the admin password in the sidebar.")
        st.stop()

    # ---- Build containers from DB meta
    meta = get_gallery_meta()
    by_cluster = {b: [] for b in buckets}
    for m in meta:
        label = f"{m['id']} | {m['team_name']}".strip()
        cluster = m["host_cluster"] if m["host_cluster"] in by_cluster else "Unsorted"
        by_cluster[cluster].append((m["host_rank"], m["id"], label))

    containers = []
    for b in buckets:
        items = [lbl for _, _, lbl in sorted(by_cluster[b], key=lambda x: (x[0], x[1]))]
        containers.append({"header": b, "items": items})

    # For image rendering
    subs_full = get_submissions_full(order_by="curated")
    subs_by_id = {s["id"]: s for s in subs_full}

    def render_bucket_images(layout_containers):
        for c in layout_containers:
            st.markdown(f"### {c['header']}")
            ids = []
            for label in c["items"]:
                try:
                    ids.append(int(str(label).split("|", 1)[0].strip()))
                except Exception:
                    pass

            imgs = [subs_by_id[i] for i in ids if i in subs_by_id]
            if not imgs:
                st.caption("—")
                continue

            cols = st.columns(n_cols)
            for i, s in enumerate(imgs):
                with cols[i % n_cols]:
                    st.image(s["image"], use_container_width=True)
                    if compact:
                        st.caption(f"**{s.get('team_name','')}**")
                    else:
                        with st.expander(f"{s.get('team_name','')} • #{s.get('id','')} • {s.get('created_at','')}"):
                            st.write(s.get("prompt", ""))

    # ---- Views
    if view == "Curate (drag & drop)":
        st.subheader("Drag & drop to cluster and reorder")

        new_containers = sort_items(containers, multi_containers=True)

        if normalize_layout(new_containers) != normalize_layout(containers):
            update_host_layout(new_containers)
            # Update local view so preview matches immediately
            containers = new_containers

        st.divider()
        st.subheader("Preview (images follow your bucket order)")
        render_bucket_images(containers)

        st.stop()

    if view == "Gallery wall":
        # Show current curated view
        render_bucket_images(containers)
        st.stop()

    if view == "Download":
        st.subheader("Download gallery data")

        csv_bytes = export_csv_bytes()
        st.download_button(
            "Download CSV (prompts + metadata)",
            data=csv_bytes,
            file_name="gallery_metadata.csv",
            mime="text/csv"
        )

        zip_bytes = export_zip_bytes(include_csv=True)
        st.download_button(
            "Download ZIP (all images + CSV)",
            data=zip_bytes,
            file_name="gallery_images_and_metadata.zip",
            mime="application/zip"
        )

        st.info("ZIP structure: images/<bucket>/<id>_<team>.png plus gallery_metadata.csv at the top level.")
        st.stop()

# ============================================================
# PARTICIPANT MODE (default)
# ============================================================
with st.sidebar:
    st.header("Your info")

    if "team_name" not in st.session_state:
        st.session_state["team_name"] = ""

    if st.session_state["team_name"]:
        st.success(f"Team: {st.session_state['team_name']}")
        if st.button("Change team name"):
            st.session_state["team_name"] = ""
            st.session_state.pop("draft_bytes", None)
            st.session_state.pop("gen_future", None)
            st.session_state.pop("gen_error", None)
            st.rerun()

# Gate: enter team name first
if not st.session_state["team_name"]:
    team = st.text_input("Enter your team name to begin", placeholder="e.g., Team Blue")
    if st.button("Continue"):
        if not team.strip():
            st.warning("Please enter a team name.")
        else:
            st.session_state["team_name"] = team.strip()
            st.rerun()
    st.stop()

team_name = st.session_state["team_name"]

# Secrets check
if "google_api" not in st.secrets or "key" not in st.secrets["google_api"]:
    st.error("Missing Google API key. Set st.secrets['google_api']['key'] in your Streamlit secrets.")
    st.stop()

api_key = st.secrets["google_api"]["key"]

# One executor per session
if "executor" not in st.session_state:
    st.session_state["executor"] = concurrent.futures.ThreadPoolExecutor(max_workers=1)

st.caption(f"Submitting as: **{team_name}**")

st.markdown(
    "**Prompt tip:** Start with *“Photorealistic documentary photograph…”* and specify a real moment, "
    "natural light, and a candid, everyday feel."
)

prompt = st.text_area("Prompt", height=180, placeholder="Photorealistic documentary photograph of...")

# Generate
if st.button("Generate image", key="gen_btn"):
    if not prompt.strip():
        st.warning("Please write a prompt.")
    else:
        fut = st.session_state.get("gen_future")
        if fut is not None and not fut.done():
            st.warning("Already generating…")
        else:
            st.session_state["job_started"] = time.time()
            st.session_state["job_prompt"] = prompt
            st.session_state.pop("draft_bytes", None)
            st.session_state.pop("gen_error", None)

            st.session_state["gen_future"] = st.session_state["executor"].submit(
                _generate_image_bytes, prompt, api_key
            )

# Poll generation
fut = st.session_state.get("gen_future")
if fut is not None:
    if fut.done():
        try:
            data = fut.result()
            st.session_state["draft_bytes"] = data
        except Exception as e:
            st.session_state["gen_error"] = str(e)
        finally:
            st.session_state.pop("gen_future", None)
    else:
        elapsed = time.time() - st.session_state.get("job_started", time.time())
        st.info(f"Generating… ({elapsed:.1f}s)")
        time.sleep(0.4)
        st.rerun()

# Error
if st.session_state.get("gen_error"):
    st.error(f"Generation failed: {st.session_state['gen_error']}")

# Preview + submit
if "draft_bytes" in st.session_state:
    img = Image.open(BytesIO(st.session_state["draft_bytes"]))
    st.image(img, use_container_width=True)

    if st.button("Submit to gallery", key="submit_btn"):
        used_prompt = st.session_state.get("job_prompt", prompt) or prompt
        save_submission(team_name or "Anonymous", used_prompt, img)
        st.session_state.pop("draft_bytes", None)
        st.success("Saved to gallery.")
        st.rerun()


