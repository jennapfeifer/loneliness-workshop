import time
import uuid
import threading
import concurrent.futures
from dataclasses import dataclass

import streamlit as st
import altair as alt

from google import genai
from google.genai import types

from io import BytesIO, StringIO
from PIL import Image

import sqlite3
import pandas as pd
from sklearn.cluster import KMeans

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Engineering loneliness with GenAI",
    layout="wide"
)

DB_PATH = "workshop.db"
HOST_PASSWORD = "admin123"

MIN_RATINGS_FOR_MAP = 1
KMEANS_K = 4
KMEANS_MIN_IMAGES = 12

# ============================================================
# DATABASE
# ============================================================
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def _table_exists(conn, name: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)
    ).fetchone()
    return r is not None

def _columns(conn, table: str) -> list[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]

def _find_free_backup_name(conn, base: str = "votes_old") -> str:
    name = base
    i = 1
    while _table_exists(conn, name):
        name = f"{base}_{i}"
        i += 1
    return name

def init_db():
    with get_conn() as conn:
        c = conn.cursor()

        c.execute("""
        CREATE TABLE IF NOT EXISTS gallery (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT,
            prompt TEXT,
            image_blob BLOB
        )
        """)

        # ---- MIGRATION: if an old "votes" table exists with valence/intensity,
        # rename it so we can create the new schema safely.
        desired_cols = {
            "id", "image_id", "participant_id",
            "loneliness", "recognizable", "relatable",
            "created_at"
        }

        if _table_exists(conn, "votes"):
            cols = set(_columns(conn, "votes"))
            if cols != desired_cols:
                backup = _find_free_backup_name(conn, "votes_old")
                conn.execute(f"ALTER TABLE votes RENAME TO {backup};")

        # New schema: loneliness + recognizability + relatability (all 1–5)
        c.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            participant_id TEXT NOT NULL,
            loneliness INTEGER NOT NULL,
            recognizable INTEGER NOT NULL,
            relatable INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(image_id, participant_id)
        )
        """)
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

@dataclass
class Job:
    future: concurrent.futures.Future
    submitted_at: float

@st.cache_resource
def shared_runtime():
    # One shared pool for the whole Streamlit process (all users)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # tune: 2–6
    sem = threading.BoundedSemaphore(4)  # max concurrent Gemini calls (same as max_workers usually)
    lock = threading.Lock()
    jobs: dict[str, Job] = {}
    return executor, sem, lock, jobs

@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=st.secrets["google_api"]["key"])

def _generate_image_bytes(prompt: str) -> bytes:
    client = get_gemini_client()
    executor, sem, lock, jobs = shared_runtime()

    sem.acquire()
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="3:2"),
            ),
        )
        img_part = next((p for p in response.parts if getattr(p, "inline_data", None)), None)
        if img_part is None:
            raise RuntimeError("No image returned (response had no inline_data).")
        return img_part.inline_data.data
    finally:
        sem.release()

def _cleanup_old_jobs(max_age_s: int = 600):
    """Prevent memory leaks if someone closes the tab mid-job."""
    executor, sem, lock, jobs = shared_runtime()
    now = time.time()
    with lock:
        for jid, job in list(jobs.items()):
            if job.future.done() or (now - job.submitted_at) > max_age_s:
                jobs.pop(jid, None)


# ============================================================
# DB OPERATIONS
# ============================================================
def save_submission(team: str, prompt: str, img: Image.Image):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO gallery (team_name, prompt, image_blob) VALUES (?, ?, ?)",
            (team, prompt, image_to_blob(img))
        )
        conn.commit()

def get_submissions():
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM gallery ORDER BY id DESC")
        rows = c.fetchall()

    out = []
    for r in rows:
        d = dict(r)
        d["image"] = blob_to_image(d["image_blob"])
        out.append(d)
    return out

def submit_vote(image_id: int, pid: str, loneliness: int, recognizable: int, relatable: int):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
        INSERT INTO votes (image_id, participant_id, loneliness, recognizable, relatable)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(image_id, participant_id) DO UPDATE SET
            loneliness=excluded.loneliness,
            recognizable=excluded.recognizable,
            relatable=excluded.relatable,
            created_at=datetime('now')
        """, (image_id, pid, loneliness, recognizable, relatable))
        conn.commit()

def vote_stats(image_id: int):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
        SELECT COUNT(*) n,
               AVG(loneliness)   avg_loneliness,
               AVG(recognizable) avg_recognizable,
               AVG(relatable)    avg_relatable
        FROM votes WHERE image_id=?
        """, (image_id,))
        r = c.fetchone()

    if not r or r["n"] == 0:
        return dict(n=0, avg_loneliness=0.0, avg_recognizable=0.0, avg_relatable=0.0)
    return dict(r)

def has_voted(image_id: int, pid: str) -> bool:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT 1 FROM votes WHERE image_id=? AND participant_id=?",
            (image_id, pid)
        )
        return c.fetchone() is not None

# ============================================================
# CLUSTERING / QUADRANTS
# ============================================================
def add_quadrants(df: pd.DataFrame):
    lon_med = df["X"].median()
    rec_med = df["Y"].median()

    def q(r):
        if r["X"] >= lon_med and r["Y"] >= rec_med:
            return "High loneliness × High recognizability"
        if r["X"] >= lon_med and r["Y"] < rec_med:
            return "High loneliness × Low recognizability"
        if r["X"] < lon_med and r["Y"] >= rec_med:
            return "Low loneliness × High recognizability"
        return "Low loneliness × Low recognizability"

    df["Cluster"] = df.apply(q, axis=1)
    return df, float(lon_med), float(rec_med)

# ============================================================
# PLOTTING (Altair with bounded axes 1–5)
# - X: Avg loneliness present
# - Y: Avg recognizability
# - Size: Avg relatability (optional)
# ============================================================
def bounded_scatter(df: pd.DataFrame, color_col: str, title: str, size_col: str | None = None):
    df = df.copy()
    df["X"] = df["X"].clip(1, 5)
    df["Y"] = df["Y"].clip(1, 5)

    tooltip_cols = []
    for col in [
        "team_name", "prompt", "n",
        "avg_loneliness", "avg_recognizable", "avg_relatable",
        "X", "Y"
    ]:
        if col in df.columns:
            tooltip_cols.append(col)

    enc = {
        "x": alt.X("X:Q", scale=alt.Scale(domain=[1, 5]), title="Loneliness present (avg, 1–5)"),
        "y": alt.Y("Y:Q", scale=alt.Scale(domain=[1, 5]), title="Recognizable student-life loneliness (avg, 1–5)"),
        "color": alt.Color(f"{color_col}:N", legend=alt.Legend(title=title)),
        "tooltip": tooltip_cols
    }

    if size_col and size_col in df.columns:
        enc["size"] = alt.Size(
            f"{size_col}:Q",
            legend=alt.Legend(title="Relatability (avg)"),
            scale=alt.Scale(domain=[1, 5])
        )

    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.9)
        .encode(**enc)
        .properties(height=520)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Setup")
    participant_id = st.text_input("Participant ID", placeholder="P01")
    team_name = st.text_input("Team Name", placeholder="Anonymous")

    role = st.radio("Role", ["Participant", "Host"])
    is_host = False
    if role == "Host":
        if st.text_input("Admin password", type="password") == HOST_PASSWORD:
            is_host = True

    if is_host:
        mode = st.radio("View", ["Gallery", "Map (Quadrants)", "Map (KMeans)", "Download"])
    else:
        mode = st.radio("Step", ["Create", "Rate"])

# ============================================================
# HEADER
# ============================================================
st.title("Engineering loneliness with GenAI")

# Only show the subtitle when participants are creating prompts/images
if mode == "Create" and not is_host:
    st.caption("Write a prompt for a photorealistic, documentary-style photograph.")

# ============================================================
# CREATE
# ============================================================
if mode == "Create" and not is_host:
    if "google_api" not in st.secrets or "key" not in st.secrets["google_api"]:
        st.error("Missing Google API key. Set st.secrets['google_api']['key'] in your Streamlit secrets.")
        st.stop()

    client = genai.Client(api_key=st.secrets["google_api"]["key"])

    st.markdown(
        "**Prompt tips:** Start with *“Photorealistic documentary photograph…”*. "
        "Specify a realistic moment, natural light, and a real-camera feel."
    )

    prompt = st.text_area(
        "Prompt",
        height=180,
        placeholder=(
            "Photorealistic documentary photograph. 3:2 aspect ratio ... "
            "Natural lighting. "
        )
    )
_cleanup_old_jobs()

# Show some queue info (optional but nice in workshops)
executor, sem, lock, jobs = shared_runtime()
with lock:
    pending = sum(1 for j in jobs.values() if not j.future.done())
st.caption(f"Generator queue: {pending} pending")

if st.button("Generate image"):
    if not prompt.strip():
        st.warning("Please write a prompt.")
    else:
        job_id = str(uuid.uuid4())
        fut = executor.submit(_generate_image_bytes, prompt)

        with lock:
            jobs[job_id] = Job(future=fut, submitted_at=time.time())

        st.session_state["job_id"] = job_id
        st.session_state.pop("draft", None)
        st.toast("Queued! You can keep editing while it generates.")

job_id = st.session_state.get("job_id")
if job_id:
    with lock:
        job = jobs.get(job_id)

    if job is None:
        st.session_state.pop("job_id", None)
    elif job.future.done():
        try:
            data = job.future.result()
            img = Image.open(BytesIO(data))
            st.session_state["draft"] = img
            st.image(img, use_container_width=True)
        except Exception as e:
            st.error(f"Generation failed: {e}")
        finally:
            with lock:
                jobs.pop(job_id, None)
            st.session_state.pop("job_id", None)
    else:
        st.info("Generating in the background…")
        if st.button("Refresh status"):
            st.rerun()

    if "draft" in st.session_state:
        if st.button("Submit image"):
            save_submission(team_name or "Anonymous", prompt, st.session_state["draft"])
            del st.session_state["draft"]
            st.success("Saved.")
            st.rerun()

# ============================================================
# RATE
# ============================================================
elif mode == "Rate" and not is_host:
    if not participant_id:
        st.warning("Enter a Participant ID to rate images.")
        st.stop()

    subs = get_submissions()
    if not subs:
        st.info("No images yet.")
        st.stop()

    cols = st.columns(3)

    for i, s in enumerate(subs):
        voted = has_voted(s["id"], participant_id)

        with cols[i % 3]:
            st.image(s["image"], use_container_width=True)

            stats = vote_stats(s["id"])
            if stats["n"] > 0:
                st.caption(f"Ratings so far: {int(stats['n'])}")

            with st.form(f"vote_{s['id']}"):
                loneliness = st.slider(
                    "How much loneliness is present in this image?",
                    1, 5, 3, key=f"lon_{s['id']}"
                )
                recognizable = st.slider(
                    "How recognizable is this as student-life loneliness?",
                    1, 5, 3, key=f"rec_{s['id']}"
                )
                relatable = st.slider(
                    "How much do you relate to this image?",
                    1, 5, 3, key=f"rel_{s['id']}"
                )

                submitted = st.form_submit_button("Submit", disabled=voted)
                if submitted:
                    submit_vote(s["id"], participant_id, loneliness, recognizable, relatable)
                    st.rerun()

            if voted:
                st.caption("✅ You already rated this image.")

# ============================================================
# GALLERY (HOST)
# ============================================================
elif mode == "Gallery" and is_host:
    subs = get_submissions()
    if not subs:
        st.info("No images yet.")
        st.stop()

    cols = st.columns(3)
    for i, s in enumerate(subs):
        stats = vote_stats(s["id"])
        with cols[i % 3]:
            st.image(s["image"], use_container_width=True)
            st.caption(f"Ratings: {int(stats['n'])}")


# ============================================================
# MAP: QUADRANTS
# - X = avg_loneliness
# - Y = avg_recognizable
# - point size = avg_relatable
# ============================================================
elif mode == "Map (Quadrants)" and is_host:
    subs = get_submissions()
    rows = []

    for s in subs:
        stats = vote_stats(s["id"])
        if stats["n"] >= MIN_RATINGS_FOR_MAP:
            rows.append({
                "id": s["id"],
                "team_name": s["team_name"],
                "prompt": s["prompt"],
                **stats,
                "X": float(stats["avg_loneliness"]),
                "Y": float(stats["avg_recognizable"]),
            })

    if len(rows) < 2:
        st.warning("Not enough rated images yet.")
        st.stop()

    df = pd.DataFrame(rows)
    df, lon_split, rec_split = add_quadrants(df)

    bounded_scatter(df, color_col="Cluster", title="Quadrant cluster", size_col="avg_relatable")
    st.caption(f"Loneliness split = {lon_split:.2f} | Recognizability split = {rec_split:.2f}")

# ============================================================
# MAP: KMEANS
# - Cluster on 3D: (avg_loneliness, avg_recognizable, avg_relatable)
# - Plot 2D: loneliness vs recognizability; size shows relatability
# ============================================================
elif mode == "Map (KMeans)" and is_host:
    subs = get_submissions()
    rows = []

    for s in subs:
        stats = vote_stats(s["id"])
        if stats["n"] >= MIN_RATINGS_FOR_MAP:
            rows.append({
                "id": s["id"],
                "team_name": s["team_name"],
                "prompt": s["prompt"],
                **stats,
                "X": float(stats["avg_loneliness"]),
                "Y": float(stats["avg_recognizable"]),
                "Z": float(stats["avg_relatable"]),
            })

    if len(rows) < KMEANS_MIN_IMAGES:
        st.warning("At least 12 rated images are required for K-means.")
        st.stop()

    df = pd.DataFrame(rows)
    km = KMeans(n_clusters=KMEANS_K, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(df[["X", "Y", "Z"]]).astype(str)

    bounded_scatter(df, color_col="Cluster", title="KMeans cluster", size_col="avg_relatable")

# ============================================================
# DOWNLOAD
# ============================================================
elif mode == "Download" and is_host:
    subs = get_submissions()
    rows = []

    for s in subs:
        stats = vote_stats(s["id"])
        rows.append({
            "id": s["id"],
            "team_name": s["team_name"],
            "prompt": s["prompt"],
            **stats
        })

    df = pd.DataFrame(rows)
    buf = StringIO()
    df.to_csv(buf, index=False)

    st.download_button(
        "Download CSV",
        buf.getvalue(),
        "loneliness_workshop.csv",
        "text/csv"
    )

else:
    st.info("Select a mode from the sidebar.")
