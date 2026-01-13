import time
import uuid
import threading
import concurrent.futures
from dataclasses import dataclass  # (not used now, but fine to keep)

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

# Limit concurrent Gemini calls across ALL users on this Streamlit instance.
# Tune this based on your quota + server size.
MAX_CONCURRENT_GEN = 8

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

# ============================================================
# GEMINI HELPERS (thread-safe-ish)
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

        # Defensive: make sure we got bytes
        if isinstance(data, str):
            # Some SDKs might return base64 as str; most return bytes.
            # If this happens, fail loudly so you see the real issue.
            raise RuntimeError("Gemini returned image data as str (expected bytes). Check SDK version/response parsing.")

        return data
    finally:
        sem.release()

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
# PLOTTING
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

if mode == "Create" and not is_host:
    st.caption("Write a prompt for a photorealistic, documentary-style photograph.")

# ============================================================
# CREATE
# ============================================================
if mode == "Create" and not is_host:
    if "google_api" not in st.secrets or "key" not in st.secrets["google_api"]:
        st.error("Missing Google API key. Set st.secrets['google_api']['key'] in your Streamlit secrets.")
        st.stop()

    api_key = st.secrets["google_api"]["key"]

    # One executor per SESSION (this is the reliable part)
    if "executor" not in st.session_state:
        st.session_state["executor"] = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    st.markdown(
        "**Prompt tips:** Start with *“Photorealistic documentary photograph…”*. "
        "Specify a realistic moment, natural light, and a real-camera feel."
    )

    prompt = st.text_area("Prompt", height=180)

    auto_refresh = st.checkbox("Auto-refresh while generating", value=True)

    # Start generation
    if st.button("Generate image", key="gen_btn"):
        if not prompt.strip():
            st.warning("Please write a prompt.")
        else:
            # Prevent double-submits while one is running
            fut = st.session_state.get("gen_future")
            if fut is not None and not fut.done():
                st.warning("Already generating… please wait.")
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
            if auto_refresh:
                time.sleep(0.5)
                st.rerun()
            else:
                if st.button("Check status"):
                    st.rerun()

    # Show any error
    if st.session_state.get("gen_error"):
        st.error(f"Generation failed: {st.session_state['gen_error']}")

    # Always display the generated image if we have it
    if "draft_bytes" in st.session_state:
        img = Image.open(BytesIO(st.session_state["draft_bytes"]))
        st.image(img, use_container_width=True)

        if st.button("Submit image", key="submit_btn"):
            used_prompt = st.session_state.get("job_prompt", prompt) or prompt
            save_submission(team_name or "Anonymous", used_prompt, img)
            st.session_state.pop("draft_bytes", None)
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
