import time
import threading
import concurrent.futures
from io import BytesIO
import sqlite3

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

# Limit concurrent Gemini calls across ALL users on this Streamlit instance.
MAX_CONCURRENT_GEN = 8

TASK_BYLINE = (
    "Create a few photorealistic, everyday student-life scenes showing a student who might be lonely. "
    "Try to inject context (e.g., home, study location, otherwise), and try to inject the student's experience "
    "(socially, emotionally, or otherwise). Try to vary the situations as much as possible."
)

@st.cache_resource
def global_gen_semaphore():
    return threading.BoundedSemaphore(MAX_CONCURRENT_GEN)

# ============================================================
# QUERY PARAM HELPERS (host mode via ?host=1)
# ============================================================
def qp_get(name: str, default: str = "") -> str:
    try:
        v = st.query_params.get(name, default)
    except Exception:
        # fallback for older Streamlit
        v = st.experimental_get_query_params().get(name, [default])
    if isinstance(v, (list, tuple)):
        return str(v[0]) if v else default
    return str(v)

HOST_FLAG = qp_get("host", "0").strip().lower() in ("1", "true", "yes")

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
            "INSERT INTO gallery (team_name, prompt, image_blob) VALUES (?, ?, ?)",
            (team, prompt, image_to_blob(img))
        )
        conn.commit()

def get_submissions():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM gallery ORDER BY id DESC").fetchall()

    out = []
    for r in rows:
        d = dict(r)
        d["image"] = blob_to_image(d["image_blob"])
        out.append(d)
    return out

# ============================================================
# HEADER
# ============================================================
st.title("Engineering loneliness with GenAI")
st.markdown(f"**Task:** {TASK_BYLINE}")

# ============================================================
# HOST VIEW (only visible via ?host=1)
# ============================================================
if HOST_FLAG:
    with st.sidebar:
        st.header("Host login")
        pw = st.text_input("Admin password", type="password")
        is_host = (pw == HOST_PASSWORD)

        cols = st.columns(2)
        with cols[0]:
            if st.button("Refresh"):
                st.rerun()
        with cols[1]:
            compact = st.toggle("Compact captions", value=True)

        n_cols = st.slider("Gallery columns", 2, 6, 4)

    if not is_host:
        st.warning("Host mode is enabled via the URL. Enter the admin password in the sidebar.")
        st.stop()

    subs = get_submissions()
    if not subs:
        st.info("No images yet.")
        st.stop()

    grid = st.columns(n_cols)
    for i, s in enumerate(subs):
        with grid[i % n_cols]:
            st.image(s["image"], use_container_width=True)

            if compact:
                st.caption(f"**{s['team_name']}**")
            else:
                with st.expander(f"{s['team_name']} • #{s['id']} • {s.get('created_at','')}"):
                    st.write(s["prompt"])

    st.stop()

# ============================================================
# PARTICIPANT FLOW (default)
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

# Step 1: team name gate
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

# Step 2: prompt + generate + submit
if "google_api" not in st.secrets or "key" not in st.secrets["google_api"]:
    st.error("Missing Google API key. Set st.secrets['google_api']['key'] in your Streamlit secrets.")
    st.stop()

api_key = st.secrets["google_api"]["key"]

if "executor" not in st.session_state:
    st.session_state["executor"] = concurrent.futures.ThreadPoolExecutor(max_workers=1)

st.caption(f"Submitting as: **{team_name}**")

st.markdown(
    "**Prompt tip:** Start with *“Photorealistic documentary photograph…”* and specify a real moment, "
    "natural light, and a candid, everyday feel."
)

prompt = st.text_area("Prompt", height=180, placeholder="Photorealistic documentary photograph of...")

# Start generation
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

# Show any error
if st.session_state.get("gen_error"):
    st.error(f"Generation failed: {st.session_state['gen_error']}")

# Show generated image + submit
if "draft_bytes" in st.session_state:
    img = Image.open(BytesIO(st.session_state["draft_bytes"]))
    st.image(img, use_container_width=True)

    if st.button("Submit to gallery", key="submit_btn"):
        used_prompt = st.session_state.get("job_prompt", prompt) or prompt
        save_submission(team_name or "Anonymous", used_prompt, img)
        st.session_state.pop("draft_bytes", None)
        st.success("Saved to gallery.")
        st.rerun()

