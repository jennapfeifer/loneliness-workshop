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

        # NOTE: New schema: valence + intensity only
        c.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            participant_id TEXT NOT NULL,
            valence INTEGER NOT NULL,
            intensity INTEGER NOT NULL,
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

def submit_vote(image_id: int, pid: str, valence: int, intensity: int):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
        INSERT INTO votes (image_id, participant_id, valence, intensity)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(image_id, participant_id) DO UPDATE SET
            valence=excluded.valence,
            intensity=excluded.intensity,
            created_at=datetime('now')
        """, (image_id, pid, valence, intensity))
        conn.commit()

def vote_stats(image_id: int):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
        SELECT COUNT(*) n,
               AVG(valence) avg_valence,
               AVG(intensity) avg_intensity
        FROM votes WHERE image_id=?
        """, (image_id,))
        r = c.fetchone()

    if not r or r["n"] == 0:
        return dict(n=0, avg_valence=0.0, avg_intensity=0.0)
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
# CLUSTERING
# ============================================================
def add_quadrants(df: pd.DataFrame):
    val_med = df["X"].median()
    int_med = df["Y"].median()

    def q(r):
        if r["X"] >= val_med and r["Y"] >= int_med:
            return "High valence × High intensity"
        if r["X"] >= val_med and r["Y"] < int_med:
            return "High valence × Low intensity"
        if r["X"] < val_med and r["Y"] >= int_med:
            return "Low valence × High intensity"
        return "Low valence × Low intensity"

    df["Cluster"] = df.apply(q, axis=1)
    return df, float(val_med), float(int_med)

# ============================================================
# PLOTTING (Altair with bounded axes 1–5)
# ============================================================
def bounded_scatter(df: pd.DataFrame, color_col: str, title: str):
    df = df.copy()
    df["X"] = df["X"].clip(1, 5)
    df["Y"] = df["Y"].clip(1, 5)

    tooltip_cols = []
    for col in ["team_name", "prompt", "n", "avg_valence", "avg_intensity", "X", "Y"]:
        if col in df.columns:
            tooltip_cols.append(col)

    chart = (
        alt.Chart(df)
        .mark_circle(size=160)
        .encode(
            x=alt.X("X:Q", scale=alt.Scale(domain=[1, 5]), title="Valence (1–5)"),
            y=alt.Y("Y:Q", scale=alt.Scale(domain=[1, 5]), title="Emotional intensity (1–5)"),
            color=alt.Color(f"{color_col}:N", legend=alt.Legend(title=title)),
            tooltip=tooltip_cols
        )
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
st.caption(
    "Write a prompt for a photorealistic, documentary-style photograph. "
)

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
        "Specify a realistic moment, natural light, and a real-camera feel. "
    )

    prompt = st.text_area(
        "Prompt",
        height=180,
        placeholder=(
            "Photorealistic documentary photograph. 3:2 aspect ratio ... "
            "Natural lighting. "
        )
    )

    if st.button("Generate image"):
        if not prompt.strip():
            st.warning("Please write a prompt.")
        else:
            with st.spinner("Generating…"):
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
                    st.error("No image returned. Try again with a simpler, more literal prompt.")
                else:
                    img = Image.open(BytesIO(img_part.inline_data.data))
                    st.session_state["draft"] = img
                    st.image(img, use_container_width=True)

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
            st.caption(s["prompt"])

            stats = vote_stats(s["id"])
            if stats["n"] > 0:
                st.caption(f"Ratings so far: {int(stats['n'])}")

            with st.form(f"vote_{s['id']}"):
                valence = st.slider("Valence (negative → positive)", 1, 5, 3, key=f"val_{s['id']}")
                intensity = st.slider("Emotional intensity (calm → intense)", 1, 5, 3, key=f"int_{s['id']}")

                submitted = st.form_submit_button("Submit", disabled=voted)
                if submitted:
                    submit_vote(s["id"], participant_id, valence, intensity)
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
            st.caption(s["prompt"])
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
                "X": float(stats["avg_valence"]),
                "Y": float(stats["avg_intensity"])
            })

    if len(rows) < 2:
        st.warning("Not enough rated images yet.")
        st.stop()

    df = pd.DataFrame(rows)
    df, val_split, int_split = add_quadrants(df)

    bounded_scatter(df, color_col="Cluster", title="Quadrant cluster")
    st.caption(f"Valence split = {val_split:.2f} | Intensity split = {int_split:.2f}")

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
                "X": float(stats["avg_valence"]),
                "Y": float(stats["avg_intensity"])
            })

    if len(rows) < KMEANS_MIN_IMAGES:
        st.warning("At least 12 rated images are required for K-means.")
        st.stop()

    df = pd.DataFrame(rows)
    km = KMeans(n_clusters=KMEANS_K, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(df[["X", "Y"]]).astype(str)

    bounded_scatter(df, color_col="Cluster", title="KMeans cluster")

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


