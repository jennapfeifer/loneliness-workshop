import streamlit as st
from google import genai
from google.genai import types

from io import BytesIO, StringIO
from PIL import Image

import time
import sqlite3
import csv
import pandas as pd
import numpy as np
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

        c.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            participant_id TEXT NOT NULL,
            iso INTEGER NOT NULL,
            inte INTEGER NOT NULL,
            pain INTEGER NOT NULL,
            rel INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(image_id, participant_id)
        )
        """)
        conn.commit()

init_db()

# ============================================================
# IMAGE HELPERS
# ============================================================
def image_to_blob(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def blob_to_image(blob):
    return Image.open(BytesIO(blob))

# ============================================================
# DB OPERATIONS
# ============================================================
def save_submission(team, prompt, img):
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

def submit_vote(image_id, pid, iso, inte, pain, rel):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
        INSERT INTO votes (image_id, participant_id, iso, inte, pain, rel)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(image_id, participant_id) DO UPDATE SET
            iso=excluded.iso,
            inte=excluded.inte,
            pain=excluded.pain,
            rel=excluded.rel,
            created_at=datetime('now')
        """, (image_id, pid, iso, inte, pain, rel))
        conn.commit()

def vote_stats(image_id):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
        SELECT COUNT(*) n,
               AVG(iso) avg_iso,
               AVG(inte) avg_int,
               AVG(pain) avg_pain,
               AVG(rel) avg_rel
        FROM votes WHERE image_id=?
        """, (image_id,))
        r = c.fetchone()

    if r["n"] == 0:
        return dict(n=0, avg_iso=0, avg_int=0, avg_pain=0, avg_rel=0)
    return dict(r)

def has_voted(image_id, pid):
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
def add_quadrants(df):
    iso_med = df["X"].median()
    int_med = df["Y"].median()

    def q(r):
        if r["X"] >= iso_med and r["Y"] >= int_med:
            return "High isolation × High intensity"
        if r["X"] >= iso_med and r["Y"] < int_med:
            return "High isolation × Low intensity"
        if r["X"] < iso_med and r["Y"] >= int_med:
            return "Low isolation × High intensity"
        return "Low isolation × Low intensity"

    df["Cluster"] = df.apply(q, axis=1)
    return df, iso_med, int_med

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
    "Describe a moment of loneliness and encourage photorealistic, documentary-style prompting."
)

# ============================================================
# CREATE
# ============================================================
if mode == "Create" and not is_host:
    client = genai.Client(api_key=st.secrets["google_api"]["key"])

    prompt = st.text_area(
        "Prompt",
        height=160,
        placeholder="Photorealistic documentary-style photograph..."
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
                img_part = next(p for p in response.parts if p.inline_data)
                img = Image.open(BytesIO(img_part.inline_data.data))
                st.session_state["draft"] = img
                st.image(img)

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
    cols = st.columns(3)

    for i, s in enumerate(subs):
        stats = vote_stats(s["id"])
        voted = has_voted(s["id"], participant_id)

        with cols[i % 3]:
            st.image(s["image"], use_container_width=True)
            st.caption(s["prompt"])

            with st.form(f"vote_{s['id']}"):
                iso = st.slider("Isolation", 1, 5, 3)
                inte = st.slider("Intensity", 1, 5, 3)
                pain = st.slider("Pain", 1, 5, 3)
                rel = st.slider("Relatability", 1, 5, 3)
                submitted = st.form_submit_button(
                    "Submit",
                    disabled=voted
                )
                if submitted:
                    submit_vote(s["id"], participant_id, iso, inte, pain, rel)
                    st.rerun()

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
                **s,
                **stats,
                "X": stats["avg_iso"],
                "Y": stats["avg_int"]
            })

    if len(rows) < 2:
        st.warning("Not enough rated images yet.")
        st.stop()

    df = pd.DataFrame(rows)
    df, iso_split, int_split = add_quadrants(df)

    st.scatter_chart(
        df,
        x="X",
        y="Y",
        color="Cluster",
        size=100,
        x_range=(1, 5),
        y_range=(1, 5),
        use_container_width=True
    )

    st.caption(
        f"Isolation split = {iso_split:.2f} | Intensity split = {int_split:.2f}"
    )

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
                **s,
                "X": stats["avg_iso"],
                "Y": stats["avg_int"]
            })

    if len(rows) < KMEANS_MIN_IMAGES:
        st.warning("At least 12 images are required for K-means.")
        st.stop()

    df = pd.DataFrame(rows)
    km = KMeans(n_clusters=KMEANS_K, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(df[["X", "Y"]])

    st.scatter_chart(
        df,
        x="X",
        y="Y",
        color="Cluster",
        size=100,
        x_range=(1, 5),
        y_range=(1, 5),
        use_container_width=True
    )

# ============================================================
# DOWNLOAD
# ============================================================
elif mode == "Download" and is_host:
    subs = get_submissions()
    rows = []

    for s in subs:
        stats = vote_stats(s["id"])
        rows.append({**s, **stats})

    df = pd.DataFrame(rows)
    buf = StringIO()
    df.drop(columns=["image_blob", "image"]).to_csv(buf, index=False)

    st.download_button(
        "Download CSV",
        buf.getvalue(),
        "loneliness_workshop.csv",
        "text/csv"
    )
