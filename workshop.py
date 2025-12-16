import streamlit as st
from google import genai
from google.genai import types

from io import BytesIO, StringIO
from PIL import Image

import json
import time
import sqlite3
import csv
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Loneliness Workshop: Subjective Profiles", layout="wide")

DB_PATH = "workshop.db"

# Put this in st.secrets in a real deployment
HOST_PASSWORD = "admin123"

# How many human ratings before an image uses human averages for XY
MIN_RATINGS_FOR_HUMAN = 3

# Default number of quadrants (always 4), optional KMeans k
KMEANS_K = 4

# ============================================================
# DB HELPERS (WAL + retry)
# ============================================================
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def with_db_retry(fn, retries=6, base_delay=0.08):
    for i in range(retries):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and i < retries - 1:
                time.sleep(base_delay * (2 ** i))
                continue
            raise

def init_db():
    with get_conn() as conn:
        c = conn.cursor()

        # Images
        c.execute("""
        CREATE TABLE IF NOT EXISTS gallery_v7 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT,
            prompt TEXT,
            image_blob BLOB,

            -- optional AI annotations
            ai_iso INTEGER DEFAULT 0,
            ai_int INTEGER DEFAULT 0,
            ai_pain INTEGER DEFAULT 0,
            ai_rel INTEGER DEFAULT 0,
            ai_reasoning TEXT DEFAULT ""
        )
        """)

        # Votes (one row per participant per image)
        c.execute("""
        CREATE TABLE IF NOT EXISTS votes_v1 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            participant_id TEXT NOT NULL,

            iso INTEGER NOT NULL,
            inte INTEGER NOT NULL,
            pain INTEGER NOT NULL,
            rel INTEGER NOT NULL,

            created_at TEXT DEFAULT (datetime('now')),

            UNIQUE(image_id, participant_id),
            FOREIGN KEY(image_id) REFERENCES gallery_v7(id)
        )
        """)

        conn.commit()

init_db()

# ============================================================
# IMAGE SERIALIZATION
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
def save_submission(team: str, prompt: str, img_obj: Image.Image) -> None:
    def _write():
        with get_conn() as conn:
            c = conn.cursor()
            img_blob = image_to_blob(img_obj)
            c.execute(
                "INSERT INTO gallery_v7 (team_name, prompt, image_blob) VALUES (?, ?, ?)",
                (team, prompt, img_blob),
            )
            conn.commit()
    with_db_retry(_write)

def get_all_submissions() -> list[dict]:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM gallery_v7 ORDER BY id DESC")
        rows = c.fetchall()

    subs = []
    for r in rows:
        d = dict(r)
        d["image"] = blob_to_image(d["image_blob"])
        subs.append(d)
    return subs

def reset_db():
    def _write():
        with get_conn() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM votes_v1")
            c.execute("DELETE FROM gallery_v7")
            conn.commit()
    with_db_retry(_write)

def submit_vote(image_id: int, participant_id: str, iso: int, inte: int, pain: int, rel: int) -> None:
    def _write():
        with get_conn() as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO votes_v1 (image_id, participant_id, iso, inte, pain, rel)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id, participant_id) DO UPDATE SET
                    iso=excluded.iso,
                    inte=excluded.inte,
                    pain=excluded.pain,
                    rel=excluded.rel,
                    created_at=datetime('now')
                """,
                (image_id, participant_id, iso, inte, pain, rel),
            )
            conn.commit()
    with_db_retry(_write)

def get_vote_stats(image_id: int) -> dict:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            SELECT
                COUNT(*) AS n,
                AVG(iso)  AS avg_iso,
                AVG(inte) AS avg_int,
                AVG(pain) AS avg_pain,
                AVG(rel)  AS avg_rel
            FROM votes_v1
            WHERE image_id=?
            """,
            (image_id,),
        )
        row = c.fetchone()

    if not row or row["n"] == 0:
        return {"n": 0, "avg_iso": 0.0, "avg_int": 0.0, "avg_pain": 0.0, "avg_rel": 0.0}
    return dict(row)

def has_participant_voted(image_id: int, participant_id: str) -> bool:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT 1 FROM votes_v1 WHERE image_id=? AND participant_id=? LIMIT 1",
            (image_id, participant_id),
        )
        return c.fetchone() is not None

def update_ai_results(img_id: int, analysis_json: dict) -> None:
    def _write():
        with get_conn() as conn:
            c = conn.cursor()
            c.execute(
                """
                UPDATE gallery_v7 SET
                    ai_iso=?,
                    ai_int=?,
                    ai_pain=?,
                    ai_rel=?,
                    ai_reasoning=?
                WHERE id=?
                """,
                (
                    int(analysis_json.get("isolation", 0)),
                    int(analysis_json.get("intensity", 0)),
                    int(analysis_json.get("pain", 0)),
                    int(analysis_json.get("relatability", 0)),
                    str(analysis_json.get("reasoning", "")),
                    img_id,
                ),
            )
            conn.commit()
    with_db_retry(_write)

def get_votes_df() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query("SELECT * FROM votes_v1", conn)

# ============================================================
# AI CLIENT + HELPERS
# ============================================================
def retry_api_call(func, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return func()
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            return None

def get_client():
    # Server-side only: prefer secrets
    api_key = None
    try:
        api_key = st.secrets["google_api"]["key"]
    except Exception:
        api_key = None

    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def generate_image(client, prompt: str):
    def _call():
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
            ),
        )
        img_part = next((p for p in response.parts if getattr(p, "inline_data", None)), None)
        if img_part:
            return Image.open(BytesIO(img_part.inline_data.data))
        return None
    return retry_api_call(_call)

def analyze_4_dimensions(client, image: Image.Image) -> dict:
    lit_prompt = """
Analyze this image on these 4 scales (1 to 5).

CORE (Clustering):
1. Social Isolation: 1=Connected/Crowded, 5=Total Isolation/Void
2. Emotional Intensity: 1=Calm/Subtle, 5=Overwhelming/Intense

CONTEXT:
3. Pain/Negativity: 1=Peaceful/Positive, 5=Painful/Tragic
4. Relatability: 1=Abstract/Alien, 5=Universal/Familiar

Return STRICTLY JSON:
{"isolation": int, "intensity": int, "pain": int, "relatability": int, "reasoning": "string"}
    """.strip()

    def _call():
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[image, lit_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text)

    result = retry_api_call(_call)
    if result:
        return result
    return {"isolation": 0, "intensity": 0, "pain": 0, "relatability": 0, "reasoning": "Failed."}

# ============================================================
# CLUSTERING (Human-first + Quadrants default, KMeans optional)
# ============================================================
def pick_xy(row: dict) -> tuple[float, float, str]:
    """
    Human-first once enough ratings exist; AI as fallback early.
    """
    n = row.get("n", 0)
    if n >= MIN_RATINGS_FOR_HUMAN:
        return float(row.get("avg_iso", 0.0)), float(row.get("avg_int", 0.0)), "human"
    if row.get("ai_iso", 0) > 0 and row.get("ai_int", 0) > 0:
        return float(row["ai_iso"]), float(row["ai_int"]), "ai"
    return float(row.get("avg_iso", 0.0)), float(row.get("avg_int", 0.0)), "human"

def add_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    iso_med = df["X"].median()
    int_med = df["Y"].median()

    def quadrant(r):
        hi_iso = r["X"] >= iso_med
        hi_int = r["Y"] >= int_med

        if hi_iso and hi_int:
            return "High isolation × High intensity"
        if hi_iso and not hi_int:
            return "High isolation × Low intensity"
        if not hi_iso and hi_int:
            return "Low isolation × High intensity"
        return "Low isolation × Low intensity"

    df["Cluster_Name"] = df.apply(quadrant, axis=1)
    df["Cluster_Method"] = "quadrants"
    df["iso_split"] = iso_med
    df["int_split"] = int_med
    return df

def add_kmeans(df: pd.DataFrame, k: int = KMEANS_K) -> pd.DataFrame:
    """
    Optional view. Only makes sense if you have enough images.
    """
    if len(df) < k * 3:
        df["KMeans_Cluster"] = -1
        return df

    X = df[["X", "Y"]].to_numpy()
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["KMeans_Cluster"] = km.fit_predict(X)
    return df

# ============================================================
# UI
# ============================================================
with st.sidebar:
    st.header("⚙️ Setup")

    # Participant identity (for vote uniqueness)
    participant_id = st.text_input("Participant ID", placeholder="e.g., P07")

    st.divider()
    team_name = st.text_input("Team Name", placeholder="Anonymous")

    st.divider()
    user_role = st.radio("Role:", ["Participant", "Host (Admin)"])

    is_host = False
    if user_role == "Host (Admin)":
        pw = st.text_input("Password", type="password")
        if pw == HOST_PASSWORD:
            is_host = True
            st.success("Admin unlocked.")
            if st.button("⚠️ RESET ALL DATA"):
                reset_db()
                st.rerun()

    st.divider()
    if is_host:
        app_mode = st.radio("Controls:", ["View Gallery", "Run Analysis", "The Map (Quadrants)", "The Map (KMeans)", "Download Data"])
    else:
        app_mode = st.radio("Steps:", ["1. Create", "2. Rate Images"])

client = get_client()

st.title("🧩 Loneliness: Subjective Profiles")
st.caption("Human-first clustering on Isolation × Intensity (AI is optional early fallback).")

# ============================================================
# PHASE 1: CREATE
# ============================================================
if app_mode == "1. Create" and not is_host:
    st.markdown("Generate an image that represents a **specific feeling of loneliness**.")

    if client is None:
        st.error("Server is missing the Google API key (st.secrets['google_api']['key']).")
        st.stop()

    if "current_draft" not in st.session_state:
        st.session_state["current_draft"] = None

    col1, col2 = st.columns([1, 1])

    with col1:
        with st.form("gen_form"):
            prompt = st.text_area("Prompt:", height=160, placeholder="Describe the scene, mood, setting, symbols, style...")
            submitted = st.form_submit_button("Generate Draft", type="primary")

            if submitted:
                if not prompt.strip():
                    st.warning("Please write a prompt first.")
                else:
                    with st.spinner("Dreaming..."):
                        img = generate_image(client, prompt.strip())
                        if img:
                            st.session_state["current_draft"] = {"image": img, "prompt": prompt.strip()}
                            st.rerun()
                        else:
                            st.error("Image generation failed. Try again.")

    with col2:
        if st.session_state["current_draft"]:
            st.image(st.session_state["current_draft"]["image"], caption="Draft", use_container_width=True)
            c1, c2 = st.columns(2)
            if c1.button("♻️ Scrap"):
                st.session_state["current_draft"] = None
                st.rerun()
            if c2.button("✅ Submit"):
                save_submission(team_name.strip() or "Anonymous", st.session_state["current_draft"]["prompt"], st.session_state["current_draft"]["image"])
                st.session_state["current_draft"] = None
                st.success("Saved!")
                time.sleep(0.6)
                st.rerun()

# ============================================================
# PHASE 2: RATING / VIEW GALLERY
# ============================================================
elif (app_mode == "2. Rate Images" and not is_host) or (app_mode == "View Gallery"):
    st.markdown("Rate images on **Core** (clustering) and **Optional** (context) dimensions.")

    subs = get_all_submissions()
    if not subs:
        st.warning("No images yet.")
        st.stop()

    if not is_host:
        if not participant_id.strip():
            st.warning("Enter a Participant ID in the sidebar to submit ratings.")
            st.stop()

    cols = st.columns(3)
    for idx, item in enumerate(subs):
        stats = get_vote_stats(item["id"])
        already_voted = participant_id.strip() and has_participant_voted(item["id"], participant_id.strip())

        with cols[idx % 3]:
            st.image(item["image"], use_container_width=True)
            st.caption(f"_{item['prompt']}_")

            # Show aggregates
            if stats["n"] > 0:
                st.caption(
                    f"Ratings: {stats['n']} | "
                    f"Iso {stats['avg_iso']:.2f} | Int {stats['avg_int']:.2f} | "
                    f"Pain {stats['avg_pain']:.2f} | Rel {stats['avg_rel']:.2f}"
                )
            else:
                st.caption("No human ratings yet.")

            if app_mode == "View Gallery" and is_host:
                # Admin view doesn't need rating form
                st.divider()
                continue

            with st.form(key=f"vote_{item['id']}"):
                st.markdown("**Core (for clustering)**")
                iso = st.slider("Social Isolation (1=Crowded, 5=Void)", 1, 5, 3, key=f"iso_{item['id']}")
                inte = st.slider("Emotional Intensity (1=Calm, 5=Overwhelming)", 1, 5, 3, key=f"inte_{item['id']}")

                with st.expander("Optional (context)"):
                    pain = st.slider("Pain/Negativity (1=Peaceful, 5=Painful)", 1, 5, 3, key=f"pain_{item['id']}")
                    rel = st.slider("Relatability (1=Alien, 5=Me)", 1, 5, 3, key=f"rel_{item['id']}")

                submitted = st.form_submit_button("Submit Ratings", disabled=already_voted)

                if submitted:
                    submit_vote(item["id"], participant_id.strip(), iso, inte, pain, rel)
                    st.success("Saved rating.")
                    time.sleep(0.3)
                    st.rerun()

            if already_voted:
                st.caption("✅ You already rated this image (re-rating is disabled).")

            st.divider()

# ============================================================
# PHASE 3: AI ANALYSIS (ADMIN)
# ============================================================
elif app_mode == "Run Analysis" and is_host:
    st.markdown("Run optional AI scoring (1–5) for all images. This does **not** control clustering once enough human ratings exist.")

    if client is None:
        st.error("Server is missing the Google API key (st.secrets['google_api']['key']).")
        st.stop()

    subs = get_all_submissions()
    if not subs:
        st.warning("No images yet.")
        st.stop()

    if st.button("🚀 Analyze All Images (AI 1–5)"):
        bar = st.progress(0)
        for i, item in enumerate(subs):
            res = analyze_4_dimensions(client, item["image"])
            update_ai_results(item["id"], res)
            bar.progress((i + 1) / len(subs))
        st.success("AI analysis complete.")
        st.rerun()

# ============================================================
# PHASE 4A: MAP (QUADRANTS) (ADMIN)
# ============================================================
elif app_mode == "The Map (Quadrants)" and is_host:
    st.markdown("### 🗺️ Cluster Map (Quadrants: median split)")

    subs = get_all_submissions()
    if not subs:
        st.warning("No images yet.")
        st.stop()

    rows = []
    for s in subs:
        stats = get_vote_stats(s["id"])
        row = {**s, **stats}
        x, y, source = pick_xy(row)
        row["X"] = x
        row["Y"] = y
        row["XY_source"] = source
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[(df["X"] > 0) & (df["Y"] > 0)].copy()

    if len(df) < 2:
        st.warning("Not enough scored data to map yet (need some human ratings or AI scores).")
        st.stop()

    df = add_quadrants(df)

    st.scatter_chart(df, x="X", y="Y", color="Cluster_Name", size=100, use_container_width=True)
    st.caption("X = Isolation | Y = Intensity | human-first once enough ratings exist")

    st.info(f"Quadrant split points: Isolation ≥ {df['iso_split'].iloc[0]:.2f}, Intensity ≥ {df['int_split'].iloc[0]:.2f}")

    st.divider()
    st.subheader("📂 Cluster Groups (Quadrants)")

    clusters = df.groupby("Cluster_Name")
    c1, c2 = st.columns(2)
    cols = [c1, c2]

    for i, (name, group) in enumerate(clusters):
        with cols[i % 2]:
            st.markdown(f"### {name}")
            for _, row in group.iterrows():
                with st.expander(
                    f"{row['team_name']} | X={row['X']:.2f}, Y={row['Y']:.2f} | source={row['XY_source']} | n={int(row['n'])}"
                ):
                    st.image(blob_to_image(row["image_blob"]))
                    st.write(f"**Prompt:** {row['prompt']}")
                    st.caption(
                        f"Human averages: Iso {row['avg_iso']:.2f} | Int {row['avg_int']:.2f} | "
                        f"Pain {row['avg_pain']:.2f} | Rel {row['avg_rel']:.2f} (n={int(row['n'])})"
                    )
                    if row.get("ai_reasoning"):
                        st.info(f"AI: {row['ai_reasoning']}")

# ============================================================
# PHASE 4B: MAP (KMEANS) (ADMIN) OPTIONAL
# ============================================================
elif app_mode == "The Map (KMeans)" and is_host:
    st.markdown("### 🗺️ Cluster Map (KMeans – optional view)")

    subs = get_all_submissions()
    if not subs:
        st.warning("No images yet.")
        st.stop()

    rows = []
    for s in subs:
        stats = get_vote_stats(s["id"])
        row = {**s, **stats}
        x, y, source = pick_xy(row)
        row["X"] = x
        row["Y"] = y
        row["XY_source"] = source
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[(df["X"] > 0) & (df["Y"] > 0)].copy()

    if len(df) < KMEANS_K * 3:
        st.warning(f"Need at least {KMEANS_K*3} scored images for a stable KMeans view. Use Quadrants instead.")
        st.stop()

    df = add_kmeans(df, k=KMEANS_K)

    st.scatter_chart(df, x="X", y="Y", color="KMeans_Cluster", size=100, use_container_width=True)
    st.caption("X = Isolation | Y = Intensity | KMeans is exploratory; Quadrants is the workshop-default.")

    st.divider()
    st.subheader("📂 KMeans Groups")

    clusters = df.groupby("KMeans_Cluster")
    c1, c2 = st.columns(2)
    cols = [c1, c2]

    for i, (k, group) in enumerate(clusters):
        with cols[i % 2]:
            st.markdown(f"### Cluster {int(k)}")
            for _, row in group.iterrows():
                with st.expander(
                    f"{row['team_name']} | X={row['X']:.2f}, Y={row['Y']:.2f} | source={row['XY_source']} | n={int(row['n'])}"
                ):
                    st.image(blob_to_image(row["image_blob"]))
                    st.write(f"**Prompt:** {row['prompt']}")
                    st.caption(
                        f"Human averages: Iso {row['avg_iso']:.2f} | Int {row['avg_int']:.2f} | "
                        f"Pain {row['avg_pain']:.2f} | Rel {row['avg_rel']:.2f} (n={int(row['n'])})"
                    )
                    if row.get("ai_reasoning"):
                        st.info(f"AI: {row['ai_reasoning']}")

# ============================================================
# DOWNLOAD (ADMIN)
# ============================================================
elif app_mode == "Download Data" and is_host:
    st.markdown("### ⬇️ Download Data")

    subs = get_all_submissions()
    if not subs:
        st.warning("No images yet.")
        st.stop()

    # Build an "images + aggregates + XY + quadrants" export
    rows = []
    for s in subs:
        stats = get_vote_stats(s["id"])
        row = {**s, **stats}
        x, y, source = pick_xy(row)
        row["X"] = x
        row["Y"] = y
        row["XY_source"] = source
        rows.append(row)
    df = pd.DataFrame(rows)
    df_scored = df[(df["X"] > 0) & (df["Y"] > 0)].copy()
    if len(df_scored) >= 2:
        df_scored = add_quadrants(df_scored)

    # 1) Images table CSV (no blobs)
    out1 = StringIO()
    w1 = csv.writer(out1)
    w1.writerow([
        "image_id","team_name","prompt",
        "n_ratings","avg_iso","avg_int","avg_pain","avg_rel",
        "ai_iso","ai_int","ai_pain","ai_rel",
        "X","Y","XY_source",
        "quadrant_cluster"
    ])

    for _, r in df.iterrows():
        cluster = ""
        if "Cluster_Name" in df_scored.columns and r["id"] in set(df_scored["id"].tolist()):
            cluster = df_scored.loc[df_scored["id"] == r["id"], "Cluster_Name"].iloc[0]

        w1.writerow([
            r["id"], r["team_name"], r["prompt"],
            int(r.get("n",0)),
            float(r.get("avg_iso",0.0)), float(r.get("avg_int",0.0)), float(r.get("avg_pain",0.0)), float(r.get("avg_rel",0.0)),
            int(r.get("ai_iso",0)), int(r.get("ai_int",0)), int(r.get("ai_pain",0)), int(r.get("ai_rel",0)),
            float(r.get("X",0.0)), float(r.get("Y",0.0)), r.get("XY_source",""),
            cluster
        ])

    st.download_button(
        "Download image-level CSV (aggregates + clusters)",
        out1.getvalue(),
        file_name="loneliness_images.csv",
        mime="text/csv"
    )

    # 2) Votes table CSV
    votes_df = get_votes_df()
    out2 = StringIO()
    votes_df.to_csv(out2, index=False)

    st.download_button(
        "Download vote-level CSV (raw ratings)",
        out2.getvalue(),
        file_name="loneliness_votes.csv",
        mime="text/csv"
    )

    st.caption("Tip: the vote-level file is what you’ll want for proper modelling later.")

else:
    st.info("Choose a mode from the sidebar.")
