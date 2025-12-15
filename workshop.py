import streamlit as st
from google import genai
from google.genai import types
from io import BytesIO, StringIO
from PIL import Image
import json
import time
import sqlite3
import base64
import csv
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="Loneliness Workshop: The Map", layout="wide")
HOST_PASSWORD = "admin123" 

# --- DATABASE FUNCTIONS (V5 Schema for Density/Valence) ---
def init_db():
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        # V5 Schema: Stores Valence (Peace/Pain) and Density (Void/Crowd)
        c.execute('''CREATE TABLE IF NOT EXISTS gallery_v5
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      team_name TEXT,
                      prompt TEXT,
                      image_blob BLOB,
                      vote_count INTEGER DEFAULT 0,
                      sum_valence INTEGER DEFAULT 0,  -- Peace vs Pain
                      sum_density INTEGER DEFAULT 0,  -- Void vs Crowd
                      avg_valence REAL DEFAULT 0,
                      avg_density REAL DEFAULT 0,
                      ai_valence INTEGER DEFAULT 0,
                      ai_density INTEGER DEFAULT 0,
                      ai_reasoning TEXT,
                      ai_details TEXT)''')
        conn.commit()

init_db()

def image_to_blob(image):
    buf = BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()

def blob_to_image(blob):
    return Image.open(BytesIO(blob))

def save_submission(team, prompt, img_obj):
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        img_blob = image_to_blob(img_obj)
        c.execute("""INSERT INTO gallery_v5 
                     (team_name, prompt, image_blob, vote_count, sum_valence, sum_density) 
                     VALUES (?, ?, ?, 0, 0, 0)""",
                  (team, prompt, img_blob))
        conn.commit()

def get_all_submissions():
    with sqlite3.connect('workshop.db') as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM gallery_v5")
        data = c.fetchall()
        
    submissions = []
    for row in data:
        sub = dict(row)
        sub['image'] = blob_to_image(row['image_blob'])
        submissions.append(sub)
    return submissions

def submit_human_vote(img_id, val, dens):
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        c.execute("SELECT sum_valence, sum_density, vote_count FROM gallery_v5 WHERE id = ?", (img_id,))
        res = c.fetchone()
        if res:
            s_val, s_dens, count = res
            n_val, n_dens = s_val + val, s_dens + dens
            n_count = count + 1
            c.execute("""UPDATE gallery_v5 SET 
                         sum_valence=?, sum_density=?, vote_count=?,
                         avg_valence=?, avg_density=?
                         WHERE id=?""", 
                      (n_val, n_dens, n_count, 
                       n_val/n_count, n_dens/n_count, img_id))
            conn.commit()

def update_ai_results(img_id, analysis_json):
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        c.execute("""UPDATE gallery_v5 SET 
                     ai_valence=?, ai_density=?, 
                     ai_reasoning=?, ai_details=? 
                     WHERE id=?""",
                  (analysis_json.get('valence_score',0), 
                   analysis_json.get('density_score',0),
                   analysis_json.get('reasoning','Failed'), 
                   json.dumps(analysis_json), 
                   img_id))
        conn.commit()

def reset_db():
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        c.execute("DELETE FROM gallery_v5")
        conn.commit()

# --- CLUSTERING LOGIC (The 4 Types) ---
def get_quadrant(valence, density):
    # Valence: 0 (Peace) -> 100 (Pain)
    # Density: 0 (Void) -> 100 (Crowd)
    
    if valence >= 50 and density >= 50:
        return "Urban Isolation", "🏙️ (Painful Crowd)"
    elif valence >= 50 and density < 50:
        return "The Abyssal Void", "🌌 (Painful Empty)"
    elif valence < 50 and density >= 50:
        return "Anonymous Observer", "☕ (Peaceful Crowd)"
    else:
        return "Sacred Solitude", "🌲 (Peaceful Empty)"

# --- AI CLIENT ---
def retry_api_call(func, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return func()
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            return None

def generate_image(client, prompt):
    def _call():
        response = client.models.generate_content(
            model='gemini-2.5-flash-image',
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
            ),
        )
        img_part = next((p for p in response.parts if p.inline_data), None)
        if img_part: return Image.open(BytesIO(img_part.inline_data.data))
        return None
    return retry_api_call(_call)

def analyze_dimensions(client, image):
    lit_prompt = """
    Analyze this image on these 2 specific scales (0-100).
    
    1. Valence (Peace vs Pain): 
       - 0 = Peaceful, Restorative, Calm, 'Happy Alone'
       - 100 = Painful, Scary, Depressing, 'Sad Alone'
       
    2. Density (Void vs Crowd):
       - 0 = The Void (Empty space, desert, ocean, single room, vastness)
       - 100 = The Crowd (City street, party, clutter, chaos, many people)
    
    Return STRICTLY JSON:
    {"valence_score": int, "density_score": int, "reasoning": "string"}
    """
    def _call():
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[image, lit_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)

    result = retry_api_call(_call)
    if result: return result
    return {"valence_score": 0, "density_score": 0, "reasoning": "Failed."}

# --- HOST DATA EXPORT ---
def convert_db_to_csv():
    submissions = get_all_submissions()
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Team", "Prompt", "Vote Count", "Human: Pain", "Human: Crowd", "AI: Pain", "AI: Crowd", "Reasoning"])
    for s in submissions:
        writer.writerow([
            s['team_name'], s['prompt'], s['vote_count'],
            f"{s['avg_valence']:.1f}", f"{s['avg_density']:.1f}",
            s['ai_valence'], s['ai_density'], s['ai_reasoning']
        ])
    return output.getvalue()

# --- UI SETUP ---
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.secrets["google_api"]["key"] if "google_api" in st.secrets else st.text_input("API Key", type="password")
    client = genai.Client(api_key=api_key) if api_key else None
    
    st.divider()
    team_name = st.text_input("Team Name", placeholder="Anonymous")
    
    st.divider()
    user_role = st.radio("Role:", ["Participant", "Host (Admin)"])
    is_host = False
    if user_role == "Host (Admin)":
        if st.text_input("Password", type="password") == HOST_PASSWORD:
            is_host = True
            if st.button("⚠️ RESET DATA"):
                reset_db()
                st.rerun()

    if is_host:
         app_mode = st.radio("Controls:", ["View Gallery", "Run Analysis", "The Map (Cluster View)", "Download Data"])
    else:
         app_mode = st.radio("Steps:", ["1. Create", "2. Gallery & Vote"])

st.title(f"🧩 The Texture of Loneliness")
if 'voted_ids' not in st.session_state: st.session_state['voted_ids'] = set()

# === PHASE 1: CREATE ===
if app_mode == "1. Create" and not is_host:
    if 'current_draft' not in st.session_state: st.session_state['current_draft'] = None
    st.markdown("Create an image. Is it a **Crowded** loneliness or an **Empty** one?")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        draft_prompt = st.text_area("Prompt:", height=150)
        if st.button("Generate Draft", type="primary", disabled=not client):
            with st.spinner("Dreaming..."):
                img = generate_image(client, draft_prompt)
                if img: st.session_state['current_draft'] = {'image': img, 'prompt': draft_prompt}
    with col2:
        if st.session_state['current_draft']:
            st.image(st.session_state['current_draft']['image'], caption="Draft", use_container_width=True)
            c1, c2 = st.columns(2)
            if c1.button("♻️ Edit"): st.session_state['current_draft'] = None; st.rerun()
            if c2.button("✅ Submit"):
                save_submission(team_name, st.session_state['current_draft']['prompt'], st.session_state['current_draft']['image'])
                st.session_state['current_draft'] = None; st.success("Submitted!"); time.sleep(1); st.rerun()

# === PHASE 2: VOTING ===
elif app_mode == "2. Gallery & Vote" or app_mode == "View Gallery":
    st.markdown("Where does this image fit on the map?")
    subs = get_all_submissions()
    if not subs: st.warning("Empty Gallery")
    else:
        subs.sort(key=lambda x: x['id'], reverse=True)
        cols = st.columns(3)
        for idx, item in enumerate(subs):
            with cols[idx % 3]:
                st.image(item['image'], use_container_width=True)
                st.caption(f"_{item['prompt']}_")
                
                has_voted = item['id'] in st.session_state['voted_ids']
                
                # --- NEW SLIDERS ---
                st.markdown("**1. Emotional Vibe**")
                v_val = st.slider("Peaceful (0) ↔ Painful (100)", 0, 100, 50, key=f"v_{item['id']}", disabled=has_voted)
                
                st.markdown("**2. Physical Space**")
                v_dens = st.slider("Empty Void (0) ↔ Crowded (100)", 0, 100, 50, key=f"d_{item['id']}", disabled=has_voted)
                
                if st.button("Submit Vote" if not has_voted else "✅ Voted", key=f"b_{item['id']}", disabled=has_voted):
                    submit_human_vote(item['id'], v_val, v_dens)
                    st.session_state['voted_ids'].add(item['id'])
                    st.rerun()
                
                if item['ai_reasoning']:
                    st.info(f"🤖 AI: Pain={item['ai_valence']} | Crowd={item['ai_density']}")
                st.divider()

# === PHASE 3: ANALYSIS ===
elif app_mode == "Run Analysis" and is_host:
    if st.button("🚀 Analyze Dimensions"):
        subs = get_all_submissions()
        bar = st.progress(0)
        for i, item in enumerate(subs):
            res = analyze_dimensions(client, item['image'])
            update_ai_results(item['id'], res)
            bar.progress((i+1)/len(subs))
        st.success("Done!")
        st.rerun()

# === PHASE 4: THE MAP (CLUSTERING) ===
elif app_mode == "The Map (Cluster View)" and is_host:
    st.markdown("### 🗺️ The Map of Loneliness")
    subs = get_all_submissions()
    
    if not subs:
        st.warning("No Data.")
    else:
        # 1. SCATTER PLOT
        df = pd.DataFrame(subs)
        # Prefer AI score, fallback to Human Avg
        df['X (Crowd)'] = df.apply(lambda r: r['ai_density'] if r['ai_density'] > 0 else r['avg_density'], axis=1)
        df['Y (Pain)'] = df.apply(lambda r: r['ai_valence'] if r['ai_valence'] > 0 else r['avg_valence'], axis=1)
        
        st.scatter_chart(
            df,
            x='X (Crowd)',
            y='Y (Pain)',
            color='team_name',
            size=100,
            use_container_width=True
        )
        st.caption("⬅️ EMPTY (Void) .................................... CROWDED (Urban) ➡️")
        
        st.divider()
        st.subheader("📂 The 4 Types of Solitude")
        
        # Group items
        clusters = {"Urban Isolation": [], "The Abyssal Void": [], "Anonymous Observer": [], "Sacred Solitude": []}
        
        for s in subs:
            val = s['ai_valence'] if s['ai_valence'] > 0 else s['avg_valence']
            dens = s['ai_density'] if s['ai_density'] > 0 else s['avg_density']
            name, emoji = get_quadrant(val, dens)
            clusters[name].append(s)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🏙️ Urban Isolation\n*(Painful + Crowded)*")
            for s in clusters["Urban Isolation"]:
                with st.expander(f"{s['team_name']}"):
                    st.image(s['image'])
                    st.write(s['prompt'])
            
            st.markdown("### ☕ Anonymous Observer\n*(Peaceful + Crowded)*")
            for s in clusters["Anonymous Observer"]:
                with st.expander(f"{s['team_name']}"):
                    st.image(s['image'])
                    st.write(s['prompt'])

        with c2:
            st.markdown("### 🌌 The Abyssal Void\n*(Painful + Empty)*")
            for s in clusters["The Abyssal Void"]:
                with st.expander(f"{s['team_name']}"):
                    st.image(s['image'])
                    st.write(s['prompt'])

            st.markdown("### 🌲 Sacred Solitude\n*(Peaceful + Empty)*")
            for s in clusters["Sacred Solitude"]:
                with st.expander(f"{s['team_name']}"):
                    st.image(s['image'])
                    st.write(s['prompt'])

# === DOWNLOAD ===
elif app_mode == "Download Data" and is_host:
    st.download_button("Download CSV", convert_db_to_csv(), "loneliness_map.csv", "text/csv")
