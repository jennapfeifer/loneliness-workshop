import streamlit as st
from google import genai
from google.genai import types
from io import BytesIO, StringIO
from PIL import Image
import json
import time
import sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# --- CONFIGURATION ---
st.set_page_config(page_title="Loneliness Workshop: Clusters", layout="wide")
HOST_PASSWORD = "admin123" 

# --- DATABASE (V6 Schema: 4 Dimensions) ---
def init_db():
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS gallery_v6
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      team_name TEXT,
                      prompt TEXT,
                      image_blob BLOB,
                      vote_count INTEGER DEFAULT 0,
                      
                      -- CORE (Clustering)
                      sum_iso INTEGER DEFAULT 0,  -- Social Isolation
                      sum_int INTEGER DEFAULT 0,  -- Emotional Intensity
                      
                      -- OPTIONAL (Context)
                      sum_pain INTEGER DEFAULT 0, -- Negative/Painful
                      sum_rel INTEGER DEFAULT 0,  -- Relatability
                      
                      -- AVERAGES
                      avg_iso REAL DEFAULT 0,
                      avg_int REAL DEFAULT 0,
                      avg_pain REAL DEFAULT 0,
                      avg_rel REAL DEFAULT 0,
                      
                      -- AI SCORES
                      ai_iso INTEGER DEFAULT 0,
                      ai_int INTEGER DEFAULT 0,
                      ai_pain INTEGER DEFAULT 0,
                      ai_rel INTEGER DEFAULT 0,
                      ai_reasoning TEXT)''')
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
        # Initialize with 0s
        c.execute("""INSERT INTO gallery_v6 
                     (team_name, prompt, image_blob) 
                     VALUES (?, ?, ?)""",
                  (team, prompt, img_blob))
        conn.commit()

def get_all_submissions():
    with sqlite3.connect('workshop.db') as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM gallery_v6")
        data = c.fetchall()
        
    submissions = []
    for row in data:
        sub = dict(row)
        sub['image'] = blob_to_image(row['image_blob'])
        submissions.append(sub)
    return submissions

def submit_human_vote(img_id, iso, intel, pain, rel):
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        c.execute("SELECT sum_iso, sum_int, sum_pain, sum_rel, vote_count FROM gallery_v6 WHERE id = ?", (img_id,))
        res = c.fetchone()
        if res:
            s_iso, s_int, s_pain, s_rel, count = res
            
            # Update Sums
            n_iso = s_iso + iso
            n_int = s_int + intel
            n_pain = s_pain + pain
            n_rel = s_rel + rel
            n_count = count + 1
            
            # Update DB with new averages
            c.execute("""UPDATE gallery_v6 SET 
                         vote_count=?,
                         sum_iso=?, sum_int=?, sum_pain=?, sum_rel=?,
                         avg_iso=?, avg_int=?, avg_pain=?, avg_rel=?
                         WHERE id=?""", 
                      (n_count, 
                       n_iso, n_int, n_pain, n_rel,
                       n_iso/n_count, n_int/n_count, n_pain/n_count, n_rel/n_count, 
                       img_id))
            conn.commit()

def update_ai_results(img_id, analysis_json):
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        c.execute("""UPDATE gallery_v6 SET 
                     ai_iso=?, ai_int=?, ai_pain=?, ai_rel=?, 
                     ai_reasoning=? 
                     WHERE id=?""",
                  (analysis_json.get('isolation',0), 
                   analysis_json.get('intensity',0),
                   analysis_json.get('pain',0),
                   analysis_json.get('relatability',0),
                   analysis_json.get('reasoning','Failed'), 
                   img_id))
        conn.commit()

def reset_db():
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        c.execute("DELETE FROM gallery_v6")
        conn.commit()

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

def analyze_4_dimensions(client, image):
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
    return {"isolation": 0, "intensity": 0, "pain": 0, "relatability": 0, "reasoning": "Failed."}

# --- CLUSTERING HELPER ---
def run_kmeans(df):
    """
    Runs K-Means on the Core Dimensions (Isolation & Intensity).
    Returns the dataframe with a new 'Cluster' column and 'Cluster_Name'.
    """
    if len(df) < 4:
        # Not enough data for 4 clusters
        df['Cluster'] = 0
        df['Cluster_Name'] = "Ungrouped"
        return df

    # Prepare data for clustering (Core dimensions only)
    # We prioritize AI score, fallback to Human Average
    X = df.apply(lambda r: [
        r['ai_iso'] if r['ai_iso'] > 0 else r['avg_iso'],
        r['ai_int'] if r['ai_int'] > 0 else r['avg_int']
    ], axis=1).tolist()
    
    # Run K-Means (K=4 for distinct groups)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    df['Cluster'] = labels
    
    # Name clusters based on Centroids
    centers = kmeans.cluster_centers_
    cluster_names = {}
    
    for i, center in enumerate(centers):
        iso_c, int_c = center[0], center[1]
        
        # Heuristic Naming Logic (1-5 Scale)
        if iso_c > 3.0 and int_c > 3.0:
            name = "The Scream (High Iso, High Int)"
        elif iso_c > 3.0 and int_c <= 3.0:
            name = "Quiet Void (High Iso, Low Int)"
        elif iso_c <= 3.0 and int_c > 3.0:
            name = "Social Friction (Low Iso, High Int)"
        else:
            name = "Calm Connection (Low Iso, Low Int)"
        
        cluster_names[i] = name
        
    df['Cluster_Name'] = df['Cluster'].map(cluster_names)
    return df

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
         app_mode = st.radio("Controls:", ["View Gallery", "Run Analysis", "The Map (Clusters)", "Download Data"])
    else:
         app_mode = st.radio("Steps:", ["1. Create", "2. Rate Images"])

st.title(f"🧩 Loneliness: Subjective Profiles")
if 'voted_ids' not in st.session_state: st.session_state['voted_ids'] = set()

# === PHASE 1: CREATE ===
if app_mode == "1. Create" and not is_host:
    st.markdown("Generate an image that represents a specific feeling of loneliness.")
    
    if 'current_draft' not in st.session_state: st.session_state['current_draft'] = None
    
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.form("gen_form"):
            prompt = st.text_area("Prompt:", height=150)
            submitted = st.form_submit_button("Generate Draft", type="primary")
            if submitted:
                if not client: st.error("No API Key"); st.stop()
                with st.spinner("Dreaming..."):
                    img = generate_image(client, prompt)
                    if img: st.session_state['current_draft'] = {'image': img, 'prompt': prompt}; st.rerun()

    with col2:
        if st.session_state['current_draft']:
            st.image(st.session_state['current_draft']['image'], caption="Draft", use_container_width=True)
            c1, c2 = st.columns(2)
            if c1.button("♻️ Scrap"): st.session_state['current_draft'] = None; st.rerun()
            if c2.button("✅ Submit"):
                save_submission(team_name, st.session_state['current_draft']['prompt'], st.session_state['current_draft']['image'])
                st.session_state['current_draft'] = None; st.success("Saved!"); time.sleep(1); st.rerun()

# === PHASE 2: RATING (1-5 SCALE) ===
elif app_mode == "2. Rate Images" or app_mode == "View Gallery":
    st.markdown("Rate images on **Core** (Clustering) and **Optional** dimensions.")
    subs = get_all_submissions()
    
    if not subs: st.warning("No images yet.")
    else:
        subs.sort(key=lambda x: x['id'], reverse=True)
        cols = st.columns(3)
        for idx, item in enumerate(subs):
            with cols[idx % 3]:
                st.image(item['image'], use_container_width=True)
                st.caption(f"_{item['prompt']}_")
                
                has_voted = item['id'] in st.session_state['voted_ids']
                
                with st.form(key=f"vote_{item['id']}"):
                    st.markdown("**Core (For Clustering)**")
                    iso = st.slider("Social Isolation (1=Crowded, 5=Void)", 1, 5, 3)
                    inte = st.slider("Intensity (1=Calm, 5=Overwhelming)", 1, 5, 3)
                    
                    with st.expander("Optional (Context)"):
                        pain = st.slider("Negativity (1=Peaceful, 5=Painful)", 1, 5, 3)
                        rel = st.slider("Relatability (1=Alien, 5=Me)", 1, 5, 3)
                    
                    if st.form_submit_button("Submit Ratings", disabled=has_voted):
                        submit_human_vote(item['id'], iso, inte, pain, rel)
                        st.session_state['voted_ids'].add(item['id'])
                        st.rerun()
                
                if item['vote_count'] > 0: st.caption(f"Votes: {item['vote_count']}")
                st.divider()

# === PHASE 3: AI ANALYSIS ===
elif app_mode == "Run Analysis" and is_host:
    if st.button("🚀 Analyze All Images (1-5 Scale)"):
        subs = get_all_submissions()
        bar = st.progress(0)
        for i, item in enumerate(subs):
            res = analyze_4_dimensions(client, item['image'])
            update_ai_results(item['id'], res)
            bar.progress((i+1)/len(subs))
        st.success("Analysis Complete!")
        st.rerun()

# === PHASE 4: THE CLUSTER MAP ===
elif app_mode == "The Map (Clusters)" and is_host:
    st.markdown("### 🗺️ Cluster Map (K-Means)")
    subs = get_all_submissions()
    
    if len(subs) < 4:
        st.warning("Need at least 4 images to run K-Means Clustering.")
    else:
        # Convert to DF and Run K-Means
        df = pd.DataFrame(subs)
        df = run_kmeans(df)
        
        # Calculate Plot Coordinates (Prioritize AI, fallback to Human)
        df['X (Iso)'] = df.apply(lambda r: r['ai_iso'] if r['ai_iso']>0 else r['avg_iso'], axis=1)
        df['Y (Int)'] = df.apply(lambda r: r['ai_int'] if r['ai_int']>0 else r['avg_int'], axis=1)
        
        # 1. SCATTER PLOT
        st.scatter_chart(
            df,
            x='X (Iso)', y='Y (Int)',
            color='Cluster_Name',
            size=100,
            use_container_width=True
        )
        st.caption("X: Isolation (1-5) | Y: Intensity (1-5)")
        
        st.divider()
        
        # 2. CLUSTER GROUPS
        st.subheader("📂 The Detected Clusters")
        
        # Group by Cluster Name
        clusters = df.groupby('Cluster_Name')
        
        c1, c2 = st.columns(2)
        cols = [c1, c2]
        
        for i, (name, group) in enumerate(clusters):
            with cols[i % 2]:
                st.markdown(f"### {name}")
                for _, row in group.iterrows():
                    with st.expander(f"{row['team_name']} (Iso:{row['X (Iso)']}, Int:{row['Y (Int)']})"):
                        # Re-open blob to show image
                        img = blob_to_image(row['image_blob'])
                        st.image(img)
                        st.write(f"**Prompt:** {row['prompt']}")
                        st.caption(f"**Context:** Pain {row['avg_pain']:.1f} | Relatable {row['avg_rel']:.1f}")
                        if row['ai_reasoning']:
                            st.info(f"AI: {row['ai_reasoning']}")

# === DOWNLOAD ===
elif app_mode == "Download Data" and is_host:
    subs = get_all_submissions()
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Team","Prompt","Iso","Int","Pain","Rel","AI_Iso","AI_Int","Cluster"])
    # Note: Cluster calculation happens in Map view, so raw CSV just dumps raw scores
    for s in subs:
        writer.writerow([s['team_name'],s['prompt'],s['avg_iso'],s['avg_int'],s['avg_pain'],s['avg_rel'],s['ai_iso'],s['ai_int'],"Run Map to Cluster"])
    
    st.download_button("Download CSV", output.getvalue(), "clusters.csv", "text/csv")
