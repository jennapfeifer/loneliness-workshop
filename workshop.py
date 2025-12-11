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

# --- CONFIGURATION ---
st.set_page_config(page_title="Engineering loneliness", layout="wide")
HOST_PASSWORD = "admin123" 

# --- DATABASE FUNCTIONS ---

def init_db():
    """Ensures the table exists."""
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        # Using v3 to match previous schema
        c.execute('''CREATE TABLE IF NOT EXISTS gallery_v3
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      team_name TEXT,
                      prompt TEXT,
                      image_blob BLOB,
                      vote_sum INTEGER DEFAULT 0,
                      vote_count INTEGER DEFAULT 0,
                      average_score REAL DEFAULT 0,
                      ai_score INTEGER,
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
        c.execute("INSERT INTO gallery_v3 (team_name, prompt, image_blob, vote_sum, vote_count, average_score) VALUES (?, ?, ?, 0, 0, 0)",
                  (team, prompt, img_blob))
        conn.commit()

def get_all_submissions():
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM gallery_v3")
        data = c.fetchall()
        
    submissions = []
    for row in data:
        submissions.append({
            'id': row[0], 'team': row[1], 'prompt': row[2],
            'image': blob_to_image(row[3]), 
            'vote_sum': row[4], 'vote_count': row[5], 'human_score': row[6],
            'ai_score': row[7], 'ai_reasoning': row[8], 'ai_details': row[9]
        })
    return submissions

def submit_human_vote(img_id, score):
    """Calculates the AVERAGE of all votes for an image."""
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        c.execute("SELECT vote_sum, vote_count FROM gallery_v3 WHERE id = ?", (img_id,))
        res = c.fetchone()
        if res:
            current_sum, current_count = res
            new_sum = current_sum + score
            new_count = current_count + 1
            new_avg = new_sum / new_count
            c.execute("UPDATE gallery_v3 SET vote_sum = ?, vote_count = ?, average_score = ? WHERE id = ?", 
                      (new_sum, new_count, new_avg, img_id))
            conn.commit()

def update_ai_results(img_id, analysis_json):
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        c.execute("UPDATE gallery_v3 SET ai_score = ?, ai_reasoning = ?, ai_details = ? WHERE id = ?",
                  (analysis_json.get('final_loneliness_index',0), 
                   analysis_json.get('reasoning','Failed'), 
                   json.dumps(analysis_json), 
                   img_id))
        conn.commit()

def reset_db():
    with sqlite3.connect('workshop.db') as conn:
        c = conn.cursor()
        c.execute("DELETE FROM gallery_v3")
        conn.commit()

# --- HOST DATA EXPORT (FIXED) ---
def convert_db_to_csv():
    """Helper to download all collected data using StringIO."""
    submissions = get_all_submissions()
    
    # Use StringIO to create a string buffer for CSV data
    output = StringIO()
    writer = csv.writer(output)
    
    # Write Header
    writer.writerow(["Team Name", "Prompt", "Human Average Score", "Vote Count", "AI Score", "AI Reasoning"])
    
    # Write Rows
    for s in submissions:
        writer.writerow([
            s['team'], 
            s['prompt'], 
            s['human_score'], 
            s['vote_count'], 
            s['ai_score'], 
            s['ai_reasoning']
        ])
        
    return output.getvalue()

# --- ROBUST AI CLIENT ---
def retry_api_call(func, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
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
        generated_image_part = next((p for p in response.parts if p.inline_data), None)
        if generated_image_part:
            img_data = generated_image_part.inline_data.data
            return Image.open(BytesIO(img_data))
        return None
    return retry_api_call(_call)

def analyze_loneliness_literature(client, image):
    lit_prompt = """
    You are an expert psychologist. Analyze this image based on the **Multidimensional Theory of Loneliness** (Weiss, 1973).
    Evaluate on these dimensions:
    1. **Social Isolation:** Absence of community/network.
    2. **Emotional Isolation:** Absence of close attachment.
    3. **Existential Void:** Sense of meaninglessness.

    Return STRICTLY JSON:
    {
        "social_score": (int 0-100),
        "emotional_score": (int 0-100),
        "existential_score": (int 0-100),
        "final_loneliness_index": (int 0-100, weighted average),
        "reasoning": "Concise explanation."
    }
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
    return {"final_loneliness_index": 0, "reasoning": "Analysis failed."}

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = None
    try:
        api_key = st.secrets["google_api"]["key"]
    except Exception:
        st.warning("Secrets not found.")
        api_key = st.text_input("Enter Google API Key", type="password")
    
    client = None
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            st.success("System Online")
        except Exception:
            pass

    st.divider()
    team_name = st.text_input("Team Name", placeholder="Anonymous")
    if not team_name: team_name = "Anonymous"

    st.divider()
    user_role = st.radio("Role:", ["Participant", "Host (Admin)"])
    is_host = False
    if user_role == "Host (Admin)":
        pwd = st.text_input("Password", type="password")
        if pwd == HOST_PASSWORD:
            is_host = True
            st.success("Admin Active")
            if st.button("⚠️ RESET DATA"):
                reset_db()
                st.rerun()

    if is_host:
         app_mode = st.radio("Controls:", ["View Gallery", "Run Analysis", "Download Data"])
         app_mode_prefix = "HOST: "
    else:
         app_mode = st.radio("Steps:", ["1. Create", "2. Gallery & Vote"])
         app_mode_prefix = "PARTICIPANT: "

# --- MAIN APP ---
st.title(f"🧩 Loneliness Experiment | {app_mode_prefix}{app_mode}")

# Initialize session state for tracking user votes
if 'voted_ids' not in st.session_state:
    st.session_state['voted_ids'] = set()

# === PHASE 1: CREATE ===
if app_mode == "1. Create" and not is_host:
    if 'current_draft' not in st.session_state: st.session_state['current_draft'] = None

    st.markdown("Draft your image. **Your prompt will be saved and analyzed.**")
    
    col1, col2 = st.columns([1.2, 1])
    with col1:
        draft_prompt = st.text_area("Describe the image:", height=150, 
            placeholder="A single chair in a vast, empty concrete room...")
        
        if st.button("Generate Draft", type="primary", disabled=not client):
            with st.spinner("Generating..."):
                img = generate_image(client, draft_prompt)
                if img:
                    st.session_state['current_draft'] = {'image': img, 'prompt': draft_prompt}

    with col2:
        if st.session_state['current_draft']:
            st.image(st.session_state['current_draft']['image'], caption="Preview", use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("♻️ Edit Prompt"):
                    st.session_state['current_draft'] = None
                    st.rerun()
            with c2:
                if st.button("✅ Submit", type="primary"):
                    save_submission(team_name, st.session_state['current_draft']['prompt'], st.session_state['current_draft']['image'])
                    st.session_state['current_draft'] = None
                    st.success("Submitted!")
                    time.sleep(1)
                    st.rerun()

# === PHASE 2: GALLERY ===
elif app_mode == "2. Gallery & Vote" or app_mode == "View Gallery":
    st.markdown("Review submissions. **Vote on how lonely the image AND prompt feel.**")
    
    submissions = get_all_submissions()
    if not submissions:
        st.warning("Gallery empty.")
    else:
        submissions.sort(key=lambda x: x['id'], reverse=True)
        cols = st.columns(3)
        for idx, item in enumerate(submissions):
            with cols[idx % 3]:
                st.image(item['image'], use_container_width=True)
                
                # --- VISIBLE PROMPT ---
                st.markdown(f"**Prompt:** _{item['prompt']}_")

                # Voting
                st.caption(f"Avg Score: **{int(item['human_score'])}** ({item['vote_count']} votes)")
                
                # Check if user already voted
                has_voted = item['id'] in st.session_state['voted_ids']
                
                vote_val = st.slider("Rate Loneliness", 0, 100, 50, key=f"s_{item['id']}", disabled=has_voted)
                
                # Button Logic
                btn_label = "✅ Vote Submitted" if has_voted else f"Submit Vote (#{item['id']})"
                
                if st.button(btn_label, key=f"b_{item['id']}", disabled=has_voted):
                    submit_human_vote(item['id'], vote_val)
                    st.session_state['voted_ids'].add(item['id'])
                    st.success("Vote Added!")
                    time.sleep(0.5)
                    st.rerun()

                if item['ai_score']:
                     st.info(f"🤖 AI Score: **{item['ai_score']}**")
                st.divider()

# === PHASE 3: ANALYSIS ===
elif app_mode == "Run Analysis" and is_host:
    st.markdown("### Host: AI Judgment")
    submissions = get_all_submissions()
    
    if st.button("🚀 Analyze All Images", type="primary"):
        progress_bar = st.progress(0)
        for i, item in enumerate(submissions):
            result = analyze_loneliness_literature(client, item['image'])
            update_ai_results(item['id'], result)
            progress_bar.progress((i + 1) / len(submissions))
        st.success("Done!")
        st.rerun()

    st.divider()
    # Leaderboard
    submissions.sort(key=lambda x: x['ai_score'] if x['ai_score'] else 0, reverse=True)
    for rank, item in enumerate(submissions):
        if item['ai_score']:
            with st.container():
                c1, c2 = st.columns([1, 3])
                with c1: st.image(item['image'], use_container_width=True)
                with c2:
                    st.subheader(f"#{rank+1} Team {item['team']}")
                    st.write(f"**Prompt:** {item['prompt']}")
                    st.write(f"**AI Score:** {item['ai_score']} | **Human Avg:** {int(item['human_score'])}")
                    st.markdown(f"> {item['ai_reasoning']}")
                st.divider()

# === PHASE 4: DOWNLOAD DATA ===
elif app_mode == "Download Data" and is_host:
    st.markdown("### 📥 Collect Data")
    st.write("Download the full workshop results (Prompts, Votes, AI Scores) as a CSV file.")
    
    csv_file = convert_db_to_csv()
    
    st.download_button(
        label="Download Results (CSV)",
        data=csv_file,
        file_name="workshop_results.csv",
        mime="text/csv"
    )