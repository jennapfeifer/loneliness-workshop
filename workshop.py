import streamlit as st
from google import genai
from google.genai import types
from io import BytesIO
from PIL import Image
import json
import time
import sqlite3
import base64

# --- CONFIGURATION ---
st.set_page_config(page_title="Loneliness Workshop: AI Experiment", layout="wide")
HOST_PASSWORD = "admin123"  # Change this password if you want

# --- DATABASE FUNCTIONS (SQLite for Shared State) ---
def init_db():
    conn = sqlite3.connect('workshop.db', check_same_thread=False)
    c = conn.cursor()
    # Create table to store image data as BLOBs and scores
    c.execute('''CREATE TABLE IF NOT EXISTS gallery
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  team_name TEXT,
                  prompt TEXT,
                  image_blob BLOB,
                  human_score INTEGER,
                  ai_score INTEGER,
                  ai_reasoning TEXT,
                  ai_details TEXT)''')
    conn.commit()
    return conn

# Initialize connection
conn = init_db()

def image_to_blob(image):
    """Convert PIL Image to BLOB for database storage."""
    buf = BytesIO()
    image.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return byte_im

def blob_to_image(blob):
    """Convert BLOB back to PIL Image."""
    return Image.open(BytesIO(blob))

def save_submission(team, prompt, img_obj):
    """Save a new image submission to the DB."""
    c = conn.cursor()
    img_blob = image_to_blob(img_obj)
    c.execute("INSERT INTO gallery (team_name, prompt, image_blob, human_score) VALUES (?, ?, ?, ?)",
              (team, prompt, img_blob, 50))
    conn.commit()

def get_all_submissions():
    """Fetch all submissions."""
    c = conn.cursor()
    c.execute("SELECT * FROM gallery")
    data = c.fetchall()
    submissions = []
    for row in data:
        submissions.append({
            'id': row[0], 'team': row[1], 'prompt': row[2],
            'image': blob_to_image(row[3]), 'human_score': row[4],
            'ai_score': row[5], 'ai_reasoning': row[6], 'ai_details': row[7]
        })
    return submissions

def update_human_score(img_id, score):
    """Update the human score for an image."""
    c = conn.cursor()
    # Note: In a real multi-user voting scenario, you'd average votes. 
    # For simplicity here, the last person to move the slider sets the score.
    c.execute("UPDATE gallery SET human_score = ? WHERE id = ?", (score, img_id))
    conn.commit()

def update_ai_results(img_id, analysis_json):
    """Save AI analysis results to DB."""
    c = conn.cursor()
    c.execute("UPDATE gallery SET ai_score = ?, ai_reasoning = ?, ai_details = ? WHERE id = ?",
              (analysis_json.get('final_loneliness_index',0), 
               analysis_json.get('reasoning','Failed'), 
               json.dumps(analysis_json), 
               img_id))
    conn.commit()

def reset_db():
    """Clears the gallery table."""
    c = conn.cursor()
    c.execute("DELETE FROM gallery")
    conn.commit()

# --- AI Helper Functions ---

def generate_image(client, prompt):
    """Generates an image using Gemini Image model."""
    try:
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
    except Exception as e:
        st.error(f"Generation Error: {e}")
        return None

def analyze_loneliness_literature(client, image):
    """Analyzes image using multidimensional loneliness framework."""
    lit_prompt = """
    You are an expert psychologist assessing art based on the **Multidimensional Theory of Loneliness** (e.g., Weiss).
    Analyze this image on three specific dimensions:
    1. **Social Isolation (Objective):** Physical absence of others, vast empty space, barriers.
    2. **Emotional Isolation (Subjective):** Atmosphere, cold colors, dark lighting, symbolism of decay.
    3. **Existential Solitude:** Sense of insignificance or being overwhelmed by the void.

    Return STRICTLY a JSON object:
    {
        "social_score": (int 0-100),
        "emotional_score": (int 0-100),
        "existential_score": (int 0-100),
        "final_loneliness_index": (int 0-100, weighted average),
        "reasoning": "A concise 2-sentence explanation justifying the final index."
    }
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[image, lit_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Analysis Error: {e}")
        return {"final_loneliness_index": 0, "reasoning": f"Error: {e}"}

# --- SIDEBAR & SETUP ---
with st.sidebar:
    st.header("⚙️ Workshop Setup")
    
    # API Key
    api_key = None
    try:
        # Try to get key from Streamlit secrets (Cloud)
        api_key = st.secrets["google_api"]["key"]
    except Exception:
        # Fallback to manual entry (Local testing)
        st.warning("Secrets not found. Using manual entry.")
        api_key = st.text_input("Enter Google API Key", type="password")
    
    client = None
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            st.success("AI System Online")
        except Exception as e:
            st.error(f"API Error: {e}")

    st.divider()
    
    # User Identity
    team_name = st.text_input("Your Team Name", placeholder="Anonymous Wanderer")
    if not team_name: team_name = "Anonymous"

    st.divider()
    
    # Role Switching
    user_role = st.radio("Select Role:", ["Participant", "Host (Admin)"])
    is_host = False
    if user_role == "Host (Admin)":
        pwd = st.text_input("Host Password", type="password")
        if pwd == HOST_PASSWORD:
            is_host = True
            st.success("Host Mode Active")
            if st.button("⚠️ RESET DATABASE (Deletes All Images)"):
                reset_db()
                st.cache_data.clear()
                st.rerun()
        elif pwd:
            st.error("Incorrect Password")

    # Navigation based on role
    if is_host:
         app_mode = st.radio("Host Controls:", 
            ["View Gallery & Votes", "Run AI Analysis"])
         app_mode_prefix = "HOST: "
    else:
         app_mode = st.radio("Participant Steps:", 
            ["1. Create & Submit", "2. View Gallery & Vote"])
         app_mode_prefix = "PARTICIPANT: "

# --- MAIN APP LOGIC ---

st.title(f"🧩 Loneliness Experiment | {app_mode_prefix}{app_mode}")

# === PHASE 1: CREATE & SUBMIT (Participant Only) ===
if app_mode == "1. Create & Submit" and not is_host:
    if 'current_draft' not in st.session_state: st.session_state['current_draft'] = None

    st.markdown("Generate an image. If you don't like it, edit the prompt and regenerate. Once satisfied, **Submit** to the shared gallery.")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        draft_prompt = st.text_area("Describe the image of loneliness:", height=150, 
            placeholder="A single, small figure sitting on a bench facing a dense fog...")
        
        generate_btn = st.button("Generate Draft Image", type="primary", disabled=not client)
        
        if generate_btn and client and draft_prompt:
            with st.spinner("Generating..."):
                img = generate_image(client, draft_prompt)
                if img:
                    st.session_state['current_draft'] = {'image': img, 'prompt': draft_prompt}
                else:
                    st.error("Generation failed.")

    with col2:
        if st.session_state['current_draft']:
            st.image(st.session_state['current_draft']['image'], caption="Your Draft", use_container_width=True)
            
            st.info("Happy with this result?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("♻️ Scrap & Edit Prompt"):
                    st.session_state['current_draft'] = None
                    st.rerun()
            with c2:
                if st.button("✅ Submit to Shared Gallery", type="primary"):
                    save_submission(team_name, st.session_state['current_draft']['prompt'], st.session_state['current_draft']['image'])
                    st.session_state['current_draft'] = None
                    st.success("Submitted! Move to Phase 2 to view the gallery.")
                    time.sleep(2)
                    st.rerun()
        else:
            st.markdown(
                """
                <div style="display: flex; justify-content: center; align-items: center; height: 400px; border: 2px dashed grey; background-color: #f0f2f6; color: grey;">
                    Draft image will appear here.
                </div>
                """, unsafe_allow_html=True
            )

# === PHASE 2 & HOST VIEW: GALLERY & VOTE ===
elif app_mode == "2. View Gallery & Vote" or app_mode == "View Gallery & Votes":
    st.markdown("Review submissions from everyone. **Use the sliders to rank how lonely they feel to you.**")
    st.caption("Note: The last person to move a slider sets the score for that image.")
    
    submissions = get_all_submissions()
    
    if not submissions:
        st.warning("The gallery is waiting for submissions.")
        st.info("Go to '1. Create & Submit' to add images.")
    else:
        # Display grid
        cols = st.columns(3)
        for idx, item in enumerate(submissions):
            with cols[idx % 3]:
                st.image(item['image'], use_container_width=True)
                st.caption(f"Team: **{item['team']}**")
                
                # Human Ranking Input - updates DB directly
                new_score = st.slider(
                    f"Loneliness Score (ID: {item['id']})", 
                    0, 100, item['human_score'], 
                    key=f"slider_{item['id']}"
                )
                if new_score != item['human_score']:
                     update_human_score(item['id'], new_score)

                if item['ai_score']:
                     st.success(f"**AI Final Score: {item['ai_score']}**")
                st.divider()

# === HOST ONLY: RUN AI ANALYSIS ===
elif app_mode == "Run AI Analysis" and is_host:
    st.markdown("### Host Control: Final Judgment")
    st.write("As host, run the AI analysis on all submitted images.")
    
    submissions = get_all_submissions()
    if not submissions:
        st.error("No images in database to analyze.")
    else:
        count_unanalyzed = len([s for s in submissions if s['ai_score'] is None])
        st.write(f"Total Images: {len(submissions)}. Unanalyzed: {count_unanalyzed}")

        if st.button("🚀 Launch AI Analysis Cycle", type="primary", disabled=not client):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, item in enumerate(submissions):
                status_text.write(f"Analyzing Image ID {item['id']} by Team '{item['team']}'...")
                analysis_result = analyze_loneliness_literature(client, item['image'])
                update_ai_results(item['id'], analysis_result)
                progress_bar.progress((i + 1) / len(submissions))
            
            status_text.success("Analysis Complete! Switch to gallery view to see results.")
            time.sleep(2)
            st.rerun()

    st.divider()
    st.subheader("🏆 Leaderboard (AI Ranking)")
    # Get fresh data and sort
    fresh_submissions = get_all_submissions()
    analyzed = [s for s in fresh_submissions if s['ai_score'] is not None]
    analyzed.sort(key=lambda x: x['ai_score'] if x['ai_score'] else 0, reverse=True)
    
    if analyzed:
         for rank, item in enumerate(analyzed):
             st.markdown(f"**#{rank+1}: Team {item['team']}** | AI Score: **{item['ai_score']}** | Human Vote: {item['human_score']}")
             st.caption(f"AI Reasoning: {item['ai_reasoning']}")
    else:
        st.write("Run analysis to see rankings.")