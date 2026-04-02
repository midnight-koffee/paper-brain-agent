import os
import re
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
import fitz  # PyMuPDF
from supabase import create_client, Client
from streamlit_cookies_controller import CookieController

load_dotenv()

# --- Global Settings ---
fitz.TOOLS.mupdf_display_errors(False)
st.set_page_config(page_title="Paper Brain", page_icon="🧠", layout="wide")

# Initialize the Cookie Controller for persistence
controller = CookieController()

# --- 1. Database & Session Initialization ---
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

if "supabase" not in st.session_state:
    try:
        st.session_state.supabase = create_client(url, key)
    except Exception as e:
        st.error(f"Failed to connect to Supabase. Error: {e}")
        st.stop()

supabase = st.session_state.supabase

# --- FIX: The Ghost Login & Session Patch ---
stored_token = controller.get("supabase_token")
if stored_token and "user" not in st.session_state:
    try:
        # 1. Recover user info
        res = supabase.auth.get_user(stored_token)
        st.session_state.user = res.user
        # 2. CRITICAL: Set the session on the client so RLS allows DB access
        supabase.auth.set_session(stored_token, stored_token)
    except Exception:
        # If token is invalid/expired, wipe it
        try:
            controller.remove("supabase_token")
        except KeyError:
            pass

# Initialize Session States
if "client" not in st.session_state:
    st.session_state.client = genai.Client()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_context" not in st.session_state:
    st.session_state.doc_context = ""
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None

# --- AI Memory Sync Helper ---
def init_gemini():
    """Syncs current messages to Gemini's history & allows chat without docs."""
    system_instruction = "You are 'Paper Brain', an expert research assistant."
    
    if st.session_state.doc_context:
        system_instruction += (
            f"\n\nAlways rely strictly on the provided text. Use strict LaTeX: $x$ for inline, $$x$$ for blocks.\n\n"
            f"VAULT CONTENTS:\n{st.session_state.doc_context}"
        )

    # Convert Streamlit messages to Gemini history format
    gemini_history = []
    for m in st.session_state.messages:
        if not m.get("is_status"): 
            role = "user" if m["role"] == "user" else "model"
            gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
            
    return st.session_state.client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(system_instruction=system_instruction),
        history=gemini_history
    )

def reset_ui_state():
    st.session_state.messages = []
    st.session_state.doc_context = ""
    st.session_state.chat_session = None
    st.session_state.processed_files = []
    st.session_state.docs_loaded = False
    st.session_state.current_thread_id = None

# --- 🚨 THE BOUNCER (LOGIN SCREEN) 🚨 ---
if not st.session_state.get("user"):
    st.title("🔐 Welcome to Paper Brain")
    st.markdown("Your personal, persistent research vault.")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        log_email = st.text_input("Email", key="log_email")
        log_password = st.text_input("Password", type="password", key="log_pass")
        if st.button("Login"):
            try:
                response = supabase.auth.sign_in_with_password({"email": log_email, "password": log_password})
                st.session_state.user = response.user
                controller.set("supabase_token", response.session.access_token)
                st.rerun()
            except Exception as e:
                st.error(f"Login Failed: {str(e)}")
                
    with tab2:
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Create Account"):
            try:
                supabase.auth.sign_up({"email": reg_email, "password": reg_password})
                st.success("✅ Account created! Check your email inbox for a confirmation link.")
            except Exception as e:
                st.error(f"Signup Error: {str(e)}")
    st.stop()

# --- 2. Load Vault Data on Login ---
if st.session_state.user and not st.session_state.docs_loaded:
    with st.spinner("Unlocking your vault..."):
        try:
            response = supabase.table("documents").select("*").eq("user_id", st.session_state.user.id).execute()
            if response.data:
                combined_text = ""
                file_names = []
                for doc in response.data:
                    file_names.append(doc["file_name"])
                    combined_text += f"\n--- START OF DOCUMENT: {doc['file_name']} ---\n{doc['summary']}\n--- END OF DOCUMENT ---\n"
                
                st.session_state.doc_context = combined_text
                st.session_state.processed_files = file_names
            
            # Initialize the AI session (with or without docs)
            st.session_state.chat_session = init_gemini()
        except Exception as e:
            st.error(f"Error loading vault: {e}")
            
    st.session_state.docs_loaded = True

# --- 3. Sidebar: ChatGPT UI & Vault ---
with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.current_thread_id = None
        st.session_state.messages = []
        st.session_state.chat_session = init_gemini()
        st.rerun()
        
    st.divider()

    st.markdown("### 💬 Past Chats")
    try:
        threads = supabase.table("chat_threads").select("*").eq("user_id", st.session_state.user.id).order("created_at", desc=True).execute().data
        if threads:
            for t in threads:
                if st.button(t['title'], key=f"thread_{t['id']}", use_container_width=True):
                    st.session_state.current_thread_id = t['id']
                    msgs = supabase.table("chat_messages").select("*").eq("thread_id", t['id']).order("created_at", desc=False).execute().data
                    st.session_state.messages = [{"role": m["role"], "content": m["content"], "is_status": False} for m in msgs]
                    st.session_state.chat_session = init_gemini()
                    st.rerun()
        else:
            st.caption("No past chats yet.")
    except Exception:
        pass
        
    st.divider()
    
    st.markdown("### 📄 Document Vault")
    if st.session_state.processed_files:
        for f in st.session_state.processed_files:
            st.caption(f"✅ {f}")
    else:
        st.caption("Your vault is empty.")
        
    uploaded_files = st.file_uploader("Upload new PDFs", type=["pdf", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process & Save Uploads"):
            with st.spinner("Extracting text..."):
                new_text_added = ""
                error_occurred = False
                
                for file in uploaded_files:
                    if file.name not in st.session_state.processed_files:
                        text = ""
                        if file.name.lower().endswith('.pdf'):
                            try:
                                doc = fitz.open(stream=file.read(), filetype="pdf")
                                for page in doc:
                                    # FIX: Robust Regex Cleaner for Null Bytes and Control Chars
                                    raw_text = page.get_text()
                                    clean_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw_text)
                                    text += clean_text + "\n"
                            except Exception as e:
                                st.error(f"Error reading {file.name}: {str(e)}")
                                error_occurred = True
                                continue
                        else:
                            raw_data = file.read().decode('utf-8', errors='ignore')
                            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw_data)
                            
                        try:
                            supabase.table("documents").insert({
                                "user_id": st.session_state.user.id,
                                "file_name": file.name,
                                "summary": text
                            }).execute()
                            
                            st.session_state.processed_files.append(file.name)
                            new_text_added += f"\n--- START OF DOCUMENT: {file.name} ---\n{text}\n--- END OF DOCUMENT ---\n"
                        except Exception as e:
                            st.error(f"Database Error for {file.name}: {e}")
                            error_occurred = True
                
                if not error_occurred and new_text_added:
                    st.session_state.doc_context += new_text_added
                    st.session_state.chat_session = init_gemini()
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Documents added to vault. Context updated.",
                        "is_status": True 
                    })
                    st.rerun()

    st.divider()
    # FIX: Safe Logout
    if st.button("🚪 Logout", use_container_width=True):
        try:
            controller.remove("supabase_token")
        except KeyError:
            pass
        supabase.auth.sign_out()
        reset_ui_state()
        st.rerun()

# --- 4. Main Chat Interface ---
st.title("🧠 Paper Brain Agent")

if st.session_state.processed_files:
    cols = st.columns(len(st.session_state.processed_files))
    for i, file_name in enumerate(st.session_state.processed_files):
        if cols[i].button(f"Summarize {file_name[:20]}...", key=f"sum_btn_{i}"):
            st.session_state.trigger_summary = file_name
    st.divider()

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and not msg.get("is_status"):
            btn_col1, btn_col2, _ = st.columns([0.1, 0.1, 0.8])
            with btn_col1:
                st.download_button(label="📥", data=msg["content"], file_name=f"Chat_{i}.md", mime="text/markdown", key=f"dl_{i}")
            with btn_col2:
                with st.popover("📋"):
                    st.code(msg["content"], language="markdown")

# --- 5. Logic Router: Summaries ---
if "trigger_summary" in st.session_state:
    target_file = st.session_state.trigger_summary
    del st.session_state.trigger_summary 
    
    prompt = f"Summarize '{target_file}' using First-Principles logic. Define technical terms inline."
    
    if not st.session_state.current_thread_id:
        title = f"Summary: {target_file}"[:50]
        res = supabase.table("chat_threads").insert({"user_id": st.session_state.user.id, "title": title}).execute()
        st.session_state.current_thread_id = res.data[0]["id"]
        
    st.session_state.messages.append({"role": "user", "content": f"Summarize `{target_file}`"})
    supabase.table("chat_messages").insert({"thread_id": st.session_state.current_thread_id, "role": "user", "content": f"Summarize `{target_file}`"}).execute()
    
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_session.send_message_stream(prompt)
        full_response = st.write_stream((chunk.text for chunk in response_stream))
        
    supabase.table("chat_messages").insert({"thread_id": st.session_state.current_thread_id, "role": "assistant", "content": full_response}).execute()
    st.session_state.messages.append({"role": "assistant", "content": full_response, "is_status": False})
    st.rerun()

# --- 6. Chat Input & Persistent Storage ---
if user_input := st.chat_input("Ask about the documents..."):
    if not st.session_state.current_thread_id:
        title = user_input[:40] + "..." if len(user_input) > 40 else user_input
        res = supabase.table("chat_threads").insert({"user_id": st.session_state.user.id, "title": title}).execute()
        st.session_state.current_thread_id = res.data[0]["id"]

    supabase.table("chat_messages").insert({"thread_id": st.session_state.current_thread_id, "role": "user", "content": user_input}).execute()
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_session.send_message_stream(user_input)
        full_response = st.write_stream((chunk.text for chunk in response_stream))
        
    supabase.table("chat_messages").insert({"thread_id": st.session_state.current_thread_id, "role": "assistant", "content": full_response}).execute()
    st.session_state.messages.append({"role": "assistant", "content": full_response, "is_status": False})
    st.rerun()