import os
import re
import time
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

# --- Robust Cookie Catching ---
time.sleep(0.15) 
stored_token = controller.get("supabase_token")

if stored_token and "user" not in st.session_state:
    try:
        supabase.auth.set_session(stored_token, stored_token)
        res = supabase.auth.get_user(stored_token)
        st.session_state.user = res.user
    except Exception:
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
    """Syncs current messages to Gemini's history & context."""
    
    # SHIVAM'S FIX: The Ironclad Guardrail Prompt
    system_instruction = (
        "You are 'Paper Brain', an elite academic research assistant. "
        "YOUR STRICT DIRECTIVE: You must ONLY answer questions related to academic research, science, methodology, literature reviews, and the uploaded documents. "
        "If the user asks for recipes, movie reviews, coding help (unless it is data-science related to a paper), or general chit-chat, you must firmly refuse. State that you are a dedicated research assistant and ask them to provide a paper. "
        "Always rely strictly on the provided text. Use strict LaTeX: $x$ for inline, $$x$$ for blocks."
    )
    
    if st.session_state.doc_context:
        system_instruction += (
            f"\n\nVAULT CONTENTS:\n{st.session_state.doc_context}"
        )

    gemini_history = []
    for m in st.session_state.messages:
        if not m.get("is_status"): 
            role = "user" if m["role"] == "user" else "model"
            gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
            
    return st.session_state.client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1  # SHIVAM'S FIX: Drops creativity to near-zero for factual rigor
        ),
        history=gemini_history
    )

def reset_ui_state():
    st.session_state.messages = []
    st.session_state.doc_context = ""
    st.session_state.chat_session = None
    st.session_state.processed_files = []
    st.session_state.docs_loaded = False
    st.session_state.current_thread_id = None

def format_chat_title(raw_text):
    clean_text = raw_text.replace('.pdf', '').replace('.txt', '').replace('_', ' ').strip()
    return (clean_text[:25] + '...') if len(clean_text) > 25 else clean_text

# --- 🚨 THE BOUNCER (LOGIN SCREEN) 🚨 ---
if not st.session_state.get("user"):
    st.title("🔐 Welcome to Paper Brain")
    st.markdown("Your personal, persistent research vault.")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            log_email = st.text_input("Email", key="log_email")
            log_password = st.text_input("Password", type="password", key="log_pass")
            submit_login = st.form_submit_button("Login")
            
            if submit_login:
                try:
                    response = supabase.auth.sign_in_with_password({"email": log_email, "password": log_password})
                    st.session_state.user = response.user
                    controller.set("supabase_token", response.session.access_token)
                    st.rerun()
                except Exception as e:
                    st.error(f"Login Failed: {str(e)}")
                
    with tab2:
        with st.form("signup_form"):
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_pass")
            submit_signup = st.form_submit_button("Create Account")
            
            if submit_signup:
                try:
                    supabase.auth.sign_up({"email": reg_email, "password": reg_password})
                    st.success("✅ Account created! Check your email for verification, or log in if email verification is turned off.")
                except Exception as e:
                    st.error(f"Signup Error: {str(e)}")
    st.stop()

# --- Blank Slate on Login ---
if st.session_state.user and not st.session_state.docs_loaded:
    st.session_state.chat_session = init_gemini()
    st.session_state.docs_loaded = True

# --- 3. Sidebar: Sleek ChatGPT UI ---
with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        reset_ui_state()
        st.session_state.docs_loaded = True 
        st.session_state.chat_session = init_gemini()
        st.rerun()
        
    st.divider()

    st.markdown("### 💬 Past Chats")
    try:
        threads = supabase.table("chat_threads").select("*").eq("user_id", st.session_state.user.id).order("created_at", desc=True).execute().data
        if threads:
            for t in threads:
                btn_type = "secondary" if t['id'] == st.session_state.current_thread_id else "tertiary"
                if st.button(t['title'], key=f"thread_{t['id']}", use_container_width=True, type=btn_type):
                    st.session_state.current_thread_id = t['id']
                    
                    msgs = supabase.table("chat_messages").select("*").eq("thread_id", t['id']).order("created_at", desc=False).execute().data
                    st.session_state.messages = [{"role": m["role"], "content": m["content"], "is_status": False} for m in msgs]
                    
                    docs = supabase.table("documents").select("*").eq("thread_id", t['id']).execute().data
                    combined_text = ""
                    file_names = []
                    if docs:
                        for doc in docs:
                            file_names.append(doc["file_name"])
                            combined_text += f"\n--- START OF DOCUMENT: {doc['file_name']} ---\n{doc['summary']}\n--- END OF DOCUMENT ---\n"
                    
                    st.session_state.doc_context = combined_text
                    st.session_state.processed_files = file_names
                    st.session_state.chat_session = init_gemini()
                    st.rerun()
        else:
            st.caption("No past chats yet.")
    except Exception:
        pass
        
    st.divider()
    
    st.markdown("### 📄 Attached Papers")
    if st.session_state.processed_files:
        for f in st.session_state.processed_files:
            st.caption(f"✅ {f}")
    else:
        st.caption("No papers attached to this chat.")
        
    st.divider()
    
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
    st.markdown("### 🛠️ Researcher's Toolkit")
    
    for i, file_name in enumerate(st.session_state.processed_files):
        st.caption(f"**Target:** {file_name}")
        col1, col2, col3, _ = st.columns([0.2, 0.2, 0.2, 0.4])
        
        if col1.button("📑 Summarize", key=f"sum_{i}", use_container_width=True):
            st.session_state.trigger_action = {"file": file_name, "type": "summary"}
        if col2.button("📖 Glossary", key=f"glos_{i}", use_container_width=True):
            st.session_state.trigger_action = {"file": file_name, "type": "glossary"}
        if col3.button("⚖️ Critique", key=f"crit_{i}", use_container_width=True):
            st.session_state.trigger_action = {"file": file_name, "type": "critique"}
            
    if len(st.session_state.processed_files) > 1:
        st.markdown("#### 🔗 Multi-Paper Synthesis")
        if st.button("📊 Compare Attached Papers", use_container_width=True, type="primary"):
            st.session_state.trigger_action = {
                "file": ", ".join(st.session_state.processed_files), 
                "type": "compare"
            }
            
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
                    
    # The Regenerate Button for Orphaned User Messages
    if i == len(st.session_state.messages) - 1 and msg["role"] == "user":
        if st.button("🔄 Regenerate Response", key="regen_btn"):
            st.session_state.trigger_action = {"file": None, "type": "regenerate", "query": msg["content"]}
            st.rerun()

# --- 5. Logic Router: Automated Tools & Regeneration ---
if "trigger_action" in st.session_state:
    target_file = st.session_state.trigger_action.get("file")
    action_type = st.session_state.trigger_action["type"]
    user_msg = ""
    prompt = ""
    
    if action_type == "summary":
        user_msg = f"Summarize `{target_file}`"
        prompt = f"Summarize '{target_file}' using First-Principles logic. Define technical terms inline. Break down the core methodology."
    elif action_type == "glossary":
        user_msg = f"Generate a glossary for `{target_file}`"
        prompt = f"Extract the top 10-15 most complex technical terms, acronyms, and mathematical variables used in '{target_file}'. Output them as a highly readable Markdown dictionary with simple, first-principles definitions."
    elif action_type == "critique":
        user_msg = f"Critique `{target_file}`"
        prompt = f"Act as a peer reviewer for '{target_file}'. Identify the core assumptions, potential biases, methodological limitations, and what future research would be needed to prove or disprove the claims."
    elif action_type == "compare":
        user_msg = f"Compare the attached papers: {target_file}"
        prompt = f"""You are analyzing the following attached papers: {target_file}.

STEP 1: THE DOMAIN CHECK
First, rigorously evaluate if these papers belong to the same general research domain. 
If they are fundamentally unrelated, STATE THIS CLEARLY. Briefly explain what each focuses on and why comparing their methodology is invalid.

STEP 2: THE COMPARISON MATRIX
If they are related, synthesize them into a side-by-side Markdown comparison table.
Rule 1: Use First-Principles logic. No unexplained jargon.
Rule 2: Be highly concise.

The table MUST include these exact rows:
- The "Big Idea" (Core Hypothesis)
- The Methodology (How did they test it?)
- Data & Scale (What did they measure?)
- The Catch (Assumptions & Limitations)

After the table, conclude which paper has stronger evidence or how they complement each other."""
    elif action_type == "regenerate":
        prompt = st.session_state.trigger_action["query"]
        
    del st.session_state.trigger_action 
    
    if action_type != "regenerate":
        if not st.session_state.current_thread_id:
            title = format_chat_title(f"{action_type.capitalize()}: {target_file}")
            res = supabase.table("chat_threads").insert({"user_id": st.session_state.user.id, "title": title}).execute()
            st.session_state.current_thread_id = res.data[0]["id"]
            
        st.session_state.messages.append({"role": "user", "content": user_msg})
        supabase.table("chat_messages").insert({"thread_id": st.session_state.current_thread_id, "role": "user", "content": user_msg}).execute()
    
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_session.send_message_stream(prompt)
        full_response = st.write_stream((chunk.text for chunk in response_stream))
        
    supabase.table("chat_messages").insert({"thread_id": st.session_state.current_thread_id, "role": "assistant", "content": full_response}).execute()
    st.session_state.messages.append({"role": "assistant", "content": full_response, "is_status": False})
    st.rerun()

# --- 6. Chat Input & Persistent Storage (The Magic Box) ---
if user_input := st.chat_input("Ask about papers or attach new ones...", accept_file="multiple", file_type=["pdf", "txt"]):
    
    text_query = getattr(user_input, 'text', user_input if isinstance(user_input, str) else "")
    uploaded_files = getattr(user_input, 'files', [])

    if not st.session_state.current_thread_id:
        raw_title = text_query if text_query else (uploaded_files[0].name if uploaded_files else "New Research Chat")
        title = format_chat_title(raw_title)
        res = supabase.table("chat_threads").insert({"user_id": st.session_state.user.id, "title": title}).execute()
        st.session_state.current_thread_id = res.data[0]["id"]

    new_text_added = ""
    error_occurred = False

    if uploaded_files:
        with st.spinner("Extracting and attaching documents..."):
            for file in uploaded_files:
                if file.name not in st.session_state.processed_files:
                    text = ""
                    if file.name.lower().endswith('.pdf'):
                        try:
                            doc = fitz.open(stream=file.read(), filetype="pdf")
                            for page in doc:
                                raw_text = page.get_text()
                                text += re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw_text) + "\n"
                                
                            if len(text.strip()) < 50:
                                st.warning(f"⚠️ '{file.name}' appears to be a scanned image or presentation. Paper Brain currently requires text-based PDFs.")
                                error_occurred = True
                                continue 
                                
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
                            "thread_id": st.session_state.current_thread_id,
                            "file_name": file.name,
                            "summary": text
                        }).execute()
                        
                        st.session_state.processed_files.append(file.name)
                        new_text_added += f"\n--- START OF DOCUMENT: {file.name} ---\n{text}\n--- END OF DOCUMENT ---\n"
                    except Exception as e:
                        st.error(f"Database Error: {e}")
                        error_occurred = True
            
            if new_text_added and not error_occurred:
                st.session_state.doc_context += new_text_added
                st.session_state.chat_session = init_gemini()
                
                if not text_query:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"✅ Successfully attached {len(uploaded_files)} document(s). You can ask a specific question, or use the **Researcher's Toolkit** buttons above to get started.",
                        "is_status": True 
                    })

    if text_query:
        supabase.table("chat_messages").insert({"thread_id": st.session_state.current_thread_id, "role": "user", "content": text_query}).execute()
        st.session_state.messages.append({"role": "user", "content": text_query})
        
        with st.chat_message("user"):
            st.markdown(text_query)
            
        with st.chat_message("assistant"):
            response_stream = st.session_state.chat_session.send_message_stream(text_query)
            full_response = st.write_stream((chunk.text for chunk in response_stream))
            
        supabase.table("chat_messages").insert({"thread_id": st.session_state.current_thread_id, "role": "assistant", "content": full_response}).execute()
        st.session_state.messages.append({"role": "assistant", "content": full_response, "is_status": False})
        
    st.rerun()