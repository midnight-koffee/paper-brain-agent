import os
import re
import time
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
import fitz  # PyMuPDF
from supabase import create_client, Client
from streamlit_cookies_controller import CookieController

# Load `.env` from this file's folder (works even if Streamlit's cwd is elsewhere)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)
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

# --- ROBUST COOKIE CATCHING (Retry Loop) ---
stored_token = None
for _ in range(3):
    time.sleep(0.15) 
    stored_token = controller.get("supabase_token")
    if stored_token:
        break

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
if "interaction_count" not in st.session_state:
    st.session_state.interaction_count = 0
if "key_index" not in st.session_state:
    st.session_state.key_index = 0

# Teacher Mode State Dictionary
if "teacher" not in st.session_state:
    st.session_state.teacher = {
        "active": False,
        "paper": None,
        "turn": 1,
        "score": 0,
        "total_turns": 5,
        "last_question": "",
        "syllabus": [] 
    }

# --- Utility Functions ---
def check_rate_limit():
    if st.session_state.interaction_count >= 50:
        st.error("⚠️ **Trial Limit Reached:** You have exceeded 50 AI interactions for this session. Please start a new session or upgrade to Pro later.")
        st.stop()
    st.session_state.interaction_count += 1

def _normalize_api_key(raw: str) -> str:
    k = raw.strip().strip("\ufeff")
    if len(k) >= 2 and ((k[0] == k[-1] == '"') or (k[0] == k[-1] == "'")):
        k = k[1:-1].strip()
    return k


def collect_gemini_api_keys() -> list[str]:
    keys: list[str] = []
    multi = os.environ.get("GEMINI_API_KEYS", "")
    if multi.strip():
        for part in multi.split(","):
            nk = _normalize_api_key(part)
            if nk:
                keys.append(nk)
    if not keys:
        for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            v = os.environ.get(env_name)
            if v:
                nk = _normalize_api_key(v)
                if nk:
                    keys.append(nk)
                    break
    return keys


def gemini_key_missing_message() -> str:
    return (
        "No Gemini API key found. In the project folder, add to `.env`:\n\n"
        "`GEMINI_API_KEY=your_key`\n\n"
        "Create a key at https://aistudio.google.com/apikey — then **restart** Streamlit."
    )


def is_invalid_api_key_error(exc: BaseException) -> bool:
    """Detect Google API_KEY_INVALID / 'API key not valid' (SDK may put details only on the exception object)."""
    parts = [str(exc), repr(exc)]
    try:
        parts.append(str(getattr(exc, "message", "") or ""))
        parts.append(str(getattr(exc, "details", "") or ""))
    except Exception:
        pass
    blob = " ".join(parts).lower()
    return (
        "api_key_invalid" in blob
        or "api key not valid" in blob
        or "invalid api key" in blob
        or ("invalid_argument" in blob and "400" in blob and "key" in blob)
    )


def invalid_api_key_user_message() -> str:
    return (
        "**Google rejected your Gemini API key** (wrong, expired, revoked, or restricted).\n\n"
        "1. Open [Google AI Studio](https://aistudio.google.com/apikey) and create a **new** API key.\n"
        "2. Put it in your project `.env` as `GEMINI_API_KEY=...` (no spaces around `=`).\n"
        "3. **Restart** Streamlit (stop the process, then run `streamlit run main.py` again).\n\n"
        "If the key is new: in Google Cloud Console, ensure **Generative Language API** is allowed for that key, "
        "and that **Application restrictions** are not blocking your machine (use “None” for testing)."
    )


def get_rotated_client():
    keys = collect_gemini_api_keys()
    if not keys:
        st.error(gemini_key_missing_message())
        st.stop()

    selected_key = keys[st.session_state.key_index % len(keys)]
    st.session_state.key_index += 1
    return genai.Client(api_key=selected_key)

def init_gemini():
    # Force a key rotation every time we initialize a new chat session
    st.session_state.client = get_rotated_client()
    
    system_instruction = (
        "You are 'Paper Brain', an elite academic research assistant and strict but encouraging tutor. "
        "YOUR STRICT DIRECTIVE: You must ONLY answer questions related to academic research, science, methodology, literature reviews, and the uploaded documents. "
        "COGNITIVE STYLE RULES: "
        "1. Be concise but highly comprehensive. Do NOT dumb down the science or skip technical details. "
        "2. Express ideas in clear, simple bullet points. Avoid walls of text. "
        "3. Translate dense jargon into first-principles intuition. "
        "4. Always use standard structural emojis (🧠, ⚙️, 🧩, 📌) to organize your thoughts. "
        "Always rely strictly on the provided text. Use strict LaTeX: $x$ for inline, $$x$$ for blocks."
    )
    
    if st.session_state.doc_context:
        system_instruction += f"\n\nVAULT CONTENTS:\n{st.session_state.doc_context}"

    gemini_history = []
    for m in st.session_state.messages:
        if not m.get("is_status"): 
            role = "user" if m["role"] == "user" else "model"
            gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
            
    return st.session_state.client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1  
        ),
        history=gemini_history
    )

if "client" not in st.session_state:
    st.session_state.client = get_rotated_client()

def reset_ui_state():
    st.session_state.messages = []
    st.session_state.doc_context = ""
    st.session_state.chat_session = None
    st.session_state.processed_files = []
    st.session_state.docs_loaded = False
    st.session_state.current_thread_id = None
    st.session_state.teacher = {
        "active": False, "paper": None, "turn": 1, "score": 0, "total_turns": 5, "last_question": "", "syllabus": []
    }

def format_chat_title(raw_text):
    clean_text = raw_text.replace('.pdf', '').replace('.txt', '').replace('_', ' ').strip()
    return (clean_text[:25] + '...') if len(clean_text) > 25 else clean_text

# --- MULTI-FILE ISOLATION ENGINE ---
def truncate_vault_for_planning(doc_context: str, max_chars: int = 15000) -> str:
    if len(doc_context) <= max_chars:
        return doc_context
    return doc_context[:max_chars] + "\n\n[... text truncated for planning analysis ...]"

def _vault_sections(doc_context: str) -> list[tuple[str, str]]:
    if not doc_context:
        return []
    pattern = r"--- START OF DOCUMENT: (.+?) ---\s*(.*?)\s*--- END OF DOCUMENT ---"
    return re.findall(pattern, doc_context, flags=re.DOTALL)

def extract_document_from_vault(doc_context: str, file_name: str) -> str | None:
    if not file_name or not doc_context:
        return None
    want = file_name.strip()
    for name, body in _vault_sections(doc_context):
        n = name.strip()
        if n == want or n.lower() == want.lower():
            return body.strip()
    return None

def list_vault_filenames(doc_context: str) -> list[str]:
    return [name.strip() for name, _ in _vault_sections(doc_context)]

def format_authoritative_source_block(file_name: str, body: str, max_chars: int = 28000) -> str:
    text = body if len(body) <= max_chars else body[:max_chars] + "\n\n[... truncated ...]"
    return (
        f"=== AUTHORITATIVE SOURCE — '{file_name}' ONLY ===\n"
        f"Use ONLY this excerpt for facts about this file. Ignore other papers in the session vault.\n\n"
        f"{text}\n"
        f"=== END SOURCE — '{file_name}' ==="
    )

def build_compare_source_block(doc_context: str, comma_separated_names: str) -> str | None:
    names = [n.strip() for n in comma_separated_names.split(",") if n.strip()]
    parts: list[str] = []
    for n in names:
        body = extract_document_from_vault(doc_context, n)
        if body is None:
            return None
        parts.append(format_authoritative_source_block(n, body, max_chars=18000))
    return "\n\n".join(parts)

def process_uploaded_files(uploaded_files):
    new_text_added = ""
    error_occurred = False
    
    if not st.session_state.current_thread_id:
        title = format_chat_title(uploaded_files[0].name)
        res = supabase.table("chat_threads").insert({"user_id": st.session_state.user.id, "title": title}).execute()
        st.session_state.current_thread_id = res.data[0]["id"]

    with st.spinner("Extracting and attaching documents..."):
        for file in uploaded_files:
            if file.name not in st.session_state.processed_files:
                text = ""
                if file.name.lower().endswith('.pdf'):
                    try:
                        doc = fitz.open(stream=file.read(), filetype="pdf")
                        if doc.page_count > 50:
                            st.error(f"⚠️ '{file.name}' is too long ({doc.page_count} pages). Please upload a focused research paper (max 50 pages).")
                            error_occurred = True
                            continue
                        for page in doc:
                            raw_text = page.get_text()
                            text += re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw_text) + "\n"
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {str(e)}")
                        error_occurred = True
                        continue
                else:
                    raw_data = file.read().decode('utf-8', errors='ignore')
                    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw_data)

                try:
                    supabase.table("documents").insert({
                        "user_id": st.session_state.user.id, "thread_id": st.session_state.current_thread_id, "file_name": file.name, "summary": text
                    }).execute()
                    st.session_state.processed_files.append(file.name)
                    new_text_added += f"\n--- START OF DOCUMENT: {file.name} ---\n{text}\n--- END OF DOCUMENT ---\n"
                except Exception as e:
                    st.error(f"Database Error: {e}")
                    error_occurred = True
        
        if new_text_added and not error_occurred:
            st.session_state.doc_context += new_text_added
            st.session_state.chat_session = init_gemini()
            st.session_state.messages.append({
                "role": "assistant", "content": f"✅ Attached {len(uploaded_files)} document(s). Select a tool above to begin.", "is_status": True 
            })
            st.rerun()
        elif error_occurred:
            st.rerun()

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
                    
                    st.session_state.teacher = {
                        "active": False, "paper": None, "turn": 1, "score": 0, "total_turns": 5, "last_question": "", "syllabus": []
                    }
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
    
    # Teacher Mode Controls & Pinned Quick Actions
    if st.session_state.teacher.get("active"):
        st.markdown("### 🎓 Tutor Quick Actions")
        if st.button("💡 Give me a hint", use_container_width=True):
            st.session_state.queued_command = "hint"
            st.rerun()
        if st.button("🧠 Explain Simpler", use_container_width=True):
            st.session_state.queued_command = "explain simpler"
            st.rerun()
        if st.button("⏭️ Skip Question", use_container_width=True):
            st.session_state.queued_command = "skip"
            st.rerun()
            
        st.divider()
        
        if st.button("🛑 Exit Teacher Mode", use_container_width=True):
            st.session_state.teacher["active"] = False
            st.session_state.messages.append({"role": "assistant", "content": "Teacher Mode deactivated. We are back to normal chat.", "is_status": True})
            st.rerun()
            
        paper = st.session_state.teacher.get("paper")
        if paper and st.button("🔄 Restart Teacher Mode", use_container_width=True):
            st.session_state.teacher = {
                "active": False, "paper": None, "turn": 1, "score": 0, "total_turns": 5, "last_question": "", "syllabus": []
            }
            st.session_state.trigger_action = {"file": paper, "type": "teach"}
            st.rerun()
            
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

# UX FIX 3: The Big Dropzone (Empty State)
if not st.session_state.processed_files and not st.session_state.messages:
    st.markdown("### 📄 Welcome to Paper Brain")
    st.markdown("Upload a research paper to automatically extract its core domain, generate summaries, or enter interactive Teacher Mode.")
    big_upload = st.file_uploader("Drag and drop PDFs here to begin", type=["pdf", "txt"], accept_multiple_files=True)
    if big_upload:
        process_uploaded_files(big_upload)

# Visual Progress Bar UI
if st.session_state.teacher.get("active"):
    turn = st.session_state.teacher['turn']
    total = st.session_state.teacher['total_turns']
    score = st.session_state.teacher['score']
    syllabus = st.session_state.teacher.get('syllabus', [])
    current_topic = syllabus[turn - 1] if turn <= len(syllabus) else "Summary"
    
    progress_val = min(turn / total, 1.0)
    st.progress(progress_val)
    st.info(f"🎓 **Teacher Mode** | Part {turn} of {total}: **{current_topic}** | **Score:** {score}")

if st.session_state.processed_files:
    if not st.session_state.teacher.get("active"):
        st.markdown("### 🛠️ Researcher's Toolkit")
        
        for i, file_name in enumerate(st.session_state.processed_files):
            st.caption(f"**Target:** {file_name}")
            col1, col2, col3, col4 = st.columns([0.22, 0.22, 0.22, 0.34])
            
            if col1.button("📑 Summarize", key=f"sum_{i}", use_container_width=True):
                st.session_state.trigger_action = {"file": file_name, "type": "summary"}
            if col2.button("📖 Glossary", key=f"glos_{i}", use_container_width=True):
                st.session_state.trigger_action = {"file": file_name, "type": "glossary"}
            if col3.button("⚖️ Critique", key=f"crit_{i}", use_container_width=True):
                st.session_state.trigger_action = {"file": file_name, "type": "critique"}
            if col4.button("🎓 Teach Me", key=f"teach_{i}", use_container_width=True, type="primary"):
                st.session_state.trigger_action = {"file": file_name, "type": "teach"}
                
        if len(st.session_state.processed_files) > 1:
            st.markdown("#### 🔗 Multi-Paper Synthesis")
            if st.button("📊 Compare Attached Papers", use_container_width=True, type="primary"):
                st.session_state.trigger_action = {
                    "file": ", ".join(st.session_state.processed_files), 
                    "type": "compare"
                }
            
    st.divider()

# Display chat history with UI Polish
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # Hide secret tokens
        clean_msg = msg["content"].replace("[CORRECT]", "").replace("[SKIP]", "").replace("[END_TEACHING]", "").replace("**[CORRECT]**", "")
        
        if msg["role"] == "assistant" and "### 📖 Concept" in clean_msg:
            st.info(clean_msg, icon="🎓")
        elif msg["role"] == "assistant" and "### ✅ Verdict" in clean_msg:
            if "Incorrect" in clean_msg or "Hint" in clean_msg or "Partial" in clean_msg:
                st.warning(clean_msg, icon="❌")
            else:
                st.success(clean_msg, icon="✅")
        else:
            st.markdown(clean_msg)
            
        if msg["role"] == "assistant" and not msg.get("is_status"):
            btn_col1, btn_col2, _ = st.columns([0.1, 0.1, 0.8])
            with btn_col1:
                st.download_button(label="📥", data=clean_msg, file_name=f"Chat_{i}.md", mime="text/markdown", key=f"dl_{i}")
            with btn_col2:
                with st.popover("📋"):
                    st.code(clean_msg, language="markdown")
                    
    if i == len(st.session_state.messages) - 1 and msg["role"] == "user" and not st.session_state.teacher.get("active"):
        if st.button("🔄 Regenerate Response", key="regen_btn"):
            st.session_state.trigger_action = {"file": None, "type": "regenerate", "query": msg["content"]}
            st.rerun()

# --- 5. Logic Router: Automated Tools & Regeneration ---
if "trigger_action" in st.session_state:
    target_file = st.session_state.trigger_action.get("file")
    action_type = st.session_state.trigger_action["type"]
    user_msg = ""
    prompt = ""

    scoped_paper_body: str | None = None
    if action_type in ("summary", "glossary", "critique", "teach"):
        if not target_file:
            del st.session_state.trigger_action
            st.error("Missing target document for this action.")
            st.stop()
        scoped_paper_body = extract_document_from_vault(st.session_state.doc_context, target_file)
        if scoped_paper_body is None:
            known = list_vault_filenames(st.session_state.doc_context)
            del st.session_state.trigger_action
            st.error(
                f"Could not find `{target_file}` in the vault. "
                f"Attached in this chat: {', '.join(known) if known else 'none'}."
            )
            st.stop()
    
    if action_type == "summary":
        user_msg = f"Summarize `{target_file}`"
        prompt = f"""You are an elite, adaptive research summarizer analyzing '{target_file}'.

STEP 1: Identify the academic domain.
STEP 2: Dynamically select 3 to 5 structural headings that perfectly fit this specific domain's logical flow.
STEP 3: Generate a comprehensive, rigorous summary.

COGNITIVE STYLE & RIGOR RULES:
- Do NOT "dumb down" the science, math, or methodology. Be exhaustive and precise, but explain it using first-principles logic.
- Define technical terms the first time you use them.
- Use bullet points. Absolutely no walls of text.

FORMAT STRICTLY USING MARKDOWN:
### 📚 Domain: [Identified Domain]

### 🧠 The Big Idea & Context
*(Pointwise: What specific problem does this solve? What is the core hypothesis?)*

### [Relevant Emoji] [Dynamic Heading 1]
*(Rigorous, detailed, comprehensive breakdown)*

### [Relevant Emoji] [Dynamic Heading 2]
*(Rigorous, detailed, comprehensive breakdown)*

*(... add more dynamic headings if necessary to capture all crucial technical details ...)*

### ⚠️ Nuance & Limitations
*(Theoretical boundaries, assumptions, or gaps in this paper)*

### 🧩 Memory Hook
*(A concrete mental model or analogy to anchor this specific paper's contribution)*"""
        prompt += "\n\n" + format_authoritative_source_block(target_file, scoped_paper_body)

    elif action_type == "glossary":
        user_msg = f"Generate a glossary for `{target_file}`"
        prompt = f"""You are an adaptive terminology expert analyzing '{target_file}'.

STEP 1: Identify the academic domain of the paper.
STEP 2: Extract the 10-15 most critical technical terms, acronyms, or mathematical variables used.
STEP 3: Define them in simple, first-principles language without losing technical rigor.

FORMAT STRICTLY USING MARKDOWN:
### 📚 Domain: [Identified Domain]

* **[Term 1]**: [Simple but rigorous definition]. *Example/Context: [Brief explanation of how it is used in the paper].*
* **[Term 2]**: ...
"""
        prompt += "\n\n" + format_authoritative_source_block(target_file, scoped_paper_body)

    elif action_type == "critique":
        user_msg = f"Critique `{target_file}`"
        prompt = f"""Act as a ruthless but constructive peer reviewer analyzing '{target_file}'.

STEP 1: Identify the research domain and paper type (e.g., Experimental Study, Review, Theoretical Proof).
STEP 2: Evaluate the paper using the strict standards of THAT specific domain (e.g., mathematical rigor for math, dataset bias for ML, sample size for clinical).
STEP 3: Generate the critique.

FORMAT STRICTLY USING MARKDOWN:
### 📚 Domain & Paper Type: [Identified]

### 🧭 Core Claims
*(What are they actually claiming? Restate simply.)*

### 🔍 Hidden Assumptions & Biases
*(List 2-4 hidden assumptions the authors make. Are they justified?)*

### ⚖️ Methodological Soundness
*(Evaluate the proofs, experiments, or logic based on the specific domain standards)*

### 🧪 Evidence Strength
*(Verdict: Weak / Moderate / Strong - and briefly explain why)*

### 🔭 Future Work & Missing Pieces
*(What breaks this paper? What exact experiment or analysis needs to be done next?)*"""
        prompt += "\n\n" + format_authoritative_source_block(target_file, scoped_paper_body)
    
    elif action_type == "teach":
        user_msg = f"🎓 Start Teacher Mode for `{target_file}`."
        
        with st.spinner("📚 Analyzing paper to build a custom syllabus..."):
            check_rate_limit()
            vault_snip = truncate_vault_for_planning(scoped_paper_body)
            syllabus_prompt = (
                f"The excerpt below is the FULL stored text for '{target_file}' ONLY. "
                f"Analyze THIS paper. Identify its academic domain, then return a JSON array of exactly 5 logical "
                f"section titles tailored to teach THIS paper step-by-step. "
                f'Example: ["Core Problem", "Methodology", "Key Results", "Limitations", "Big Picture"]. '
                f"Output ONLY a valid JSON array and no other text."
            )
            
            try:
                # Direct API call to generate JSON string, not part of chat session history
                res = st.session_state.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"SINGLE-PAPER EXCERPT FOR '{target_file}':\n{vault_snip}\n\n{syllabus_prompt}",
                )
                raw_json = res.text.strip().replace('```json', '').replace('```', '')
                extracted_syllabus = json.loads(raw_json)
                if not isinstance(extracted_syllabus, list) or len(extracted_syllabus) < 3:
                    extracted_syllabus = ["Core Problem", "Methodology", "Key Results", "Limitations", "Big Picture"]
            except Exception as e:
                if is_invalid_api_key_error(e):
                    del st.session_state.trigger_action
                    st.error(invalid_api_key_user_message())
                    st.stop()
                if "429" in str(e) or "Resource" in str(e) or "quota" in str(e).lower():
                    st.session_state.client = get_rotated_client()
                    st.error(
                        "⚠️ Rate limit while building the syllabus. Switched API key. "
                        "Try **Teach Me** again in a few seconds."
                    )
                else:
                    st.error(f"⚠️ Failed to generate syllabus: {str(e)}")
                del st.session_state.trigger_action
                st.stop()
                
        total_turns = len(extracted_syllabus)
        st.session_state.teacher.update({
            "active": True, "paper": target_file, "turn": 1, "score": 0, 
            "last_question": "", "syllabus": extracted_syllabus, "total_turns": total_turns
        })
        
        first_topic = extracted_syllabus[0]
        lesson_source = format_authoritative_source_block(target_file, scoped_paper_body, max_chars=32000)
        
        prompt = f"""You are initiating TEACHER MODE for the paper '{target_file}' ONLY.
Other papers may exist in the session; ignore them. The syllabus below was built from '{target_file}' alone.
Syllabus: {extracted_syllabus}
Current turn: 1 — topic: {first_topic}.

{lesson_source}

YOU MUST FORMAT YOUR RESPONSE EXACTLY LIKE THIS USING MARKDOWN:
### 📖 Concept 1: {first_topic}
**🧠 Intuition:**
*(Bullet points explaining it comprehensively but simply. Do not skip rigorous details.)*

**⚙️ How it Works:**
*(Technical breakdown. Use math or formal logic if present in the text.)*

**🧩 Mnemonic:**
*(A clever memory trick)*

> **📌 Key Takeaway:** *(One sentence summarizing the concept)*

---
### 🧪 Knowledge Check
Please answer with three letters (e.g., A C B).

**Q1. [Question Text]**
* A) 
* B) 
* C) 
* D) 

**Q2. [Question Text]**
* A) 
* B) 
* C) 
* D) 

**Q3. [Question Text]**
* A) 
* B) 
* C) 
* D) 

Stop generating after the questions."""
    
    elif action_type == "compare":
        compare_sources = build_compare_source_block(st.session_state.doc_context, target_file)
        if compare_sources is None:
            known = list_vault_filenames(st.session_state.doc_context)
            del st.session_state.trigger_action
            st.error(
                f"Could not load all papers for comparison. "
                f"Expected names from the button to match vault files. "
                f"Attached: {', '.join(known) if known else 'none'}."
            )
            st.stop()
            
        user_msg = f"Compare the attached papers: {target_file}"
        prompt = f"""You are comparing ONLY the papers named in the source blocks below.

STEP 1: THE DOMAIN CHECK
First, rigorously evaluate if these papers belong to the same general research domain. 
If they are fundamentally unrelated, STATE THIS CLEARLY.

STEP 2: THE COMPARISON MATRIX
If they are related, synthesize them into a side-by-side Markdown comparison table. Include: The "Big Idea", Methodology, Data & Scale, The Catch. Be concise.

{compare_sources}"""

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
    
    if st.session_state.chat_session is None:
        st.session_state.chat_session = init_gemini()

    with st.chat_message("assistant"):
        check_rate_limit()
        try:
            response_stream = st.session_state.chat_session.send_message_stream(prompt)
            full_response = st.write_stream((chunk.text for chunk in response_stream))
        except Exception as e:
            if is_invalid_api_key_error(e):
                st.error(invalid_api_key_user_message())
            elif "429" in str(e) or "Resource" in str(e) or "quota" in str(e).lower():
                st.session_state.chat_session = init_gemini()
                st.error(
                    "⚠️ Rate limit reached. Switched API key and rebuilt the chat. "
                    "Wait a few seconds, then click **Regenerate Response**."
                )
            else:
                st.error(f"⚠️ The AI hit an error. Try **Regenerate Response**. (Details: {str(e)})")
            st.stop()
            
    if action_type == "teach":
        st.session_state.teacher["last_question"] = full_response
        
    supabase.table("chat_messages").insert({"thread_id": st.session_state.current_thread_id, "role": "assistant", "content": full_response}).execute()
    st.session_state.messages.append({"role": "assistant", "content": full_response, "is_status": False})
    st.rerun()

# --- 6. Chat Input & Persistent Storage ---
placeholder_text = "Type your answers (e.g. A C B) or ask a question..." if st.session_state.teacher.get("active") else "Ask about papers, or attach new ones..."

user_input = st.chat_input(placeholder_text, accept_file="multiple", file_type=["pdf", "txt"])
queued_command = st.session_state.pop("queued_command", None)

if user_input or queued_command:
    
    if queued_command:
        text_query = queued_command
        uploaded_files = []
    else:
        text_query = getattr(user_input, 'text', user_input if isinstance(user_input, str) else "")
        uploaded_files = getattr(user_input, 'files', [])

    if not st.session_state.current_thread_id:
        raw_title = text_query if text_query else (uploaded_files[0].name if uploaded_files else "New Research Chat")
        title = format_chat_title(raw_title)
        res = supabase.table("chat_threads").insert({"user_id": st.session_state.user.id, "title": title}).execute()
        st.session_state.current_thread_id = res.data[0]["id"]

    if uploaded_files:
        if st.session_state.teacher.get("active"):
            st.warning("⚠️ Please exit Teacher Mode before uploading new papers to avoid confusing the syllabus.")
        else:
            process_uploaded_files(uploaded_files)

    if text_query:
        # User input goes straight to the DB and chat stream.
        supabase.table("chat_messages").insert({"thread_id": st.session_state.current_thread_id, "role": "user", "content": text_query}).execute()
        st.session_state.messages.append({"role": "user", "content": text_query})
        
        with st.chat_message("user"):
            st.markdown(text_query)

        # Evaluation Interceptor 
        if st.session_state.teacher.get("active"):
            turn = st.session_state.teacher["turn"]
            total_turns = st.session_state.teacher["total_turns"]
            paper = st.session_state.teacher["paper"]
            last_q = st.session_state.teacher["last_question"]
            syllabus = st.session_state.teacher.get("syllabus", [])
            
            if len(last_q) > 10000:
                last_q = last_q[:10000] + "\n[... earlier text truncated for evaluation context ...]"
                
            next_topic = syllabus[turn] if turn < len(syllabus) else "Final Summary"

            prompt = f"""TEACHER MODE EVALUATION - Turn {turn} of {total_turns} for '{paper}'.
PREVIOUS 3 QUESTIONS ASKED: "{last_q}"
THE STUDENT'S INPUT: "{text_query}"

TUTOR COMMAND RULES:
1. If the user asks for a 'hint', 'explain', 'example', 'why', 'repeat', 'harder', or 'easier' (or uses conversational text indicating they are confused): DO NOT grade. Provide the requested help intuitively, then repeat the SAME 3 questions.
2. If the user says 'skip': Advance to the next turn immediately. YOU MUST INCLUDE "[SKIP]" anywhere in your response so the system advances without awarding points.
3. If the input contains attempts at answers (e.g., natural language identifying A, B, C or conversational guessing), evaluate ALL 3 answers.
   - If ANY answer is INCORRECT: Explain which are wrong, give a hint, and ask them to retry. (Do NOT advance).
   - If ALL 3 answers are CORRECT: Congratulate them. YOU MUST INCLUDE THE EXACT TEXT "[CORRECT]" anywhere in your response.

IF (ALL 3 CORRECT OR SKIPPED) AND Turn < {total_turns}:
Teach the next concept: {next_topic}. 
Use bullet points, an intuition block, and a mnemonic. Do not skip rigorous details. End with exactly THREE new MCQs.

IF (ALL 3 CORRECT OR SKIPPED) AND Turn == {total_turns}:
Congratulate them. Summarize the whole paper in your own words. YOU MUST INCLUDE THE EXACT TEXT "[END_TEACHING]".

YOU MUST FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
### ✅ Verdict: (All Correct / Partial / Incorrect / Hint Provided / Skipped)
*(Brief explanation)*

*(IF ADVANCING TO NEXT TURN AND Turn < {total_turns})*
---
### 📖 Concept {turn + 1}: {next_topic}
**🧠 Intuition:**
*(Bullet points)*

**⚙️ How it Works:**
*(Technical breakdown. Be rigorous.)*

**🧩 Mnemonic:**
*(Trick)*

> **📌 Key Takeaway:** *(One sentence)*

---
### 🧪 Knowledge Check
*(3 new MCQs here: Q1, Q2, Q3)*
"""
        else:
            prompt = text_query
            
        if st.session_state.chat_session is None:
            st.session_state.chat_session = init_gemini()

        with st.chat_message("assistant"):
            check_rate_limit()
            try:
                response_stream = st.session_state.chat_session.send_message_stream(prompt)
                full_response = st.write_stream((chunk.text for chunk in response_stream))
            except Exception as e:
                if is_invalid_api_key_error(e):
                    st.error(invalid_api_key_user_message())
                elif "429" in str(e) or "Resource" in str(e) or "quota" in str(e).lower():
                    st.session_state.chat_session = init_gemini()
                    st.error(
                        "⚠️ Rate limit reached. Switched API key and rebuilt the chat. "
                        "Wait a few seconds, then click **Regenerate Response**."
                    )
                else:
                    st.error(f"⚠️ The AI hit an error. Try **Regenerate Response**. (Details: {str(e)})")
                st.stop()
            
        if st.session_state.teacher.get("active"):
            clean_response = full_response.upper().replace(" ", "").replace("*", "")
            
            if "[CORRECT]" in clean_response:
                st.session_state.teacher["score"] += 1
                st.session_state.teacher["turn"] += 1
                st.session_state.teacher["last_question"] = full_response
            elif "[SKIP]" in clean_response:
                st.session_state.teacher["turn"] += 1
                st.session_state.teacher["last_question"] = full_response
                
            if "[END_TEACHING]" in clean_response:
                st.session_state.teacher["active"] = False
                st.toast("🎉 Teacher Mode Completed!")
                
        supabase.table("chat_messages").insert({"thread_id": st.session_state.current_thread_id, "role": "assistant", "content": full_response}).execute()
        st.session_state.messages.append({"role": "assistant", "content": full_response, "is_status": False})
        
    st.rerun()