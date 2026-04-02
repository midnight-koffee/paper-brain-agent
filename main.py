import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
import fitz  # PyMuPDF

load_dotenv()

# Silence MuPDF visual/color warnings in the terminal
fitz.TOOLS.mupdf_display_errors(False)

st.set_page_config(page_title="Paper Brain", page_icon="🧠", layout="centered")

# --- 1. Session State Initialization ---
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

def reset_chat():
    """Clears the memory to prevent context poisoning."""
    st.session_state.messages = []
    st.session_state.doc_context = ""
    st.session_state.chat_session = None
    st.session_state.processed_files = []

# --- 2. Sidebar: Vault & Proactive Processing ---
with st.sidebar:
    st.header("📄 Document Vault")
    
    uploaded_files = st.file_uploader(
        "Upload PDFs to the lab", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("🗑️ Clear Memory & Start Fresh"):
        reset_chat()
        st.rerun()

    if uploaded_files:
        if st.button("Process Uploaded Papers"):
            with st.spinner("Extracting text and injecting context..."):
                combined_text = ""
                file_names = []
                
                for file in uploaded_files:
                    file_names.append(file.name)
                    if file.name.lower().endswith('.pdf'):
                        try:
                            doc = fitz.open(stream=file.read(), filetype="pdf")
                            text = ""
                            for page in doc:
                                text += page.get_text() + "\n"
                            combined_text += f"\n--- START OF DOCUMENT: {file.name} ({len(doc)} pages) ---\n{text}\n--- END OF DOCUMENT ---\n"
                        except Exception as e:
                            st.error(f"Error reading {file.name}: {str(e)}")
                    else:
                        combined_text += f"\n--- START OF DOCUMENT: {file.name} ---\n{file.read().decode('utf-8')}\n--- END OF DOCUMENT ---\n"
                
                st.session_state.doc_context = combined_text
                st.session_state.processed_files = file_names
                
                st.session_state.chat_session = st.session_state.client.chats.create(
                    model="gemini-2.5-flash",
                    config=types.GenerateContentConfig(
                        system_instruction=f"""You are 'Paper Brain', an expert research assistant. 
                        The user has uploaded documents into your memory. 
                        Always rely strictly on this text. Use strict LaTeX: $x$ for inline, $$x$$ for blocks.
                        
                        VAULT CONTENTS:
                        {st.session_state.doc_context}
                        """
                    )
                )
                
                # STATUS MESSAGE: Marked so no buttons appear on it
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Successfully processed {len(uploaded_files)} document(s). Ready for analysis.",
                    "is_status": True 
                })
                st.rerun()

# --- 3. Main Chat Interface & Dynamic Buttons ---
st.title("🧠 Paper Brain Agent")

if st.session_state.processed_files and st.session_state.chat_session:
    st.markdown("### ⚡ Quick Summaries")
    cols = st.columns(len(st.session_state.processed_files))
    
    for i, file_name in enumerate(st.session_state.processed_files):
        if cols[i].button(f"Summarize {file_name}", key=f"sum_btn_{i}"):
            st.session_state.trigger_summary = file_name
    st.divider()

# --- Display chat history with Elegant Export Icons ---
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Only show icons for real AI content (not status updates)
        if msg["role"] == "assistant" and not msg.get("is_status"):
            btn_col1, btn_col2, _ = st.columns([0.1, 0.1, 0.8])
            
            with btn_col1:
                st.download_button(
                    label="📥",
                    help="Download as Markdown",
                    data=msg["content"],
                    file_name=f"Summary_{i}.md",
                    mime="text/markdown",
                    key=f"dl_{i}"
                )
            with btn_col2:
                with st.popover("📋", help="Copy Text"):
                    st.caption("Copy from block below:")
                    st.code(msg["content"], language="markdown")

# --- 4. Logic Router ---

# Scenario A: The Summary Button
if "trigger_summary" in st.session_state:
    target_file = st.session_state.trigger_summary
    del st.session_state.trigger_summary 
    
    st.session_state.messages.append({"role": "user", "content": f"Summarize `{target_file}`"})
    
    hidden_prompt = f"""You are analyzing '{target_file}'. TEACH the core mechanics.
    --- GLOBAL RULES ---
    1. Anti-Jargon: Define technical terms inline immediately.
    2. First-Principles: Explain HOW it works conceptually/physically.
    --- STRUCTURE ---
    ### The Core Intuition (The "Big Idea")
    ### [Dynamic Headers based on Paper's Logic]
    ### The Bottom Line
    """
    
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_session.send_message_stream(hidden_prompt)
        full_response = st.write_stream((chunk.text for chunk in response_stream))
        
    st.session_state.messages.append({"role": "assistant", "content": full_response, "is_status": False})
    st.rerun()

# Scenario B: Chat Input
if user_input := st.chat_input("Ask about the documents..."):
    if not st.session_state.chat_session:
        st.warning("Please upload and process papers first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_session.send_message_stream(user_input)
        full_response = st.write_stream((chunk.text for chunk in response_stream))
        
    st.session_state.messages.append({"role": "assistant", "content": full_response, "is_status": False})
    st.rerun()