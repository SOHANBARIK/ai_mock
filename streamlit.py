import streamlit as st
import tempfile
import os
import asyncio
import time
import requests
from dotenv import load_dotenv
from groq import Groq
import edge_tts

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- CONFIGURATION & KEYS ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GROQ_API_KEY or not GOOGLE_API_KEY:
    st.error("Missing API Keys. Please ensure GROQ_API_KEY and GOOGLE_API_KEY are set.")
    st.stop()

st.set_page_config(page_title="AI Mock Interview", page_icon="👔", layout="wide")

# --- INITIALIZE CLOUD CLIENTS ---
groq_client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

embeddings = load_embeddings()

# --- SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "interview_active" not in st.session_state: st.session_state.interview_active = False
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "evaluation" not in st.session_state: st.session_state.evaluation = None
if "chat_history_display" not in st.session_state: st.session_state.chat_history_display = []
if "latest_audio" not in st.session_state: st.session_state.latest_audio = None
if "llm" not in st.session_state: st.session_state.llm = None

# NEW: IDE & Timer States
if "is_dsa_mode" not in st.session_state: st.session_state.is_dsa_mode = False
if "dsa_start_time" not in st.session_state: st.session_state.dsa_start_time = 0
if "code_output" not in st.session_state: st.session_state.code_output = ""
if "ide_code" not in st.session_state: st.session_state.ide_code = ""

# --- TTS FUNCTION ---
def generate_tts_audio(text):
    voice = "en-US-ChristopherNeural" 
    async def _generate():
        communicate = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            await communicate.save(tmp_file.name)
            with open(tmp_file.name, "rb") as f:
                audio_bytes = f.read()
        os.remove(tmp_file.name)
        return audio_bytes
    return asyncio.run(_generate())

# --- RAG PIPELINE ---
def process_resume(uploaded_file):
    with st.spinner("Processing Resume..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        vector_store = FAISS.from_documents(chunks, embeddings)
        st.session_state.vector_store = vector_store
        os.remove(tmp_path)

# --- FINAL EVALUATION ---
def generate_evaluation(role):
    with st.spinner("Analyzing Interview Transcript..."):
        transcript = ""
        for msg in st.session_state.messages:
            role_name = "Interviewer (Sohan)" if msg.type == "ai" else "Candidate"
            transcript += f"{role_name}: {msg.content}\n\n"

        eval_prompt = f"""
        You are an expert technical recruiter evaluating a candidate for the role of {role}.
        Review the following interview transcript and provide a detailed, professional evaluation.
        Format your response cleanly using Markdown. Include:
        ### 📊 Performance Scores
        * **Technical Accuracy:** [Score/10] - [Justification]
        * **Communication Skills:** [Score/10] - [Justification]
        * **Coding Efficiency & Logic:** [Score/10] - [Evaluate their written code and time taken if applicable]
        * **Overall Score:** [Score/100]

        ### 🌟 Strengths
        [Bullet points]

        ### 📈 Areas for Improvement
        [Bullet points]

        ### 💡 Final Hiring Recommendation
        [Hire / No-Hire / Proceed]

        TRANSCRIPT:
        {transcript}
        """
        response = st.session_state.llm.invoke([HumanMessage(content=eval_prompt)])
        st.session_state.evaluation = response.content

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("📝 Interview Setup")
    target_role = st.text_input("Target Role", placeholder="e.g., Software Engineer")
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    st.write("---")
    st.header("🤖 Interviewer Personality")
    interview_style = st.slider("Strictness Level", 0.0, 1.0, 0.5, 0.1)

    if st.button("Start Interview", type="primary", use_container_width=True):
        if target_role and resume_file:
            process_resume(resume_file)
            st.session_state.interview_active = True
            st.session_state.messages = []
            st.session_state.chat_history_display = []
            st.session_state.evaluation = None
            st.session_state.latest_audio = None
            st.session_state.is_dsa_mode = False
            
            st.session_state.llm = ChatGroq(temperature=interview_style, model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
            
            docs = st.session_state.vector_store.similarity_search(target_role, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            
            # --- UPDATED INITIAL PROMPT ---
            system_msg = SystemMessage(content=f"""
            Your name is Sohan. You are an expert technical recruiter evaluating a candidate for the role of {target_role}.
            Use this resume context to ask your first question: {context}.

            CRITICAL: If your question requires the candidate to write code, you must do TWO things:
            1. Verbally tell them you are opening a coding workspace (e.g., "Let's write some code, I'll open an editor on your screen.")
            2. Append the exact tag <DSA_MODE> at the very end of your response.
            3. If it's just a verbal question and not a coding question, remove the coding workspace and DO NOT use <DSA_MODE>
            """)
            initial_prompt = f"I am applying for the {target_role} role. Introduce yourself briefly and ask me my first interview question based on my experience."
            
            response = st.session_state.llm.invoke([system_msg, HumanMessage(content=initial_prompt)])
            ai_text = response.content
            
            # Check for IDE trigger
            if "<DSA_MODE>" in ai_text:
                st.session_state.is_dsa_mode = True
                st.session_state.dsa_start_time = time.time()
                ai_text = ai_text.replace("<DSA_MODE>", "").strip()
            
            st.session_state.messages.append(AIMessage(content=ai_text))
            st.session_state.chat_history_display.append({"role": "assistant", "content": ai_text})
            
            with st.spinner("Sohan is preparing to speak..."):
                st.session_state.latest_audio = generate_tts_audio(ai_text)
            st.rerun()
        else:
            st.error("Please provide both a role and a resume.")

    if st.session_state.interview_active:
        st.write("---")
        if st.button("🔴 Finish & Evaluate", use_container_width=True):
            st.session_state.interview_active = False
            st.session_state.latest_audio = None
            generate_evaluation(target_role)
            st.rerun()

# --- MAIN INTERFACE LAYOUT ---
st.title("👔 AI Mock Interview")

if not st.session_state.interview_active and not st.session_state.evaluation:
    st.info("👈 Upload your resume and enter a role in the sidebar to begin.")

# Dynamic Layout: Split screen if DSA question is active
if st.session_state.get("is_dsa_mode"):
    chat_col, ide_col = st.columns([1, 1], gap="large")
else:
    chat_col = st.container()
    ide_col = None

# 1. CHAT COLUMN
with chat_col:
    for msg in st.session_state.chat_history_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.interview_active:
        st.write("---")
        if st.session_state.get("is_dsa_mode"):
            st.warning("⏱️ Timer is running! Write your code on the right, then record your verbal explanation here to submit.")
            
        audio_value = st.audio_input("Record your answer")

        if audio_value:
            current_audio_bytes = audio_value.getvalue()
            
            if "prev_audio_bytes" not in st.session_state or st.session_state.prev_audio_bytes != current_audio_bytes:
                st.session_state.prev_audio_bytes = current_audio_bytes
                
                with st.spinner("Transcribing answer..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                        tmp_audio.write(current_audio_bytes)
                        tmp_audio_path = tmp_audio.name
                    
                    with open(tmp_audio_path, "rb") as file:
                        transcription = groq_client.audio.transcriptions.create(
                          file=(tmp_audio_path, file.read()), model="whisper-large-v3", language="en"
                        )
                    spoken_text = transcription.text.strip()
                    os.remove(tmp_audio_path)

                if spoken_text:
                    # FUSION: Combine spoken text with IDE code and timer if DSA mode was active
                    if st.session_state.get("is_dsa_mode"):
                        elapsed_time = int(time.time() - st.session_state.dsa_start_time)
                        mins, secs = divmod(elapsed_time, 60)
                        
                        final_user_submission = f"""
**Spoken Explanation:** {spoken_text}

**Code Written:**
{st.session_state.ide_code}


**Compiler Output:**
{st.session_state.code_output}


**Time Taken:** {mins}m {secs}s
                        """
                        # Reset IDE state for next question
                        st.session_state.is_dsa_mode = False 
                        st.session_state.ide_code = ""
                        st.session_state.code_output = ""
                    else:
                        final_user_submission = spoken_text

                    st.session_state.messages.append(HumanMessage(content=final_user_submission))
                    st.session_state.chat_history_display.append({"role": "user", "content": final_user_submission})
                    
                    with st.spinner("Interviewer is reviewing your answer..."):
                        docs = st.session_state.vector_store.similarity_search(spoken_text, k=2)
                        context = "\n".join([doc.page_content for doc in docs])
                        
                        # --- UPDATED ACTIVE INTERVIEW PROMPT ---
                        system_msg = SystemMessage(content=f"""
                        Your name is Sohan. You are a Senior AI Engineer conducting an interview.
                        Review the candidate's latest answer (including their code and time taken if provided).
                        Ask ONE relevant follow-up question or move to a new technical topic.

                        CRITICAL INSTRUCTION: If you want to test their Data Structures and Algorithms skills by having them write code:
                        1. Verbally announce that you are opening a shared coding workspace for them (e.g. "Let's test that logic. I'm opening a coding editor for you now...").
                        2. Append the EXACT tag <DSA_MODE> at the very end of your text response.

                        If it is just a verbal/conceptual question, do NOT use the tag. Do NOT use emojis.
                        Relevant Resume Context: {context}
                        """)
                        
                        response = st.session_state.llm.invoke([system_msg] + st.session_state.messages)
                        ai_text = response.content
                        
                        # Check for IDE trigger in follow-up
                        if "<DSA_MODE>" in ai_text:
                            st.session_state.is_dsa_mode = True
                            st.session_state.dsa_start_time = time.time()
                            ai_text = ai_text.replace("<DSA_MODE>", "").strip()

                        st.session_state.messages.append(AIMessage(content=ai_text))
                        st.session_state.chat_history_display.append({"role": "assistant", "content": ai_text})
                        
                    with st.spinner("Sohan is preparing to speak..."):
                        st.session_state.latest_audio = generate_tts_audio(ai_text)
                    st.rerun()

# 2. IDE COLUMN (Only visible during DSA questions)
if ide_col:
    with ide_col:
        st.markdown("### 💻 Coding Workspace")
        lang = st.selectbox("Language", ["python", "c", "cpp", "javascript", "java", "bash"])
        
        # The key="ide_code" automatically saves what you type to st.session_state.ide_code
        code_input = st.text_area("Write your solution here:", height=300, key="ide_code")
        
        if st.button("▶ Run Code", use_container_width=True):
            with st.spinner("Compiling..."):
                try:
                    # --- DYNAMIC API ROUTING ---
                    if lang == "java":
                        backend_url = "https://vibe-coding-ide-1.onrender.com/execute"
                    else:
                        backend_url = "https://vibe-coding-ide.onrender.com/execute"
                        
                    res = requests.post(
                        backend_url,
                        json={"language": lang, "code": code_input},
                        timeout=15
                    )
                    if res.status_code == 200:
                        st.session_state.code_output = res.text
                    else:
                        st.session_state.code_output = f"API Error: {res.status_code}\n{res.text}"
                except Exception as e:
                    st.session_state.code_output = f"Failed to connect to backend: {e}"
        
        st.text_area("Console Output:", value=st.session_state.code_output, height=150, disabled=True)

# --- AUDIO PLAYBACK ---
if st.session_state.latest_audio:
    st.audio(st.session_state.latest_audio, format="audio/mp3", autoplay=True)
    st.session_state.latest_audio = None

# --- EVALUATION DISPLAY ---
if st.session_state.evaluation:
    st.write("---")
    st.subheader("📊 Interview Evaluation Report")
    st.markdown(st.session_state.evaluation)
    st.download_button(
        label="📥 Download Evaluation Report",
        data=st.session_state.evaluation,
        file_name="Interview_Evaluation_Report.txt",
        mime="text/plain",
        use_container_width=True
    )
    if st.button("🔄 Start New Interview", use_container_width=True):
        st.session_state.clear()
        st.rerun()