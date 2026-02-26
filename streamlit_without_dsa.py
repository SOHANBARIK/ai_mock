import streamlit as st
import tempfile
import os
import asyncio
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
# 1. Groq Client for Lightning-Fast STT (Whisper)
groq_client = Groq(api_key=GROQ_API_KEY)

# 2. Gemini for Lightweight Cloud Embeddings
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

# --- TTS FUNCTION (Professional Male Voice) ---
def generate_tts_audio(text):
    """Generates a professional male voice using edge-tts."""
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

        # Create FAISS Vector Store in memory (very lightweight)
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
        You are an expert technical recruiter and Senior Engineer evaluating a candidate for the role of {role}.
        Review the following interview transcript and provide a detailed, professional evaluation.
        Format your response cleanly using Markdown. You MUST include the following exact sections:

        ### 📊 Performance Scores
        * **Technical Accuracy:** [Score out of 10] - [1 sentence justification]
        * **Communication Skills:** [Score out of 10] - [1 sentence justification]
        * **Overall Score:** [Score out of 100]

        ### 🌟 Strengths
        [Bullet point 2-3 specific positive observations from their answers]

        ### 📈 Areas for Improvement
        [Bullet point 1-2 constructive points on how they can improve technically or in their delivery]

        ### 💡 Final Hiring Recommendation
        [Provide a realistic Hire / No-Hire / Proceed-to-Next-Round recommendation based purely on this transcript]

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
    interview_style = st.slider(
        "Strictness Level", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.1,
        help="0.0 is strictly factual and rigorous. 1.0 is highly creative and conversational."
    )

    if st.button("Start Interview", type="primary", use_container_width=True):
        if target_role and resume_file:
            process_resume(resume_file)
            st.session_state.interview_active = True
            st.session_state.messages = []
            st.session_state.chat_history_display = []
            st.session_state.evaluation = None
            st.session_state.latest_audio = None
            
            # Initialize dynamic LLM based on slider
            st.session_state.llm = ChatGroq(
                temperature=interview_style, 
                model_name="llama-3.3-70b-versatile", 
                api_key=GROQ_API_KEY
            )
            
            # Initial Setup Prompt
            docs = st.session_state.vector_store.similarity_search(target_role, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            
            system_msg = SystemMessage(content=f"Your name is Sohan. You are a Senior AI Engineer conducting a technical interview. Use this resume context to ask your first question: {context}")
            initial_prompt = f"I am applying for the {target_role} role. Introduce yourself briefly and ask me my first interview question based on my experience."
            
            response = st.session_state.llm.invoke([system_msg, HumanMessage(content=initial_prompt)])
            
            # Save AI text
            st.session_state.messages.append(AIMessage(content=response.content))
            st.session_state.chat_history_display.append({"role": "assistant", "content": response.content})
            
            with st.spinner("Sohan is preparing to speak..."):
                st.session_state.latest_audio = generate_tts_audio(response.content)
            
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

# --- MAIN CHAT INTERFACE ---
st.title("👔 Technical Mock Interview")

if not st.session_state.interview_active and not st.session_state.evaluation:
    st.info("👈 Upload your resume and enter a role in the sidebar to begin.")

# Display Chat History
for msg in st.session_state.chat_history_display:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle Active Interview Audio Input
if st.session_state.interview_active:
    st.write("---")
    audio_value = st.audio_input("Record your answer")

    if audio_value:
        current_audio_bytes = audio_value.getvalue()
        
        if "prev_audio_bytes" not in st.session_state or st.session_state.prev_audio_bytes != current_audio_bytes:
            st.session_state.prev_audio_bytes = current_audio_bytes
            
            with st.spinner("Transcribing answer..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                    tmp_audio.write(current_audio_bytes)
                    tmp_audio_path = tmp_audio.name
                
                # Transcribe via Groq API (Zero RAM overhead)
                with open(tmp_audio_path, "rb") as file:
                    transcription = groq_client.audio.transcriptions.create(
                      file=(tmp_audio_path, file.read()),
                      model="whisper-large-v3",
                      language="en"
                    )
                
                user_text = transcription.text.strip()
                os.remove(tmp_audio_path)

            if user_text:
                st.session_state.messages.append(HumanMessage(content=user_text))
                st.session_state.chat_history_display.append({"role": "user", "content": user_text})
                
                with st.spinner("Interviewer is reviewing your answer..."):
                    # Retrieve context
                    docs = st.session_state.vector_store.similarity_search(user_text, k=2)
                    context = "\n".join([doc.page_content for doc in docs])
                    
                    system_msg = SystemMessage(content=f"""
                    Your name is Sohan. You are a Senior AI Engineer conducting an interview. 
                    Review the chat history and the candidate's latest answer. Ask ONE relevant follow-up question or move to a new technical topic.
                    
                    CRITICAL INSTRUCTION: Analyze the Resume Context below for the candidate's primary programming languages. 
                    If you have not done so already, you MUST ask at least one complex Data Structures and Algorithms question tailored specifically to their strongest programming language.
                    
                    Keep your tone conversational but highly rigorous. Do NOT use emojis.
                    
                    Relevant Resume Context for reference: {context}
                    """)
                    
                    response = st.session_state.llm.invoke([system_msg] + st.session_state.messages)
                    
                    # Save AI text
                    st.session_state.messages.append(AIMessage(content=response.content))
                    st.session_state.chat_history_display.append({"role": "assistant", "content": response.content})
                    
                with st.spinner("Sohan is preparing to speak..."):
                    st.session_state.latest_audio = generate_tts_audio(response.content)
                    
                st.rerun()

# --- AUDIO PLAYBACK ---
if st.session_state.latest_audio:
    st.audio(st.session_state.latest_audio, format="audio/mp3", autoplay=True)
    st.session_state.latest_audio = None

# --- EVALUATION DISPLAY ---
if st.session_state.evaluation:
    st.write("---")
    st.subheader("📊 Interview Evaluation Report")
    st.markdown(st.session_state.evaluation)
    
    st.write("") 
    
    st.download_button(
        label="📥 Download Evaluation Report",
        data=st.session_state.evaluation,
        file_name="Interview_Evaluation_Report.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    if st.button("🔄 Start New Interview", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history_display = []
        st.session_state.evaluation = None
        st.session_state.latest_audio = None
        st.rerun()