import streamlit as st
import time
import datetime
import random
import google.generativeai as genai
import pandas as pd
import chromadb
import warnings
warnings.filterwarnings('ignore') # Suppress warnings

GOOGLE_API_KEY = "AIzaSyBlAiRqnNHmm-Hfu8dCAx6dlVMROQ-c180"
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the AI model
model = genai.GenerativeModel('gemini-1.5-flash')

# --- SECTION 1: KNOWLEDGE BASE SETUP ---
print("🧠 Building your crime mitigation knowledge base...")

# Load your CSV data
csv_filename = "securo_crime.csv" # Make sure this file is in the same directory as your app
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    st.error(f"Error: '{csv_filename}' not found. Please upload your CSV file.")
    st.stop()

# Combine question and answer for documents
csv_documents = (df['question'] + " " + df['answer']).tolist()

all_documents = csv_documents

print(f"📚 Total knowledge base: {len(all_documents)} documents")
print(f"\n📊 Your data preview (first 5 entries):")
for i, doc in enumerate(all_documents[:5]):
    print(f"  - {doc}")

# Create your smart memory with ChromaDB
print("🗄️ Setting up your smart memory...")

# Create ChromaDB client and collection
client = chromadb.Client()
collection_name = "securo_chatbot_enhanced"

# Delete old collection if exists to ensure a clean start
try:
    client.delete_collection(collection_name)
    print("🧹 Cleaned up old collection.")
except Exception:
    pass # Collection might not exist, which is fine

collection = client.create_collection(name=collection_name)
print(f"📁 Created collection: '{collection_name}'")

# Add all documents to the collection
collection.add(
    documents=all_documents,
    ids=[str(i) for i in range(len(all_documents))]
)
print(f"✅ Smart memory ready with {len(all_documents)} pieces of knowledge!")

# --- SECTION 2: PROMPT TEMPLATE ---
print("📝 Creating smart response templates...")

def create_crime_mitigation_prompt(user_question, relevant_context):
    """Create a smart prompt for the AI using our knowledge"""
   
    prompt = f"""You are a helpful and friendly St. Kitts & Nevis crime mitigation assistant. You have access to specific crime prevention and mitigation information and you should use this information to provide accurate, helpful answers. Speak in an informative, warm, and professional tone.

**CRIME MITIGATION KNOWLEDGE (from your data):**
{relevant_context}

**USER QUESTION:** {user_question}

**INSTRUCTIONS:**
- Use the provided crime mitigation knowledge to answer the question thoroughly and accurately.
- Be warm and professional in your responses, focusing on crime mitigation.
- If the knowledge base contains relevant details, include specific information like statistics, locations of police stations, lawyer services, or other pertinent data from the provided context.
- If the exact answer or specific details are NOT in your provided knowledge, politely state that you don't have that specific information in your current database. Do not hallucinate.
- Keep your answer helpful, concise, and easy to understand.
- Always use a warm, welcoming, and reassuring tone.
- If the user asks about a general crime term (e.g., "What is phishing?"), and the answer is in your knowledge, provide it directly.
- If the user asks a very general question that could benefit from broader context (e.g., "Tell me about crime in St. Kitts & Nevis"), synthesize information from the provided knowledge if available, or state you are focused on mitigation and prevention.

**ANSWER:**
"""
   
    return prompt

# --- SECTION 3: Streamlit App Configuration ---
# Page configuration
st.set_page_config(
    page_title="SECURO - St. Kitts & Nevis Crime AI Assistant",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling - keeping the exact same design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');
   
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
   
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #2e1a1a 50%, #3e1616 100%);
        font-family: 'JetBrains Mono', monospace;
    }
   
    /* Particles animation */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
    }

    .particle {
        position: absolute;
        width: 2px;
        height: 2px;
        background: rgba(255, 68, 68, 0.3);
        border-radius: 50%;
        animation: float 10s infinite linear;
    }

    @keyframes float {
        0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateY(-100px) rotate(360deg); opacity: 0; }
    }
   
    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
        background: rgba(0, 0, 0, 0.7);
        border-radius: 15px;
        border: 1px solid rgba(255, 68, 68, 0.3);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 68, 68, 0.1), transparent);
        animation: scan 3s infinite;
    }

    @keyframes scan {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }

    .main-header h1 {
        font-size: 3rem;
        color: #ff4444;
        text-shadow: 0 0 20px rgba(255, 68, 68, 0.5);
        margin-bottom: 10px;
        position: relative;
        z-index: 2;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .main-header .tagline {
        font-size: 1rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        z-index: 2;
        font-family: 'JetBrains Mono', monospace;
    }

    .main-header .location {
        font-size: 0.9rem;
        color: #ff4444;
        margin-top: 5px;
        position: relative;
        z-index: 2;
        font-family: 'JetBrains Mono', monospace;
    }
   
    /* Sidebar styling - Multiple selectors for different Streamlit versions */
    .css-1d391kg, .css-1cypcdb, .css-k1vhr6, .css-1lcbmhc, .css-17eq0hr,
    section[data-testid="stSidebar"], .stSidebar, [data-testid="stSidebar"] > div,
    .css-1aumxhk, .css-hxt7ib, .css-17lntkn {
        background: rgba(40, 20, 20, 0.9) !important;
        border-right: 1px solid rgba(255, 68, 68, 0.3) !important;
        backdrop-filter: blur(10px) !important;
    }
   
    /* Sidebar header styling */
    section[data-testid="stSidebar"] .css-10trblm {
        color: #ff4444 !important;
    }
   
    /* Sidebar content background */
    .css-1cypcdb .css-17lntkn {
        background: transparent !important;
    }
   
    /* Emergency contacts styling */
    .contact-item {
        background: rgba(0, 0, 0, 0.5);
        padding: 12px;
        margin-bottom: 8px;
        border-radius: 8px;
        border-left: 3px solid #ff4444;
        cursor: pointer;
        transition: all 0.3s ease;
        color: #e0e0e0;
        font-family: 'JetBrains Mono', monospace;
    }

    .contact-item:hover {
        background: rgba(255, 68, 68, 0.1);
        transform: translateX(5px);
    }

    .contact-name {
        color: #e0e0e0;
        font-size: 0.9rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
    }

    .contact-number {
        color: #ff4444;
        font-size: 0.8rem;
        margin-top: 3px;
        font-family: 'JetBrains Mono', monospace;
    }
   
    /* Sidebar toggle button */
    .sidebar-toggle {
        position: fixed;
        top: 70px;
        left: 20px;
        z-index: 999;
        background: linear-gradient(135deg, #ff4444, #cc3333);
        border: none;
        border-radius: 8px;
        color: white;
        padding: 10px 15px;
        cursor: pointer;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.3);
    }
   
    .sidebar-toggle:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(255, 68, 68, 0.5);
    }
   
    /* Map container with better styling */
    .map-container {
        background: rgba(0, 0, 0, 0.8);
        border-radius: 10px;
        padding: 0;
        border: 1px solid rgba(255, 68, 68, 0.3);
        position: relative;
        height: 300px;
        overflow: hidden;
        margin-bottom: 15px;
    }
   
    /* Map iframe styling */
    .crime-map iframe {
        width: 100%;
        height: 100%;
        border: none;
        border-radius: 10px;
        filter: invert(0.9) hue-rotate(180deg) saturate(1.2);
    }

    .map-placeholder {
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, #2e1a1a, #3e1616);
        border-radius: 8px;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #666;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .hotspot {
        position: absolute;
        width: 12px;
        height: 12px;
        background: #ff4444;
        border-radius: 50%;
        animation: pulse-hotspot 2s infinite;
        cursor: pointer;
    }

    .hotspot::after {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        border: 2px solid rgba(255, 68, 68, 0.5);
        border-radius: 50%;
        animation: ripple 2s infinite;
    }

    @keyframes pulse-hotspot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.2); }
    }

    @keyframes ripple {
        0% { transform: scale(1); opacity: 1; }
        100% { transform: scale(2); opacity: 0; }
    }

    .hotspot-1 { top: 30%; left: 25%; }
    .hotspot-2 { top: 45%; left: 60%; }
    .hotspot-3 { top: 70%; left: 40%; }
    .hotspot-4 { top: 25%; left: 75%; }
   
    /* Chat styling */
    .chat-message {
        margin-bottom: 20px;
        animation: fadeInUp 0.5s ease;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .user-message {
        text-align: right;
    }

    .bot-message {
        text-align: left;
    }

    .message-content {
        display: inline-block;
        padding: 15px 20px;
        border-radius: 15px;
        max-width: 80%;
        position: relative;
        font-family: 'JetBrains Mono', monospace;
    }

    .user-message .message-content {
        background: linear-gradient(135deg, #ff4444, #cc3333);
        color: #fff;
        border-bottom-right-radius: 5px;
    }

    .bot-message .message-content {
        background: rgba(0, 0, 0, 0.6);
        color: #e0e0e0;
        border: 1px solid rgba(255, 68, 68, 0.3);
        border-bottom-left-radius: 5px;
    }

    .message-time {
        font-size: 0.7rem;
        color: #888;
        margin-top: 5px;
        font-family: 'JetBrains Mono', monospace;
    }
   
    /* Status bar */
    .status-bar {
        background: rgba(0, 0, 0, 0.8);
        padding: 10px 20px;
        border-radius: 25px;
        margin-top: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(255, 68, 68, 0.2);
        font-family: 'JetBrains Mono', monospace;
    }

    .status-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        color: #e0e0e0;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    .status-online { background: #ff4444; }
    .status-processing { background: #cc3333; }
    .status-evidence { background: #ff6666; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
   
    /* Input styling */
    .stTextInput input {
        background: rgba(0, 0, 0, 0.5) !important;
        border: 1px solid rgba(255, 68, 68, 0.3) !important;
        border-radius: 25px !important;
        color: #e0e0e0 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stTextInput input:focus {
        border-color: #ff4444 !important;
        box-shadow: 0 0 20px rgba(255, 68, 68, 0.2) !important;
    }
   
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #ff4444, #cc3333) !important;
        border: none !important;
        border-radius: 25px !important;
        color: #fff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    .stButton button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 20px rgba(255, 68, 68, 0.4) !important;
    }
   
    /* Section headers */
    .section-header {
        color: #ff4444;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 15px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Function to search ChromaDB and get relevant context
def get_relevant_context(query, num_results=3):
    """Searches ChromaDB for relevant documents and returns them as a single string."""
    results = collection.query(
        query_texts=[query],
        n_results=num_results
    )
    if results and results['documents']:
        # Flatten the list of lists into a single list of documents
        flat_documents = [doc for sublist in results['documents'] for doc in sublist]
        return "\n".join(flat_documents)
    return "No specific crime mitigation knowledge found."

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    # Add initial bot message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Welcome to SECURO, your AI crime investigation assistant for St. Kitts & Nevis law enforcement.\n\nI'm here to assist criminologists, police officers, forensic experts, and autopsy professionals with case analysis, evidence correlation, and investigative insights.\n\nHow can I assist with your investigation today?",
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
    })

if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = "expanded"

# Header with sidebar toggle
col1, col2 = st.columns([1, 10])

with col1:
    if st.button(" ", help="Toggle Sidebar", key="sidebar_toggle"):
        if st.session_state.sidebar_state == "expanded":
            st.session_state.sidebar_state = "collapsed"
        else:
            st.session_state.sidebar_state = "expanded"
        st.rerun()

with col2:
    st.markdown("""
    <div class="main-header">
        <div class="particles" id="particles"></div>
        <h1>SECURO</h1>
        <div class="tagline">AI Crime Investigation Assistant</div>
        <div class="location">  St. Kitts & Nevis Law Enforcement</div>
    </div>
    """, unsafe_allow_html=True)

# Particles animation script
st.markdown("""
<script>
function createParticles() {
    const particlesContainer = document.getElementById('particles');
    if (particlesContainer) {
        const particleCount = 40;
       
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 10 + 's';
            particle.style.animationDuration = (Math.random() * 10 + 15) + 's';
            particlesContainer.appendChild(particle);
        }
    }
}
createParticles();
</script>
""", unsafe_allow_html=True)

# Sidebar (only show if expanded)
if st.session_state.sidebar_state == "expanded":
    with st.sidebar:
        st.markdown('<div class="section-header">  Emergency Contacts</div>', unsafe_allow_html=True)
       
        emergency_contacts = [
            {"name": "Emergency Hotline", "number": "911", "type": "police"},
            {"name": "Police Department", "number": "465-2241", "type": "police"},
            {"name": "Hospital", "number": "465-2551", "type": "hospital"},
            {"name": "Fire Department", "number": "465-2515 / 465-7167", "type": "fire"},
            {"name": "Coast Guard", "number": "465-8384 / 466-9280", "type": "legal"},
            {"name": "Red Cross", "number": "465-2584", "type": "forensic"},
            {"name": "NEMA (Emergency Mgmt)", "number": "466-5100", "type": "legal"}
        ]
       
        for contact in emergency_contacts:
            if st.button(f"  {contact['name']}\n{contact['number']}", key=contact['name']):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"  Emergency contact information accessed: **{contact['name']}** - **{contact['number']}**. This contact has been logged for case documentation.",
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                })
                st.rerun()
       
        st.markdown('<div class="section-header">  Crime Hotspots Map</div>', unsafe_allow_html=True)
       
        # Real Google Maps embed for St. Kitts & Nevis with API key
        st.markdown("""
        <div class="map-container crime-map">
            <iframe
                src="https://www.google.com/maps/embed/v1/place?key=YOUR_Maps_API_KEY&q=St.+Kitts+and+Nevis"
                allowfullscreen=""
                loading="lazy"
                referrerpolicy="no-referrer-when-downgrade">
            </iframe>
        </div>
        """, unsafe_allow_html=True)
        # Note: Replace 'YOUR_Maps_API_KEY' with an actual Google Maps API Key if you want the map to work.
        # Otherwise, it will show a grey box. You can also replace it with a static image placeholder.
       
        # Interactive hotspot buttons
        st.markdown('<div class="section-header">  Active Crime Zones</div>', unsafe_allow_html=True)
       
        hotspots = [
            {"name": "Basseterre Downtown", "level": "  High Risk", "coords": "17.3026, -62.7261"},
            {"name": "Sandy Point", "level": "  Medium Risk", "coords": "17.3580, -62.8419"},
            {"name": "Charlestown (Nevis)", "level": "  Active Cases", "coords": "17.1373, -62.6131"},
            {"name": "Frigate Bay", "level": "  Tourist Area", "coords": "17.2742, -62.6897"}
        ]
       
        for hotspot in hotspots:
            if st.button(f"{hotspot['level']} {hotspot['name']}", key=f"hotspot_{hotspot['name']}"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"  Crime hotspot analysis: **{hotspot['name']}** ({hotspot['coords']})\n\n**{hotspot['level']}** - Recommend increased patrol presence and witness canvassing in this area. Coordinating with local units for enhanced surveillance.",
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                })
                st.rerun()
else:
    # Show collapsed sidebar info in main area if needed
    pass

# Main chat area
st.markdown('<div class="section-header">  Crime Investigation Chat</div>', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-content">{message["content"]}</div>
            <div class="message-time">{message["timestamp"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <div class="message-content">{message["content"]}</div>
            <div class="message-time">{message["timestamp"]}</div>
        </div>
        """, unsafe_allow_html=True)

# Chat input
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Message",
        placeholder="Describe evidence, case details, or ask forensic questions...",
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    if st.button("Send", type="primary"):
        if user_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            })
           
            # Get relevant context from ChromaDB
            relevant_context = get_relevant_context(user_input)
           
            # Create the prompt for the AI model
            full_prompt = create_crime_mitigation_prompt(user_input, relevant_context)
           
            # Generate AI response using the enhanced prompt
            try:
                ai_response = model.generate_content(full_prompt).text
            except Exception as e:
                ai_response = f"I'm sorry, I encountered an error while processing your request: {e}. Please try again."
           
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            })
           
            st.rerun()

# Status bar
st.markdown("""
<div class="status-bar">
    <div class="status-item">
        <div class="status-dot status-online"></div>
        <span>SECURO AI Online</span>
    </div>
    <div class="status-item">
        <div class="status-dot status-processing"></div>
        <span>SKN Crime Database Connected</span>
    </div>
    <div class="status-item">
        <div class="status-dot status-evidence"></div>
        <span>Emergency Services Linked</span>
    </div>
</div>
""", unsafe_allow_html=True)
