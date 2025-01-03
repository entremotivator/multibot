import streamlit as st
import langchain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from textblob import TextBlob  # For sentiment analysis
import requests
import time

# Configuration
DEFAULT_API_URL = "https://theaisource-u29564.vm.elestio.app:57987"
DEFAULT_USERNAME = "root"
DEFAULT_PASSWORD = "eZfLK3X4-SX0i-UmgUBe6E"

DEFAULT_PERSONALITIES = {
    "Customer Support": "A friendly and empathetic assistant resolving customer issues.",
    "Sales Expert": "A persuasive professional specializing in closing deals.",
    "Tech Consultant": "An analytical advisor providing technical solutions.",
    "HR Specialist": "A compassionate HR professional offering workplace guidance.",
    "Financial Advisor": "A meticulous advisor specializing in financial planning.",
    "Marketing Guru": "A creative strategist for branding and campaigns.",
    "Project Manager": "A detail-oriented leader focusing on team management.",
    "Legal Consultant": "An authoritative figure providing concise legal advice.",
    "Healthcare Specialist": "A knowledgeable advisor on medical and wellness topics.",
    "Education Mentor": "A supportive guide for learning and career development.",
}

# Helper Functions
def get_ollama_models() -> list:
    """Retrieve available models from the Ollama API."""
    try:
        response = requests.get(
            f"{DEFAULT_API_URL}/api/tags",
            auth=(DEFAULT_USERNAME, DEFAULT_PASSWORD)
        )
        if response.status_code == 200:
            models = response.json()
            return [
                model['name'] for model in models['models']
                if all(keyword not in model['name'].lower()
                       for keyword in ('failed', 'embed', 'bge'))
            ]
        return []
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def create_custom_personality() -> str:
    """Allow users to create a custom personality."""
    st.sidebar.subheader("Create Your Custom Personality")
    tone = st.sidebar.text_input("Tone (e.g., formal, casual, empathetic):", "friendly")
    expertise = st.sidebar.text_input("Expertise (e.g., technology, finance):", "general")
    style = st.sidebar.text_input("Response Style (e.g., concise, detailed):", "concise")
    return f"A {tone} expert in {expertise}, responding in a {style} manner."

def get_conversation_chain(model_name: str, personality: str) -> ConversationChain:
    """Initialize LangChain conversation chain with selected model."""
    llm = Ollama(
        model=model_name,
        temperature=0.2,
        base_url=DEFAULT_API_URL,
        auth=(DEFAULT_USERNAME, DEFAULT_PASSWORD),
    )
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=f"""You are a chatbot with the following personality: {personality}.
                     Current conversation:
                     {{history}}
                     Human: {{input}}
                     Assistant:"""
    )
    memory = ConversationBufferMemory(return_messages=True)
    return ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=True)

def sentiment_analysis(response: str) -> str:
    """Perform sentiment analysis on the response."""
    sentiment = TextBlob(response).sentiment
    if sentiment.polarity > 0:
        return "Positive"
    elif sentiment.polarity < 0:
        return "Negative"
    return "Neutral"

def summarize_conversation(messages):
    """Summarize the conversation."""
    summary = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    return summary[:2000] + "..." if len(summary) > 2000 else summary

# Main Application
def run():
    """Main function to run the Streamlit chatbot interface."""
    st.title("🤖 Advanced Business Chatbot")
    st.markdown("Interact with various chatbot personalities tailored for business roles.")

    # Sidebar Configuration
    st.sidebar.subheader("Settings")
    model_name = st.sidebar.selectbox("Select Model", get_ollama_models(), key="model_select")
    personality_choice = st.sidebar.radio("Select Personality", list(DEFAULT_PERSONALITIES.keys()) + ["Custom"])
    custom_personality = create_custom_personality() if personality_choice == "Custom" else None

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize conversation chain
    if st.session_state.conversation is None:
        personality = custom_personality or DEFAULT_PERSONALITIES[personality_choice]
        st.session_state.conversation = get_conversation_chain(model_name, personality)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input and processing
    if prompt := st.chat_input("Type your message here:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            try:
                response = st.session_state.conversation.run(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                sentiment = sentiment_analysis(response)
                response_placeholder.markdown(f"Response:\n\n{response}\n\nSentiment: **{sentiment}**")
            except Exception as e:
                error_message = f"Error: {str(e)}"
                response_placeholder.error(error_message)

    # Session management
    st.sidebar.subheader("Session Management")
    if st.sidebar.button("Summarize Conversation"):
        st.sidebar.text_area("Summary", summarize_conversation(st.session_state.messages))

if __name__ == "__main__":
    run()
