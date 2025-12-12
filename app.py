import os
import re
import utills
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from groq import Groq  # for Whisper speech recognition

# Load API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM
llm = ChatGroq(model="qwen/qwen3-32b", groq_api_key=api_key)

# Groq Whisper client
groq_client = Groq(api_key=api_key)

# Streamlit UI
st.title("Muubii LangChain + Groq Chatbot 🎤🤖🔥")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Show previous messages
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

# --------- Voice Input ----------
st.subheader("🎙 Voice Input")
audio_file = st.audio_input("Click to record your voice")

voice_text = None
if audio_file is not None:
    st.success("Voice received! Converting to text...")

    # Whisper speech-to-text
    transcription = groq_client.audio.transcriptions.create(
        file=(audio_file.name, audio_file.getvalue()),
        model="whisper-large-v3-turbo"
    )

    voice_text = transcription.text
    st.info(f"🎧 You said: **{voice_text}**")

# --------- Text Input ----------
typed_input = st.chat_input("Say something...")

# Prefer voice input if available
if voice_text:
    user_input = voice_text
else:
    user_input = typed_input

# --------- Process Message ----------
if user_input:
    # Add user message
    st.session_state.history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    utills.log_interaction("user", user_input)

    system_prompt = SystemMessage(
    content=(
        "You were created by Muubii Bytes. Anytime anyone asks who made you, "
        "who created you, who built you, who owns you, or anything similar, "
        "you must clearly say: 'I was created by Mubarak Dalhatu(Muubii Bytes) the Great AI Engineer.'"
    )
)
    # Convert to LangChain style messages
    lc_messages = [system_prompt]
    for msg in st.session_state.history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    # Generate LLM response
    try:
        ai_response = llm.invoke(lc_messages).content
    except Exception as e:
        error_msg = f"LLM Error: {str(e)}"
        utills.log_error(error_msg)
        clean_response = "⚠️ Something went wrong while generating a response."
        st.chat_message("assistant").write(clean_response)
        st.session_state.history.append({"role": "assistant", "content": clean_response})

    # Clean AI response
    clean_response = re.sub(r"<think>.*?</think>", "", ai_response, flags=re.DOTALL).strip()

    # Save + display reply
    st.session_state.history.append({"role": "assistant", "content": ai_response})
    st.chat_message("assistant").write(clean_response)
    utills.log_interaction("assistant", clean_response)

    # ------- GROQ TTS -------
    for part in utills.chunk_text(clean_response, max_chars=1000):
        try:
            speech_response = groq_client.audio.speech.create(
                model="playai-tts",
                voice="Fritz-PlayAI",
                input=part,
                response_format="wav"
            )

            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                speech_response.write_to_file(tmp.name)
                st.audio(tmp.name, format="audio/wav")

        except Exception as e:
            error_msg = f"TTS Error: {str(e)}"
            utills.log_error(error_msg)
            st.warning("⚠️ Voice output disabled temporarily (rate limit or error).")
            break



