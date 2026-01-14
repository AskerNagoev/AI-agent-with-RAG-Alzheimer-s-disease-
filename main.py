import streamlit as st
import os
from dotenv import load_dotenv

# Подготовленные функции
from rag.rag import startRAG, answer_question, chat_update

st.set_page_config(page_title="AI assistant in Alzheimer's disease")

st.title("🧬 AI assistant")

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
db_path = "db_collecting/chroma_artilcles"

# Запуск моделей БЯМ и retriever с кэшем
@st.cache_resource
def load_rag():
    return startRAG(db_path, api_key)

if "ready_toast_shown" not in st.session_state:
    st.session_state.ready_toast_shown = False

try:
    identified_answer_chain, answer_chain, structurized_answer_chain, retriever = load_rag()
    if not st.session_state.ready_toast_shown:
        st.toast("The system is ready!")
        st.session_state.ready_toast_shown = True
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

if "chat" not in st.session_state:
    st.session_state.chat = []
if "current_files" not in st.session_state:
    st.session_state.current_files = {}

for msg in st.session_state.chat:
    role = "assistant" if msg["role"] == "ai" else "user"
    with st.chat_message(role):
        st.markdown(msg["text"])

if prompt := st.chat_input("Your question..."):
    # Текст пользователя
    st.chat_message("user").write(prompt)
    st.session_state.chat = chat_update(st.session_state.chat, "user", prompt)

    # Текст агента
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            updated_chat, file_dict = answer_question(identified_answer_chain, answer_chain, structurized_answer_chain, retriever, st.session_state.chat)
            st.markdown(updated_chat[-1]["text"])

    # Обновление истории чата и списка файлов
    st.session_state.chat = updated_chat
    st.session_state.current_files = file_dict

def clear_chat():
    st.session_state.chat = []
    st.session_state.current_files = {}

with st.sidebar:
    st.header("📚 Articles")
    
    if st.session_state.current_files:
        for filename, title in st.session_state.current_files.items():
            short_title = title if len(title) < 50 else title[:50] + "..."
            
            with st.expander(f"📄 {short_title}"):
                file_url = f"app/static/files/{filename}"
                
                html_link = f"""
                <a href="{file_url}" target="_blank" style="
                    text-decoration: none; color: #0068c9; font-weight: bold;
                ">
                    🔗 Open {filename}
                </a>
                <br><br>
                <div style="font-size: 0.85em; color: #555;">
                    <b>Full title:</b><br>{title}
                </div>
                """
                st.markdown(html_link, unsafe_allow_html=True)
    else:
        st.info("There are no sources for the current question.")
    
    st.markdown("---")

    st.button("🗑️ Clear chat", use_container_width=True, on_click=clear_chat)
