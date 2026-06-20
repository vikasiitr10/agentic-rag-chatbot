import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    # store thread ids as strings for easier session_state keys
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    # default title for a new chat
    st.session_state['thread_titles'][thread_id] = "New Chat"

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        # ensure there is an entry in titles mapping
        if thread_id not in st.session_state['thread_titles']:
            st.session_state['thread_titles'][thread_id] = "New Chat"

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    # state may be a mapping or object with .values; normalize safely
    try:
        return state.get('messages', [])
    except Exception:
        try:
            return state.values.get('messages', [])
        except Exception:
            return []


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}

add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    # display human-friendly title when available
    title = st.session_state['thread_titles'].get(thread_id)
    label = title if title and title != "New Chat" else f"Chat {thread_id[:8]}"
    # use a stable key so Streamlit button keys don't collide
    if st.sidebar.button(label, key=f"open_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        # ensure title exists
        if not st.session_state['thread_titles'].get(thread_id):
            st.session_state['thread_titles'][thread_id] = label

# show current thread title in sidebar
current_title = st.session_state['thread_titles'].get(st.session_state['thread_id'], f"Chat {st.session_state['thread_id'][:8]}")
st.sidebar.subheader(current_title)


# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # derive a thread title from the first user message if it's still default
    if st.session_state['thread_titles'].get(st.session_state['thread_id']) in (None, "New Chat"):
        # take first line or first 6 words as a short title
        first_line = user_input.strip().split('\n')[0]
        words = first_line.split()
        short = ' '.join(words[:6])
        title = (short + '...') if len(words) > 6 else short
        st.session_state['thread_titles'][st.session_state['thread_id']] = title or "Chat"

    # stream assistant response and capture full content
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        assistant_content = ""
        placeholder = st.empty()
        for chunk in ai_only_stream():
            assistant_content += chunk
            placeholder.write(assistant_content)

    st.session_state['message_history'].append({'role': 'assistant', 'content': assistant_content})