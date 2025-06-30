import streamlit as st
from utils import (
    load_data_and_setup_lancedb,
    get_llm_question,
    fields,
    get_recommendations_from_preferences,
    select_best_listing_and_describe
)

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Real Estate Chatbot", page_icon="🤖")
st.title("🤖 Real Estate Recommender Chatbot")

# --- Load Data and Setup LanceDB ---
@st.cache_resource
def get_table():
    return load_data_and_setup_lancedb()

table = get_table()

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "preferences" not in st.session_state:
    st.session_state.preferences = {}
if 'chat_started' not in st.session_state:
    st.session_state.chat_started = False

# --- Helper Functions ---
def start_chat():
    st.session_state.chat_started = True
    st.session_state.messages = []
    st.session_state.current_question = 0
    st.session_state.preferences = {}
    question = fields[0]["question"]
    llm_question = get_llm_question(question)
    st.session_state.messages.append({"role": "assistant", "content": llm_question})

def restart_chat():
    st.session_state.clear()
    start_chat()

def get_valid_input(answer, expected_type):
    try:
        if expected_type == int:
            return int(answer)
        elif expected_type == float:
            return float(answer)
        elif expected_type == str:
            if answer and not answer.isspace():
                return str(answer)
            else:
                return None
    except ValueError:
        return None

# --- UI Rendering ---
if not st.session_state.chat_started:
    st.button("Start Chat", on_click=start_chat)
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process user input
    if prompt := st.chat_input("Your answer"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        current_q_index = st.session_state.current_question
        field = fields[current_q_index]
        
        # Validate input
        user_input = get_valid_input(prompt, field["type"])
        if user_input is None:
            with st.chat_message("assistant"):
                st.markdown(f"⚠️ Please enter a valid {field['type'].__name__}.")
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Please enter a valid {field['type'].__name__}."})
        else:
            st.session_state.preferences[field["key"]] = user_input
            st.session_state.current_question += 1

            if st.session_state.current_question < len(fields):
                # Ask next question
                next_q = fields[st.session_state.current_question]["question"]
                llm_question = get_llm_question(next_q)
                with st.chat_message("assistant"):
                    st.markdown(llm_question)
                st.session_state.messages.append({"role": "assistant", "content": llm_question})
            else:
                # All questions answered, get recommendations
                with st.spinner("Analyzing your preferences and finding the best match..."):
                    user_prefs_str = "\\n".join([f"{k}: {v}" for k, v in st.session_state.preferences.items()])
                    recommendations = get_recommendations_from_preferences(table, user_prefs_str)
                    
                    if recommendations:
                        best_listing_description = select_best_listing_and_describe(user_prefs_str, recommendations)
                        
                        st.session_state.messages.append({"role": "assistant", "content": "Here are the top recommendations for you:"})
                        with st.chat_message("assistant"):
                            st.markdown("Here are the top recommendations for you:")

                        for rec in recommendations:
                            pid, title, address, price, desc, img_url = rec
                            with st.chat_message("assistant"):
                                st.image(img_url, caption=f"Image of {title}", width=200)
                                st.markdown(f"**{title}**")
                                st.markdown(f"**Address:** {address}")
                                st.markdown(f"**Price:** ${price:,.2f} CAD")
                                st.markdown(f"**Description:** {desc}")

                        st.session_state.messages.append({"role": "assistant", "content": best_listing_description})
                        with st.chat_message("assistant"):
                             st.markdown(best_listing_description)
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Sorry, I couldn't find any listings that match your criteria."})
                        with st.chat_message("assistant"):
                            st.markdown("Sorry, I couldn't find any listings that match your criteria.")
                
                st.button("Start Over", on_click=restart_chat)

    if st.session_state.current_question >= len(fields):
        if st.button("Restart Conversation", on_click=restart_chat):
            pass 