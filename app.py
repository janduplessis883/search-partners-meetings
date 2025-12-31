import streamlit as st
import pandas as pd
from airweave import AirweaveSDK, SearchRequest
from groq import Groq

# Initialize Groq client
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# LLM Models
MODEL_OPTIONS = [
    "moonshotai/kimi-k2-instruct-0905",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
]


# Simple function to get a response from Groq
def ask_groq(prompt: str, model: str = "openai/gpt-oss-120b"):
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )

    try:
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting response from Groq: {e}")
        return "Error: Could not get a response from Groq."


# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Initialize session state for query and results
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "show_results" not in st.session_state:
    st.session_state.show_results = False



# PIN Authentication
if not st.session_state.authenticated:
    st.title("Access Required")
    with st.form("pin_form"):
        pin_input = st.text_input("Enter PIN:", type="password")
        pin_submit = st.form_submit_button("Submit")

    if pin_submit:
        if pin_input == st.secrets["PIN"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect PIN. Please try again.")
    st.stop()

# Main app starts here (only shown after authentication)
AIRWEAVER_API_KEY = st.secrets["AIRWEAVER_API_KEY"]
airweave = AirweaveSDK(
    framework_name="streamlit",
    framework_version="1.0",
    api_key=AIRWEAVER_API_KEY,
)


st.title(":material/search: Partners' Meetings")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    num_sources = st.number_input("Number of Sources (n):", min_value=1, max_value=10, value=5)
    selected_model = st.selectbox("Select LLM Model:", MODEL_OPTIONS, index=3)

# Create a form for user input
with st.form("search_form", border=False):
    query = st.text_input("Enter your search query:", value=st.session_state.query_input)
    col1, col2, col3 = st.columns([1,1,4], gap='small')
    with col1:
        submit_button = st.form_submit_button(":material/search: Search")
    with col2:
        clear_button = st.form_submit_button(":material/clear: Clear")
    with col3:
        pass





# Handle clear button
if clear_button:
    st.session_state.query_input = ""
    st.session_state.show_results = False
    st.rerun()

# Process the query when form is submitted
if submit_button and query:
    st.session_state.query_input = query
    st.session_state.show_results = True

    with st.spinner("Searching..."):
        results = airweave.collections.search(
            readable_id='stanhope-meetings-brxsu8',
            request=SearchRequest(
                query=query,
            ),
        )

    if results.results:
        # Sort results by score (highest first) and take top n sources
        sorted_results = sorted(
            results.results,
            key=lambda x: x.get('score', 0) if isinstance(x, dict) else getattr(x, 'score', 0),
            reverse=True
        )
        top_results = sorted_results[:int(num_sources)]

        # Format results for the LLM
        sources_text = "\n\n".join(
            [f"Source {i+1} (Score: {result.get('score', 0)}):\n{result.get('payload', {}).get('textual_representation', str(result))}"
             for i, result in enumerate(top_results)]
        )


        # Call Groq LLM with the query and top sources
        with st.spinner("Processing with AI..."):
            try:
                prompt = f"""Based on the following search query and source materials, provide a comprehensive answer.

Search Query: {query}

Source Materials:
{sources_text}

Please synthesize the information from these sources to answer the search query thoroughly. If the answer to the questions is not in the source material say so."""

                llm_response = ask_groq(prompt, selected_model)

                st.markdown("### AI Generated Answer")
                st.markdown(llm_response)

                st.divider()
                st.markdown("### Source Materials Used")
                for i, result in enumerate(top_results, 1):
                    score = result.get('score', 0) if isinstance(result, dict) else getattr(result, 'score', 0)
                    payload = result.get('payload', {}) if isinstance(result, dict) else getattr(result, 'payload', {})
                    title = payload.get('title', 'Untitled')
                    textual_rep = payload.get('textual_representation', str(result))
                    with st.expander(f"Source {i} - {title} (Score: {score:.2f})"):
                        st.markdown(textual_rep)

            except Exception as e:
                st.error(f"Error processing with AI: {str(e)}")
    else:
        st.markdown("No results found.")
