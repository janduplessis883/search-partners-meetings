import streamlit as st
import pandas as pd
from airweave import AirweaveSDK, SearchRequest
from groq import Groq
import re

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


# Initialize session state for query and results
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# Main app starts here (only shown after authentication)
AIRWEAVER_API_KEY = st.secrets["AIRWEAVER_API_KEY"]
airweave = AirweaveSDK(
    framework_name="streamlit",
    framework_version="1.0",
    api_key=AIRWEAVER_API_KEY,
)


st.subheader(":material/search: Chat with Stanhope Policies")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Select LLM Model:", MODEL_OPTIONS, index=3)
    st.caption("Use **Kimi-K2** for a shorter answer & **GPT-OSS** for detailed answers.")
    st.divider()
    st.caption("Defualt setting are for optimal performance, adjust if you get an error.")
    num_sources = st.number_input("Number of Sources (n):", min_value=1, max_value=10, value=3)
    retrieval_strategy = st.selectbox("Retrieval Strategy:", ["hybrid", "vector", "keyword"], index=0, help="The retrieval strategy to use")
    temporal_relevance = st.number_input("Temporal Relevance:", min_value=0.0, max_value=1.0, step=0.1,value=0.8, help="Weight recent content higher than older content; 0 = no recency effect, 1 = only recent items matter")
    expand_query = st.toggle("Expand Query", value=True, help="Generate a few query variations to improve recall")
    rerank = st.toggle("Rerank Results", value=True, help="Reorder the top candidate results for improved relevance. Max number of results that can be reranked is capped to around 1000.")
    generate_answer = st.toggle("Generate Answer", value=False, help="Generate a natural-language answer to the query")



# Create a form for user input
with st.form("search_form", border=False):
    query = st.text_input("Ask me anything about policies:", value=st.session_state.query_input)
    col1, col2, col3 = st.columns([1.5,1,3], gap='small')
    with col1:
        submit_button = st.form_submit_button(":material/search: Search", type="primary", use_container_width=True)
    with col2:
        clear_button = st.form_submit_button(":material/clear: Clear", use_container_width=True)
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

    with st.spinner("Searching...", show_time=True):
        results = airweave.collections.search(
            readable_id='stanhope-policies-lotv7i',
            request=SearchRequest(
                query=query,
                temporal_relevance=temporal_relevance,
                retrieval_strategy=retrieval_strategy,
                expand_query=expand_query,
                rerank=rerank,
                generate_answer=generate_answer,
                limit=num_sources,
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


        # Display AI Generated Answer
        with st.spinner("Processing with AI...", show_time=True):
            try:
                st.markdown("### :material/robot_2: AI Generated Answer")

                if generate_answer:
                    # Extract completion from API response
                    # Try different ways to access completion field
                    api_completion = None

                    # Try as attribute
                    if hasattr(results, 'completion'):
                        api_completion = results.completion
                    # Try as dict key
                    elif isinstance(results, dict) and 'completion' in results:
                        api_completion = results['completion']
                    # Try converting to dict
                    elif hasattr(results, '__dict__'):
                        results_dict = vars(results)
                        api_completion = results_dict.get('completion')

                    if api_completion:
                        with st.container(border=True):
                            st.markdown(api_completion)
                    else:
                        st.warning("No completion found in API response")
                        st.write("Response structure:")
                        st.json(str(results))
                else:
                    # Call Groq LLM with the query and top sources
                    prompt = f"""Based on the following search query and source materials, provide a comprehensive answer.

Search Query: {query}

Source Materials:
{sources_text}

Please synthesize the information from these sources to answer the search query thoroughly. If the answer to the questions is not in the source material say so.
Always reference the policy name and Date Last Edited in your answer where applicable.
Format your answer in markdown."""

                    llm_response = ask_groq(prompt, selected_model)
                    with st.container(border=True):
                            # extract reasoning separately if you still want to make it optional
                            match = re.search(r"<think>(.*?)</think>", llm_response, flags=re.DOTALL)
                            reasoning = match.group(1).strip() if match else None
                            visible_text = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL)
                            if reasoning:
                                with st.expander("Show hidden reasoning", icon=":material/neurology:"):
                                    st.markdown(f"{reasoning}")
                            st.markdown(f"{visible_text}")


                st.divider()
                st.markdown("### :material/save: Source Materials Used")
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
