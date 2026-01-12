"""Streamlit application for Text-to-SQL chatbot with BigQuery and Gemini Pro."""

import os
import streamlit as st
from dotenv import load_dotenv

from src.bq_client import BigQueryClient
from src.llm_logic import GenAIChain

load_dotenv()

st.set_page_config(
    page_title="Text-to-SQL Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .sql-box {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        overflow-x: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "bq_client" not in st.session_state:
        st.session_state.bq_client = None
    if "genai_chain" not in st.session_state:
        st.session_state.genai_chain = None
    if "connected" not in st.session_state:
        st.session_state.connected = False


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("Configuration")
        st.markdown("---")

        st.subheader("BigQuery Settings")
        project_id = st.text_input(
            "Project ID",
            value=os.getenv("BQ_PROJECT_ID", ""),
            help="Your Google Cloud Project ID",
        )
        dataset_id = st.text_input(
            "Dataset ID",
            value=os.getenv("BQ_DATASET_ID", ""),
            help="The BigQuery dataset to query",
        )

        st.markdown("---")
        st.subheader("Credentials")

        creds_path = st.text_input(
            "Service Account Path",
            value=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
            help="Path to your service account JSON file",
        )
        api_key = st.text_input(
            "Gemini API Key",
            value=os.getenv("GOOGLE_API_KEY", ""),
            type="password",
            help="Your Google AI Studio API key",
        )

        st.markdown("---")

        connect_btn = st.button("Connect", use_container_width=True)

        if connect_btn:
            if not all([project_id, dataset_id, creds_path, api_key]):
                st.error("Please fill in all configuration fields.")
            else:
                with st.spinner("Connecting to BigQuery..."):
                    try:
                        bq_client = BigQueryClient(
                            project_id=project_id,
                            credentials_path=creds_path,
                        )
                        bq_client.test_connection()

                        connection_uri = bq_client.get_connection_uri(dataset_id)
                        genai_chain = GenAIChain(
                            connection_uri=connection_uri,
                            google_api_key=api_key,
                        )

                        st.session_state.bq_client = bq_client
                        st.session_state.genai_chain = genai_chain
                        st.session_state.connected = True
                        st.session_state.project_id = project_id
                        st.session_state.dataset_id = dataset_id

                        st.success("Connected successfully!")
                    except FileNotFoundError as e:
                        st.error(f"Credentials file not found: {e}")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")

        if st.session_state.connected:
            st.markdown("---")
            st.subheader("Connection Status")
            st.success(f"**Project:** {st.session_state.project_id}")
            st.success(f"**Dataset:** {st.session_state.dataset_id}")

            if st.button("Show Tables", use_container_width=True):
                try:
                    tables = st.session_state.genai_chain.get_table_names()
                    st.write("**Available Tables:**")
                    for table in tables:
                        st.write(f"- {table}")
                except Exception as e:
                    st.error(f"Failed to list tables: {e}")

            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


def render_chat_interface():
    """Render the main chat interface."""
    st.title("Text-to-SQL Chatbot")
    st.markdown("Ask questions about your BigQuery data in natural language.")

    if not st.session_state.connected:
        st.info("Please configure your connection in the sidebar to get started.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sql_query" in message:
                with st.expander("View SQL Query"):
                    st.code(message["sql_query"], language="sql")
            if "sql_result" in message:
                with st.expander("View Raw Results"):
                    st.text(message["sql_result"])

    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating SQL and querying database..."):
                try:
                    result = st.session_state.genai_chain.ask(prompt)

                    st.markdown(result["answer"])
                    with st.expander("View SQL Query"):
                        st.code(result["sql_query"], language="sql")
                    with st.expander("View Raw Results"):
                        st.text(result["sql_result"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sql_query": result["sql_query"],
                        "sql_result": result["sql_result"],
                    })

                except Exception as e:
                    error_message = f"**Error executing query:** {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                    })


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
