import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from datetime import date
import pandas as pd
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain import hub
# Show title and description
st.title("💬 Financial Support Chatbot")
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"

# Load the dataset if a valid URL is provided
if url:
    try:
        df1 = pd.read_csv(url)
        st.write("Dataset loaded successfully.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

product_categories = df1['Product'].unique().tolist()

# Initialize session state variables
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=10, return_messages=True)

if "conversation_closed" not in st.session_state:
    st.session_state.conversation_closed = False

if "ready_to_submit" not in st.session_state:
    st.session_state.ready_to_submit = False

# Initialize OpenAI Chat
try:
    model_type = "gpt-4o-mini"
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)
except KeyError:
    st.error("API key missing! Please set 'OPENAI_API_KEY' in your Streamlit secrets.")
    st.stop()

# Helper function for detailed interaction
def evaluate_input_details(chat, user_input, memory_messages):
    """
    Analyze user input to identify missing details using memory context.
    """
    prompt = (
        f"You are a helpful assistant analyzing customer complaints. "
        f"Based on the conversation so far:\n"
        f"{memory_messages}\n\n"
        f"The user just mentioned:\n'{user_input}'\n\n"
        f"Determine if the user has provided:\n"
        f"1. A product (e.g., credit card, savings account).\n"
        f"2. A specific issue or problem (e.g., 'fraudulent transactions', 'stolen card').\n\n"
        f"Respond naturally and warmly to acknowledge provided details and politely ask for any missing information. "
        f"Conclude data collection once sufficient details for classification (product and issue) are provided."
    )
    return chat.predict(prompt).strip()

# Display chat history
for message in st.session_state.memory.buffer:
    st.chat_message(message["type"]).write(message["content"])

if not st.session_state.conversation_closed:
    # User input handling
    if user_input := st.chat_input("Describe your issue:"):
        st.session_state.memory.buffer.append({"type": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Context from chat history
        memory_messages = "\n".join([f"{msg['type']}: {msg['content']}" for msg in st.session_state.memory.buffer])

        # Evaluate and request missing information
        evaluation_response = evaluate_input_details(chat, user_input, memory_messages)
        st.session_state.memory.buffer.append({"type": "assistant", "content": evaluation_response})
        st.chat_message("assistant").write(evaluation_response)

        # Check if ready to submit
        if "sufficient details" in evaluation_response.lower():
            st.session_state.ready_to_submit = True
            st.chat_message("assistant").write(
                "Thank you! Please press the 'Submit' button to finalize your complaint."
            )

# Submit Button
if st.button("Submit"):
    try:
        memory_messages = "\n".join([f"{msg['type']}: {msg['content']}" for msg in st.session_state.memory.buffer])

        # Classification Prompts
        product_prompt = (
            f"Based on the user's complaint and the following context:\n"
            f"{memory_messages}\n\n"
            f"Classify the user's complaint into one of these product categories: {product_categories}."
        )
        assigned_product = chat.predict(product_prompt).strip()
        st.write(f"Assigned Product: {assigned_product}")

        subproducts = df1[df1['Product'] == assigned_product]['Sub-product'].unique().tolist()
        subproduct_prompt = (
            f"Based on the user's complaint about '{assigned_product}' and the following context:\n"
            f"{memory_messages}\n\n"
            f"Classify the user's complaint into one of these sub-product categories: {subproducts}."
        )
        assigned_subproduct = chat.predict(subproduct_prompt).strip()
        st.write(f"Assigned Sub-product: {assigned_subproduct}")

        issues = df1[
            (df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)
        ]['Issue'].unique().tolist()
        issue_prompt = (
            f"Based on the user's complaint about '{assigned_product}' -> '{assigned_subproduct}' and the context:\n"
            f"{memory_messages}\n\n"
            f"Classify the user's complaint into one of these issue categories: {issues}."
        )
        assigned_issue = chat.predict(issue_prompt).strip()
        st.write(f"Assigned Issue: {assigned_issue}")

        # Classification Results
        st.session_state.conversation_closed = True
        classification_summary = (
            f"- **Product**: {assigned_product}\n"
            f"- **Sub-product**: {assigned_subproduct}\n"
            f"- **Issue**: {assigned_issue}\n"
        )
        st.chat_message("assistant").write(
            f"Thank you for your submission! Here are the details:\n{classification_summary}"
        )

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
