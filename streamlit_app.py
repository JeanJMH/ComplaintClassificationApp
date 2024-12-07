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
# Title and Description
st.title("💬 Financial Complaint Classifier")
st.write("A chatbot to classify customer complaints and create Jira tasks if needed.")

# Initialize Session State
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=10, return_messages=True)

if "classification_results" not in st.session_state:
    st.session_state.classification_results = {}

if "conversation_closed" not in st.session_state:
    st.session_state.conversation_closed = False

if "ready_to_submit" not in st.session_state:
    st.session_state.ready_to_submit = False

# Load Dataset
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"
try:
    df1 = pd.read_csv(url)
    st.write("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

product_categories = df1['Product'].unique()

# Initialize OpenAI Chat
try:
    model_type = "gpt-4o-mini"
    chat = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model=model_type)
except KeyError:
    st.error("API key missing! Please set 'OPENAI_API_KEY' in your Streamlit secrets.")
    st.stop()

# Helper function to analyze input details
def evaluate_input_details(chat, user_input, memory_messages):
    """
    Analyze user input to identify missing details using memory context.
    """
    prompt = (
        f"You are a helpful assistant analyzing customer complaints. "
        f"Always greet the user warmly in the first interaction. "
        f"Based on the conversation so far:\n"
        f"{memory_messages}\n\n"
        f"The user just mentioned:\n'{user_input}'\n\n"
        f"Determine if the user has provided:\n"
        f"1. A product (e.g., credit card, savings account).\n"
        f"2. A specific issue or problem (e.g., 'fraudulent transactions', 'stolen card').\n\n"
        f"Respond naturally and warmly to acknowledge provided details and politely ask for any missing information. "
        f"Conclude data collection once sufficient details for classification (product and issue) are provided."
        f"If more information is not necessary, or the client say doesn't have more details say: thank you! and provide a summary.  Then say to user press'Submit'botton your to send the complaint and finalize."
    )
    return chat.predict(prompt).strip()

# Helper function for classification
def classify_complaint(chat, prompt):
    response = chat.predict(prompt).strip()
    return response

# Display Chat History
st.write("### Chat History")
for message in st.session_state.memory.chat_memory.messages:
    st.chat_message(message["role"]).write(message["content"])

if not st.session_state.conversation_closed:
    if user_input := st.chat_input("Describe your issue:"):
        st.session_state.memory.chat_memory.add_message({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Get memory messages for context
        memory_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.memory.chat_memory.messages])

        # Evaluate Input Details
        evaluation_response = evaluate_input_details(chat, user_input, memory_messages)
        st.session_state.memory.chat_memory.add_message({"role": "assistant", "content": evaluation_response})
        st.chat_message("assistant").write(evaluation_response)

        # Check if ready to submit
        if "sufficient details" in evaluation_response.lower():
            st.session_state.ready_to_submit = True
            st.chat_message("assistant").write(
                "Thank you! This is the summary of your complaint. Please review the summary below and press the 'Submit' button to finalize."
            )

# Submit Button Always Visible
if st.button("Submit"):
    try:
        memory_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.memory.chat_memory.messages])

               # Step 1: Classify by Product
        product_prompt = (
            f"Based on the user's complaint and the following conversation context:\n"
            f"{memory_messages}\n\n"
            f"You are a financial expert. Classify the user's complaint into one of these product categories: {product_categories.tolist()}.\n"
            f"Respond with the exact product name as listed."
        )
        assigned_product = classify_complaint(chat, product_prompt)
        st.write(f"Assigned Product: {assigned_product}")

        # Step 2: Classify by Sub-product
        subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
        subproduct_prompt = (
            f"Based on the user's complaint about '{assigned_product}' and the following context:\n"
            f"{memory_messages}\n\n"
            f"You are a financial expert. Classify the user's complaint into one of these sub-product categories: {subproduct_options.tolist()}.\n"
            f"Respond with the exact sub-product name as listed."
        )
        assigned_subproduct = classify_complaint(chat, subproduct_prompt)
        st.write(f"Assigned Sub-product: {assigned_subproduct}")

        # Step 3: Classify by Issue
        issue_options = df1[
            (df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)
        ]['Issue'].unique()
        issue_prompt = (
            f"Based on the user's complaint about '{assigned_product}' -> '{assigned_subproduct}' and the following context:\n"
            f"{memory_messages}\n\n"
            f"You are a financial expert. Classify the user's complaint into one of these issue categories: {issue_options.tolist()}.\n"
            f"Respond with the exact issue name as listed."
        )
        assigned_issue = classify_complaint(chat, issue_prompt)
        st.write(f"Assigned Issue: {assigned_issue}")

        # Store results
        st.session_state.classification_results = {
            "Product": assigned_product,
            "Sub-product": assigned_subproduct,
            "Issue": assigned_issue,
        }

        # Display Results and Close Conversation
        classification_summary = (
            f"Classification Results:\n"
            f"- **Product**: {assigned_product}\n"
            f"- **Sub-product**: {assigned_subproduct}\n"
            f"- **Issue**: {assigned_issue}\n\n"
            f"Thank you for submitting your complaint. Our support team will get back to you shortly!"
        )
        st.session_state.memory.chat_memory.add_message({"role": "assistant", "content": classification_summary})
        st.chat_message("assistant").write(classification_summary)
        st.session_state.conversation_closed = True

    except Exception as e:
        error_message = f"Error during classification: {e}"
        st.chat_message("assistant").write(error_message)

# Summary Button
if st.button("Show Classification Summary"):
    st.write("### Classification Summary")
    st.write(f"- **Product**: {st.session_state.classification_results.get('Product', 'N/A')}")
    st.write(f"- **Sub-product**: {st.session_state.classification_results.get('Sub-product', 'N/A')}")
    st.write(f"- **Issue**: {st.session_state.classification_results.get('Issue', 'N/A')}")

