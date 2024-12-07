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
    st.write("Dataset loaded successfully. 🎉")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

product_categories = df1['Product'].unique()

# Initialize OpenAI Chat and Agent
try:
    model_type = "gpt-4o-mini"
    chat = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model=model_type)
    
    @tool
    def datetoday(dummy: str) -> str:
        """Returns today's date."""
        return "Today is " + str(date.today())

    tools = [datetoday]

    # Define agent prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"You are a financial support assistant. "
                       f"Guide users to classify their complaints. "
                       f"Classify complaints based on these categories: {product_categories}. "
                       f"Provide clear and empathetic feedback."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    agent = create_tool_calling_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)
except KeyError:
    st.error("API key missing! Please set 'OPENAI_API_KEY' in your Streamlit secrets.")
    st.stop()

# Display Chat History
st.write("### Chat History")
if "buffer" in st.session_state.memory and st.session_state.memory.buffer:
    for message in st.session_state.memory.buffer:
        if isinstance(message, dict) and "type" in message and "content" in message:
            st.chat_message(message["type"]).write(message["content"])
        else:
            st.write("Skipping invalid message:", message)
else:
    st.write("No chat history available.")

# User Input Handling
if not st.session_state.conversation_closed:
    if user_input := st.chat_input("Describe your issue:"):
        st.session_state.memory.chat_memory.add_message({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Generate agent response
        response = st.session_state.agent_executor.invoke({"input": user_input})['output']
        st.session_state.memory.chat_memory.add_message({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

        # Check if ready to submit
        if "sufficient details" in response.lower():
            st.session_state.ready_to_submit = True
            st.chat_message("assistant").write(
                "Thank you! This is the summary of your complaint. Please review the summary below and press the 'Submit' button to finalize."
            )

# Submit Button Always Visible
if st.button("Submit") and st.session_state.ready_to_submit:
    try:
        memory_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.memory.chat_memory.messages])

        # Step 1: Classify by Product
        product_prompt = (
            f"Based on the user's complaint and the following conversation context:\n"
            f"{memory_messages}\n\n"
            f"Classify the user's complaint into one of these product categories: {product_categories}."
        )
        assigned_product = st.session_state.agent_executor.invoke({"input": product_prompt})['output']
        st.write(f"**Assigned Product:** {assigned_product}")

        # Step 2: Classify by Sub-product
        subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
        subproduct_prompt = (
            f"Classify the complaint related to '{assigned_product}' into one of these sub-product categories: {subproduct_options}."
        )
        assigned_subproduct = st.session_state.agent_executor.invoke({"input": subproduct_prompt})['output']
        st.write(f"**Assigned Sub-product:** {assigned_subproduct}")

        # Step 3: Classify by Issue
        issue_options = df1[
            (df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)
        ]['Issue'].unique()
        issue_prompt = (
            f"Classify the complaint about '{assigned_product}' -> '{assigned_subproduct}' into one of these issues: {issue_options}."
        )
        assigned_issue = st.session_state.agent_executor.invoke({"input": issue_prompt})['output']
        st.write(f"**Assigned Issue:** {assigned_issue}")

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
