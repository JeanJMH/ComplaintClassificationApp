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
st.title("💬 Financial Support Chatbot")
st.write("A chatbot to classify customer complaints and assist users effectively.")

# Load Dataset
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"
try:
    df1 = pd.read_csv(url)
    st.write("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

product_categories = df1['Product'].unique().tolist()

# Initialize Session State
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=10, return_messages=True)

if "classification_results" not in st.session_state:
    st.session_state.classification_results = {}

if "conversation_closed" not in st.session_state:
    st.session_state.conversation_closed = False

if "agent_executor" not in st.session_state:
    # Initialize Chat Model
    try:
        chat = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini")
    except KeyError:
        st.error("API key missing! Please set 'OPENAI_API_KEY' in your Streamlit secrets.")
        st.stop()

    # Define Tools
    @tool
    def datetoday(dummy: str) -> str:
        """Returns today's date."""
        return "Today is " + str(date.today())

    tools = [datetoday]

    # Create Prompt for Agent
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"You are a financial support assistant. Start by greeting the user warmly and asking them to describe their issue. "
                       f"Ensure the user provides both:\n"
                       f"1. A product (e.g., 'credit card', 'savings account').\n"
                       f"2. A specific issue (e.g., 'fraudulent transactions', 'stolen card').\n"
                       f"Politely ask for missing details if necessary.\n\n"
                       f"Once both product and issue are provided, classify the complaint strictly based on these possible categories: {product_categories}. "
                       f"Inform the user that a ticket has been created, provide the assigned category, and reassure them. "
                       f"Maintain a professional and empathetic tone throughout."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create Agent Executor
    agent = create_tool_calling_agent(chat, tools, agent_prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

# Define a key in session state to store the identified product and subproduct
if "identified_product" not in st.session_state:
    st.session_state.identified_product = None
if "identified_subproduct" not in st.session_state:
    st.session_state.identified_subproduct = None

# Display Chat History
st.write("### Chat History")
for message in st.session_state.memory.buffer:
    st.chat_message(message["type"]).write(message["content"])

# Create a chat input field to allow the user to enter a message
if not st.session_state.conversation_closed:
    if user_input := st.chat_input("Describe your issue:"):
        # User input handling
        st.chat_message("user").write(user_input)
        st.session_state.memory.buffer.append({"type": "user", "content": user_input})

        try:
            # Generate response using agent executor
            response = st.session_state.agent_executor.invoke({"input": user_input})['output']
            st.chat_message("assistant").write(response)

            # Extract identified product from response
            identified_product = None
            for category in product_categories:
                if category.lower() in response.lower():
                    identified_product = category
                    st.session_state.identified_product = category
                    break

            # Ask for more details if issue is missing
            if not st.session_state.identified_product:
                st.chat_message("assistant").write("Could you provide more details about the issue?")
            else:
                st.chat_message("assistant").write(
                    f"Thank you for providing the details of your issue with **{identified_product}**. Let me categorize your complaint."
                )

                # Further processing (Subproduct and Issue categorization)
                subproducts = df1[df1['Product'] == identified_product]['Sub-product'].unique().tolist()
                subproduct_prompt = (
                    f"The user described the following issue: '{user_input}'. "
                    f"Identify the most relevant subproduct from this list: {subproducts}."
                )
                assigned_subproduct = chat.predict(subproduct_prompt).strip()
                st.session_state.identified_subproduct = assigned_subproduct

                issues = df1[
                    (df1['Product'] == identified_product) & (df1['Sub-product'] == assigned_subproduct)
                ]['Issue'].unique().tolist()
                issue_prompt = (
                    f"The user described the following issue: '{user_input}'. "
                    f"Identify the most relevant issue from this list: {issues}."
                )
                assigned_issue = chat.predict(issue_prompt).strip()
                st.session_state.identified_issue = assigned_issue

                # Display final results
                final_response = (
                    f"Your complaint has been categorized as:\n"
                    f"- **Product**: {identified_product}\n"
                    f"- **Sub-product**: {assigned_subproduct}\n"
                    f"- **Issue**: {assigned_issue}\n\n"
                    f"A ticket has been created, and our support team will get back to you shortly!"
                )
                st.chat_message("assistant").write(final_response)
                st.session_state.conversation_closed = True

        except Exception as e:
            st.error(f"Error processing input: {e}")

# Consolidate sidebar display
if st.session_state.identified_product:
    st.sidebar.write(f"Stored Product: {st.session_state.identified_product}")
if "identified_subproduct" in st.session_state:
    st.sidebar.write(f"Stored Subproduct: {st.session_state.identified_subproduct}")
if "identified_issue" in st.session_state:
    st.sidebar.write(f"Stored Issue: {st.session_state.identified_issue}")
