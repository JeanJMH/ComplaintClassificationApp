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

# Dataset URL
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
    model_type = "gpt-4o-mini"
    max_number_of_exchanges = 10

    # Initialize memory
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=max_number_of_exchanges, return_messages=True)

    # Initialize LLM
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

    # Tools
    @tool
    def datetoday(dummy: str) -> str:
        """Returns today's date."""
        return "Today is " + str(date.today())

    tools = [datetoday]

    # Create agent with memory
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"You are a financial support assistant. Begin by greeting the user warmly and asking them to describe their issue. "
                       f"Classify the complaint strictly based on these possible categories: {product_categories}. "
                       f"Inform the user that a ticket has been created, provide the assigned category, and reassure them. "
                       f"Maintain a professional and empathetic tone throughout."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

if "identified_product" not in st.session_state:
    st.session_state.identified_product = None
if "identified_subproduct" not in st.session_state:
    st.session_state.identified_subproduct = None
if "identified_issue" not in st.session_state:
    st.session_state.identified_issue = None

# Display chat history
st.write("### Chat History")
for message in st.session_state.memory.buffer:
    if isinstance(message, dict) and "type" in message and "content" in message:
        st.chat_message(message["type"]).write(message["content"])
    else:
        st.warning("Unexpected message format in chat history.")

# Chat input
if prompt := st.chat_input("How can I help?"):
    st.chat_message("user").write(prompt)
    st.session_state.memory.buffer.append({"type": "user", "content": prompt})

    # Generate a response using the agent
    try:
        response = st.session_state.agent_executor.invoke({"input": prompt})['output']
        st.chat_message("assistant").write(response)

        # Extract and process the identified product
        identified_product = None
        for category in product_categories:
            if category.lower() in response.lower():
                identified_product = category
                st.session_state.identified_product = category
                break

        if identified_product:
            # Filter subproducts for the identified product
            subproducts = df1[df1['Product'] == identified_product]['Sub-product'].unique().tolist()

            # Subproduct prompt
            subproduct_prompt = (
                f"The user described the following issue: '{prompt}'. Based on the description, "
                f"please identify the most relevant subproduct from the following list: {subproducts}. "
                "If none of the subproducts match exactly, respond with the most general category."
            )
            subproduct_response = st.session_state.agent_executor.invoke({"input": subproduct_prompt})['output']
            identified_subproduct = None

            for subproduct in subproducts:
                if subproduct.lower() in subproduct_response.lower():
                    identified_subproduct = subproduct
                    st.session_state.identified_subproduct = identified_subproduct
                    break

            # Filter issues for the identified product and subproduct
            issues = df1[
                (df1['Product'] == identified_product) & (df1['Sub-product'] == identified_subproduct)
            ]['Issue'].unique().tolist()

            # Issue prompt
            issue_prompt = (
                f"The user described the following issue: '{prompt}'. Based on the description, "
                f"please identify the most relevant issue from the following list: {issues}. "
                "If none of the issues match exactly, respond with the most general category."
            )
            issue_response = st.session_state.agent_executor.invoke({"input": issue_prompt})['output']
            identified_issue = None

            for issue in issues:
                if issue.lower() in issue_response.lower():
                    identified_issue = issue
                    st.session_state.identified_issue = identified_issue
                    break

            # Acknowledge the user
            unified_response = (
                f"Thank you! Your complaint has been categorized under **{identified_product}**, "
                f"subcategory: **{identified_subproduct}**, with the issue: **{identified_issue}**. "
                "A ticket has been created, and our support team will assist you shortly."
            )
            st.chat_message("assistant").write(unified_response)

        else:
            st.chat_message("assistant").write(response)

    except Exception as e:
        st.error(f"Error processing input: {e}")

# Sidebar display
if st.session_state.identified_product:
    st.sidebar.write(f"Stored Product: {st.session_state.identified_product}")
if st.session_state.identified_subproduct:
    st.sidebar.write(f"Stored Subproduct: {st.session_state.identified_subproduct}")
if st.session_state.identified_issue:
    st.sidebar.write(f"Stored Issue: {st.session_state.identified_issue}")
