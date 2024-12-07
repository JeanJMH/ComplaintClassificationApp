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



st.title("💬 Financial Support Chatbot")

# Dataset URL
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"

# Load dataset
if url:
    try:
        df1 = pd.read_csv(url)
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")

product_categories = df1['Product'].unique().tolist()

# Session state initialization
if "memory" not in st.session_state:
    model_type = "gpt-4o-mini"
    max_number_of_exchanges = 10

    # Initialize memory
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        k=max_number_of_exchanges, 
        return_messages=True
    )

    # LLM initialization
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

    # Define a simple tool
    @tool
    def datetoday(dummy: str) -> str:
        """Returns today's date."""
        return "Today is " + str(date.today())

    tools = [datetoday]

    # Define prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            {
                "role": "system",
                "content": (
                    f"You are a financial support assistant. Begin by greeting the user warmly. "
                    f"Ask them to describe their issue, ensuring they specify the product and the issue. "
                    f"If either is missing, request the missing information. "
                    f"Once all details are provided, classify the complaint into product, subproduct, and issue categories, "
                    f"and confirm with the user. Inform them that a ticket has been created and reassure them the support team will contact them."
                )
            },
            {"role": "system", "content": "{chat_history}"},
            {"role": "user", "content": "{input}"},
            {"role": "system", "content": "{agent_scratchpad}"}
        ]
    )

    # Create agent
    agent = create_tool_calling_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

# Initialize session state for identified details
if "identified_product" not in st.session_state:
    st.session_state.identified_product = None
if "identified_subproduct" not in st.session_state:
    st.session_state.identified_subproduct = None
if "identified_issue" not in st.session_state:
    st.session_state.identified_issue = None

# Display existing chat messages
for message in st.session_state.memory.buffer:
    st.chat_message(message.type).write(message.content)

# Chat input field
if prompt := st.chat_input("How can I help?"):
    # Display user input
    st.chat_message("user").write(prompt)

    # Step 1: Identify product
    if not st.session_state.identified_product:
        response = st.session_state.agent_executor.invoke({
            "input": f"The user said: '{prompt}'. Could you identify the product from this description? Available options: {product_categories}."
        })['output']

        for category in product_categories:
            if category.lower() in response.lower():
                st.session_state.identified_product = category
                st.chat_message("assistant").write(f"Thank you! I’ve identified the product as: **{category}**.")
                break

    # Step 2: Identify subproduct
    if st.session_state.identified_product and not st.session_state.identified_subproduct:
        subproducts = df1[df1['Product'] == st.session_state.identified_product]['Sub-product'].unique().tolist()
        response = st.session_state.agent_executor.invoke({
            "input": f"The user described: '{prompt}'. Could you identify the subproduct? Available options: {subproducts}."
        })['output']

        for subproduct in subproducts:
            if subproduct.lower() in response.lower():
                st.session_state.identified_subproduct = subproduct
                st.chat_message("assistant").write(f"Thank you! I’ve identified the subproduct as: **{subproduct}**.")
                break

    # Step 3: Identify issue
    if st.session_state.identified_product and st.session_state.identified_subproduct and not st.session_state.identified_issue:
        issues = df1[
            (df1['Product'] == st.session_state.identified_product) &
            (df1['Sub-product'] == st.session_state.identified_subproduct)
        ]['Issue'].unique().tolist()
        response = st.session_state.agent_executor.invoke({
            "input": f"The user described: '{prompt}'. Could you identify the issue? Available options: {issues}."
        })['output']

        for issue in issues:
            if issue.lower() in response.lower():
                st.session_state.identified_issue = issue
                st.chat_message("assistant").write(f"Thank you! I’ve identified the issue as: **{issue}**.")
                break

    # Final confirmation and response
    if st.session_state.identified_product and st.session_state.identified_subproduct and st.session_state.identified_issue:
        st.chat_message("assistant").write(
            f"Based on your description, your complaint has been categorized under: **{st.session_state.identified_product}**, "
            f"subcategory: **{st.session_state.identified_subproduct}**, and issue: **{st.session_state.identified_issue}**. "
            "A ticket has been created, and our support team will reach out to you shortly. If you have more questions, feel free to ask!"
        )
    else:
        st.chat_message("assistant").write("I’m sorry, I couldn’t fully categorize your issue. Could you provide more details?")

# Sidebar display
if st.session_state.identified_product:
    st.sidebar.write(f"Stored Product: {st.session_state.identified_product}")
if st.session_state.identified_subproduct:
    st.sidebar.write(f"Stored Subproduct: {st.session_state.identified_subproduct}")
if st.session_state.identified_issue:
    st.sidebar.write(f"Stored Issue: {st.session_state.identified_issue}")

