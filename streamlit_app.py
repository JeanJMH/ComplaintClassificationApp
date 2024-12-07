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
from langchain.agents import tool

# Title and Description
st.title("💬 Financial Complaint Classifier")
st.write("A chatbot to classify customer complaints with enhanced user interaction.")

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

product_categories = df1['Product'].unique().tolist()

# Initialize OpenAI Chat
try:
    chat = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini")
except KeyError:
    st.error("API key missing! Please set 'OPENAI_API_KEY' in your Streamlit secrets.")
    st.stop()

# Define tools
@tool
def datetoday(dummy: str) -> str:
    """Returns today's date."""
    return "Today is " + str(date.today())

tools = [datetoday]

# Create the agent
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

        # Use the agent for classification
        try:
            response = st.session_state.agent_executor.invoke({"input": user_input})['output']
            st.chat_message("assistant").write(response)

            # Extract classification details
            assigned_product = None
            for category in product_categories:
                if category.lower() in response.lower():
                    assigned_product = category
                    st.session_state.classification_results["Product"] = assigned_product
                    break

            if assigned_product:
                subproducts = df1[df1['Product'] == assigned_product]['Sub-product'].unique().tolist()
                subproduct_prompt = (
                    f"The user described the following issue: '{user_input}'. Based on the description, "
                    f"please identify the most relevant subproduct from the following list: {subproducts}. "
                )
                assigned_subproduct = classify_complaint(chat, subproduct_prompt)
                st.session_state.classification_results["Sub-product"] = assigned_subproduct

                issues = df1[
                    (df1['Product'] == assigned_product) & (df1['Sub-product'] == assigned_subproduct)
                ]['Issue'].unique().tolist()
                issue_prompt = (
                    f"The user described the following issue: '{user_input}'. Based on the description, "
                    f"please identify the most relevant issue from the following list: {issues}."
                )
                assigned_issue = classify_complaint(chat, issue_prompt)
                st.session_state.classification_results["Issue"] = assigned_issue

                # Acknowledge the user
                summary = (
                    f"Classification Results:\n"
                    f"- **Product**: {assigned_product}\n"
                    f"- **Sub-product**: {assigned_subproduct}\n"
                    f"- **Issue**: {assigned_issue}\n\n"
                    f"Thank you for submitting your complaint. Our support team will get back to you shortly!"
                )
                st.chat_message("assistant").write(summary)
                st.session_state.conversation_closed = True
            else:
                st.chat_message("assistant").write("Unable to classify the complaint. Please try again with more details.")

        except Exception as e:
            st.error(f"Error during classification: {e}")

# Summary Button Always Visible
if st.button("Show Classification Summary"):
    st.write("### Classification Summary")
    st.write(f"- **Product**: {st.session_state.classification_results.get('Product', 'N/A')}")
    st.write(f"- **Sub-product**: {st.session_state.classification_results.get('Sub-product', 'N/A')}")
    st.write(f"- **Issue**: {st.session_state.classification_results.get('Issue', 'N/A')}")
