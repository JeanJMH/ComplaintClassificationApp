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
### Adding subproducts
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"
#st.write(url)

# Load the dataset if a valid URL is provided
if url:
    try:
        df1 = pd.read_csv(url)
        #st.write(df1)
    except Exception as e:
        st.error(f"An error occurred: {e}")

product_categories = df1['Product'].unique().tolist()

### Important part.
# Create a session state variable to flag whether the app has been initialized.
# This code will only be run the first time the app is loaded.
if "memory" not in st.session_state:
    model_type = "gpt-4o-mini"

    # Initialize the memory
    max_number_of_exchanges = 10
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=max_number_of_exchanges, return_messages=True)

    # LLM
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

    # Tools
    from langchain.agents import tool
    @tool
    def datetoday(dummy: str) -> str:
        """Returns today's date, use this for any \
        questions that need today's date to be answered. \
        This tool returns a string with today's date."""
        return "Today is " + str(date.today())

    tools = [datetoday]
    
    # Create the agent with memory
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"You are a financial support assistant. Begin by greeting the user warmly and asking them to describe their issue. Wait for the user to describe their problem. Once the issue is described, classify the complaint strictly based on these possible categories: {product_categories}. Kindly inform the user that a ticket has been created, provide the category assigned to their complaint, and reassure them that the issue will be forwarded to the appropriate support team, who will reach out to them shortly. Maintain a professional and empathetic tone throughout."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

# Define a key in session state to store the identified product and subproduct
if "identified_product" not in st.session_state:
    st.session_state.identified_product = None
if "identified_subproduct" not in st.session_state:
    st.session_state.identified_subproduct = None

# Display the existing chat messages via `st.chat_message`
for message in st.session_state.memory.buffer:
    st.chat_message(message.type).write(message.content)


#################


# Initialize subproduct_source and issue_source globally
subproduct_source = "No source identified"
issue_source = "No source identified"

# Create a chat input field to allow the user to enter a message
if prompt := st.chat_input("How can I help?"):
    
    # User message
    st.chat_message("user").write(prompt)

    # Generate a response using the OpenAI API
    response = st.session_state.agent_executor.invoke({"input": prompt})['output']
    
    # Extract the identified product category from the response
    identified_product = None
    for category in product_categories:
        if category.lower() in response.lower():
            identified_product = category
            st.session_state.identified_product = category
            break

    # Create a single unified response message
    if identified_product:
        # Filter the dataset to find subcategories for the identified product
        subproducts = df1[df1['Product'] == identified_product]['Sub-product'].unique().tolist()

        # Use the model to identify the best matching subproduct
        identified_subproduct = None
        if subproducts:
            # Create a prompt to evaluate the closest subproduct
            subproduct_prompt = (
                f"The user described the following issue: '{prompt}'. Based on the description, "
                f"please identify the most relevant subproduct from the following list: {subproducts}. "
                "If none of the subproducts match exactly, respond with the most general category."
            )

            # Invoke the model to determine the subproduct
            subproduct_response = st.session_state.agent_executor.invoke({"input": subproduct_prompt})['output']

            # Check if the model identified a valid subproduct
            for subproduct in subproducts:
                if subproduct.lower() in subproduct_response.lower():
                    identified_subproduct = subproduct
                    st.session_state.identified_subproduct = identified_subproduct
                    subproduct_source = "LLM"
                    break

            # Fallback: Select the first subproduct if none is confidently identified
            if not identified_subproduct:
                identified_subproduct = subproducts[0]
                st.session_state.identified_subproduct = identified_subproduct
                subproduct_source = "Fallback (most general category)"

        # Filter the dataset to find "Issues" for the identified product and subproduct
        issues = df1[(df1['Product'] == identified_product) & (df1['Sub-product'] == identified_subproduct)]['Issue'].unique().tolist()

        # Use the model to identify the most relevant "Issue"
        identified_issue = None
        if issues:
            # Create a prompt to evaluate the closest issue
            issue_prompt = (
                f"The user described the following issue: '{prompt}'. Based on the description, "
                f"please identify the most relevant issue from the following list: {issues}. "
                "If none of the issues match exactly, respond with the most general category."
            )

            # Invoke the model to determine the issue
            issue_response = st.session_state.agent_executor.invoke({"input": issue_prompt})['output']

            # Check if the model identified a valid issue
            for issue in issues:
                if issue.lower() in issue_response.lower():
                    identified_issue = issue
                    st.session_state.identified_issue = identified_issue
                    issue_source = "LLM"
                    break

            # Fallback: Select the first issue if none is confidently identified
            if not identified_issue:
                identified_issue = issues[0]
                st.session_state.identified_issue = identified_issue
                issue_source = "Fallback (most general category)"

        # Create acknowledgment message
        unified_response = (
            f"Thank you for providing the details of your issue. Based on your description, your complaint has been categorized under: **{identified_product}**, "
            f"specifically the subcategory: **{identified_subproduct}**, with the issue categorized as: **{identified_issue}**. A ticket has been created for your issue, and it will be forwarded to the appropriate support team. "
            "They will reach out to you shortly to assist you further. If you have any more questions or need additional assistance, please let me know!"
        )

        # Display acknowledgment message
        st.chat_message("assistant").write(unified_response)

        # Add a message to confirm the issue identification source
        #if issue_source == "LLM":
        #    st.write("The issue was directly identified by the model.")
        #else:
        #    st.write("The issue was not directly identified by the model. The most general category was selected.")

        # For troubleshooting purposes, print the identified product, subproduct, and issue
        #st.write("Troubleshooting: Identified Product, Subproduct, and Issue")
        #st.write(f"Product: {identified_product}")
        #st.write(f"Subproduct: {identified_subproduct if identified_subproduct else 'No subproduct identified'}")
        #st.write(f"Issue: {identified_issue if identified_issue else 'No issue identified'}")
        #st.write("Troubleshooting: List of issues for the identified product and subproduct:")
        #st.write(issues)

    else:
        st.chat_message("assistant").write(response)  # Default response when no category is identified

# Consolidate sidebar display here (only once)
if st.session_state.identified_product:
    st.sidebar.write(f"Stored Product: {st.session_state.identified_product}")
if "identified_subproduct" in st.session_state:
    st.sidebar.write(f"Stored Subproduct: {st.session_state.identified_subproduct}")
    #st.sidebar.write(f"Subproduct Identification Source: {subproduct_source}")
if "identified_issue" in st.session_state:
    st.sidebar.write(f"Stored Issue: {st.session_state.identified_issue}")
    #st.sidebar.write(f"Issue Identification Source: {issue_source}")


############Addign Jira



# Initialize Jira-related session states
if "jira_task_created" not in st.session_state:
    st.session_state.jira_task_created = False
if "jira_task_description" not in st.session_state:
    st.session_state.jira_task_description = None

# Jira task creation logic
if (
    "identified_product" in st.session_state
    and "identified_subproduct" in st.session_state
    and "identified_issue" in st.session_state
    and not st.session_state.jira_task_created
):
    st.write("Starting Jira task creation process...")  # Debugging step
    try:
        # Setup Jira API credentials
        os.environ["JIRA_API_TOKEN"] = st.secrets["JIRA_API_TOKEN"]
        os.environ["JIRA_USERNAME"] = "rich@bu.edu"
        os.environ["JIRA_INSTANCE_URL"] = "https://is883-genai-r.atlassian.net/"
        os.environ["JIRA_CLOUD"] = "True"

        # Extract user description from memory buffer
        if st.session_state.memory.buffer:
            user_description = st.session_state.memory.buffer[-1].content.strip()  # Latest user message
            #st.write(f"User description extracted: {user_description}")  # Debugging step
        else:
            st.error("Memory buffer is empty. Cannot extract user description.")
            raise ValueError("Memory buffer is empty.")

        # Define Jira task details
        product = st.session_state.identified_product
        subproduct = st.session_state.identified_subproduct
        issue = st.session_state.identified_issue

        # Create the assigned issue summary
        assigned_issue = f"Issue with {product} - {subproduct}: {issue}"
        #st.write(f"Assigned issue: {assigned_issue}")  # Debugging step

        # Define task creation question for the model
        question = (
            f"Create a task in my project with the key FST. The task's type is 'Task', assigned to rich@bu.edu. "
            f"The summary is '{assigned_issue}'. "
            f"Always assign 'Highest' priority if the issue is related to fraudulent activities. "
            f"Use 'High' priority for other issues. "
            f"The description is '{user_description}' which provides additional details."
        )
        st.write("Prepared Jira task details.")  # Debugging step

        # Initialize Jira toolkit and agent
        jira = JiraAPIWrapper()
        toolkit = JiraToolkit.from_jira_api_wrapper(jira)

        # Fix tool names and descriptions
        for idx, tool in enumerate(toolkit.tools):
            toolkit.tools[idx].name = toolkit.tools[idx].name.replace(" ", "_")
            if "create_issue" in toolkit.tools[idx].name:
                toolkit.tools[idx].description += " Ensure to specify the project ID."

        tools = toolkit.get_tools()
        chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model="gpt-4o-mini")
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(chat, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Invoke agent to create Jira task
        result = agent_executor.invoke({"input": question})
        #st.write(f"Agent execution result: {result}")  # Debugging step
        st.success(f"Jira task created successfully for the issue: {assigned_issue}")

        # Store Jira task details in session state
        st.session_state.jira_task_created = True
        st.session_state.jira_task_description = assigned_issue
        st.write("Task creation process completed successfully.")  # Debugging step

    except Exception as e:
        st.error(f"Error during Jira task creation: {e}")
        st.session_state.jira_task_created = False
