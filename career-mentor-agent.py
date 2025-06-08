# Import Dependencies
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage, TextMessage, ToolCallRequestEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
import json
import requests
import streamlit as st
import sys
from typing import AsyncGenerator, Sequence

# Create the Streamlit Interface

st.header("Autogen 0.4 Agent with Streamlit UI")


######################################################################################

# Define the custom agent class

class TrackableAssistantAgent(AssistantAgent):
    """
    A specialized assistant agent that tracks and displays responses on Streamlit 
    while processing messages. This class extends the functionality of the 
    `AssistantAgent` by adding methods to handle and visualize responses, including 
    tool call requests, text messages, and responses containing image references.
    Methods:
        on_messages_stream(messages, cancellation_token):
            Asynchronously processes a stream of messages, tracks responses on 
            Streamlit, and yields each message (modified or unmodified).
        _track_response_on_streamlit(msg):
            Tracks and displays the response on Streamlit based on the type of 
            message (e.g., tool call requests, text messages, or responses).
        _handle_text_message(msg):
            Handles text messages by formatting the content, appending it to 
            Streamlit's session state, and displaying it in the chat interface. 
    """
    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
         async for msg in super().on_messages_stream(
            messages=messages,
            cancellation_token=cancellation_token,
         ):
            self._track_response_on_streamlit(msg)
            
            # Yield the item (modified or unmodified)
            yield msg

    def _track_response_on_streamlit(self, msg):
        if isinstance(msg, ToolCallRequestEvent):
            content = f"**[{msg.source}] Tool calls requested:**\n- " + "\n- ".join(f"`{tool.name}` with arguments `{tool.arguments})`" for tool in msg.content)
            st.session_state["messages"].append(
                {"role": "assistant", "content": content}
            )
            with st.chat_message("assistant", avatar="🛠️"):
                st.markdown(content)
        
        elif (isinstance(msg, TextMessage)) and msg.source != "user":
            self._handle_text_message(msg)

        elif isinstance(msg, Response):
            if isinstance(msg.chat_message, TextMessage):
                self._handle_text_message(msg.chat_message)
        else:
            pass

    def _handle_text_message(self, msg: TextMessage) -> None:
        msg_content = f"**[{msg.source}]**\n{msg.content.replace('TERMINATE', '').strip()}"
        st.session_state["messages"].append(
            {"role": "assistant", "content": msg_content}
        )
        with st.chat_message("assistant"):
            st.markdown(msg_content)

######################################################################################

# Configure the GitHub PAT & Model for the Agent

async def reset_chat():
    """Clear the chat history."""
    if "messages" in st.session_state:
        st.session_state["messages"] = []
    if "agent" in st.session_state:
        await st.session_state["agent"].reset()

gh_pat = None
model = None

with st.sidebar:
    st.header("Configuration")
    gh_pat = st.text_input(
        "GitHub Personal Access Token (PAT)",
        placeholder="Enter your GitHub PAT here",
        type="password"
    )
    model = st.selectbox(
        "Select a model",
        options=["gpt-4o-mini", "gpt-4o", "o1-mini" "o1"],
        index=0
    )

    st.text_input(
        "Serper API Key",
        placeholder="Enter your Serper API key here",
        type="password",
        key="serper_api_key"
    )
    st.info("The SERPER API key is needed for the agent to perform web searches. You can get it from [Serper](https://serper.dev/).")

    st.button(
        "New Chat",
        on_click=lambda: asyncio.run(reset_chat()),
        type="secondary",
        use_container_width=True
    )

######################################################################################

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Solution for Windows users
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Ensure the event loop is created only once
if "event_loop" not in st.session_state:
    st.session_state["event_loop"] = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state["event_loop"])

######################################################################################

# Tools Accessible to the agent
def serper_web_search(query: str) -> str:
    """
    Perform a web search using the Serper API and return the results.

    Args:
        query (str): The search query.

    Returns:
        str: The search results in JSON format.
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "gl": "in"
    })
    headers = {
    'X-API-KEY': st.session_state["serper_api_key"],
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"
    return response.text

######################################################################################

# Initialize the agent with the provided GitHub PAT and model
def initialize_agent(gh_pat: str, model: str):
    model_client = AzureAIChatCompletionClient(
        model=model,
        endpoint="https://models.inference.ai.azure.com",
        # To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
        credential=AzureKeyCredential(gh_pat),
        model_info={
            "json_output": True,
            "function_calling": True,
            "vision": True,
            "family": "unknown",
        },
    )

    career_mentor_agent = TrackableAssistantAgent(
        name="CareerMentorAgent",
        description="An agent that provides career advice and mentorship.",
        model_client=model_client,
        tools=[serper_web_search] if st.session_state.get("serper_api_key") else [],
        system_message="You are a Career Mentor Agent with deep expertise in career development, professional growth, and industry trends. Your goal is to provide thoughtful, strategic, and actionable advice to help users navigate career challenges, make informed decisions, and achieve long-term success. Use the tools at your disposal whenever required. Offer clear, empathetic guidance based on your knowledge, considering the user's background and goals. If the question is outside the domain of career development, politely redirect the user to a more appropriate topic. You must end with the word 'TERMINATE' only at the end of your final response."
    )

     # Termination condition that stops the task if the agent responds with a message containing "TERMINATE"
    termination_condition = TextMentionTermination("TERMINATE")

    return RoundRobinGroupChat([career_mentor_agent], termination_condition)


######################################################################################

# Initiate the Chat if GitHub PAT and model are provided
if not (gh_pat and model):
    st.warning("Please enter your GitHub PAT and select a model to continue.")
else:
    if "agent" not in st.session_state:
        # Initialize the agent with the provided GitHub PAT and model
        st.session_state["agent"] = initialize_agent(gh_pat, model)

    # Display chat history messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_query = st.chat_input("How can I help you with your?")

    if user_query:
        if user_query.strip() == "":
            st.warning("Please enter a query.")
        else:
            st.session_state["messages"].append({"role": "user", "content": user_query})
            
            with st.chat_message("user"):
                st.markdown(user_query)

            # Define an asynchronous function: this is needed to use await
            async def initiate_chat():
                await st.session_state["agent"].run(
                    task=[TextMessage(content=user_query, source="user")],
                    cancellation_token=CancellationToken(),
                )
                st.stop()  # Stop code execution after termination command

            # Run the asynchronous function within the event loop
            st.session_state["event_loop"].run_until_complete(initiate_chat())