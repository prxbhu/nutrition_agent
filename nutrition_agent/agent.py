from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from dotenv import load_dotenv
import os
import asyncio
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

load_dotenv()

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

nutrition_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="/home/prxbhu/.nvm/versions/node/v22.17.0/bin/node",  # or your nvm exact node path
            args=[
                "/home/prxbhu/Documents/mcp-opennutrition/build/index.js"
            ],
            tool_filter=["search-food-by-name"] # or list of tool names if you want to filter
        ),
        timeout=30,
    )
)

root_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="root_agent",
    instruction="Use the MCP Tool to generate give nutritional information about the food for user queries",
    tools=[nutrition_server],
)