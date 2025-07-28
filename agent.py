#!/usr/bin/env python3
"""
Simple Pydantic AI Agent with Tools for AI Agent Platform using FastAPI.
This agent can use tools to answer questions and perform calculations.
"""

import logging
import os
import random
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Pydantic AI imports
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is required!")
    exit(1)


# Pydantic models for structured data
class ChatHistory(BaseModel):
    messages: List[Dict[str, str]] = []


class AgentDeps(BaseModel):
    chat_history: ChatHistory = ChatHistory()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    response_time: float
    timestamp: str
    usage: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    agent_responsive: bool
    timestamp: str
    error: str = None


class ToolInfo(BaseModel):
    name: str
    description: str


class ToolsResponse(BaseModel):
    tools: List[ToolInfo]


class AgentInfo(BaseModel):
    name: str
    type: str
    tools: List[str]
    endpoints: Dict[str, str]


class MessageResponse(BaseModel):
    message: str
    timestamp: str


class HistoryResponse(BaseModel):
    history: List[Dict[str, str]]
    count: int
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="Simple Pydantic AI Agent",
    description="A helpful assistant with access to various tools using Pydantic AI",
    version="1.0.0"
)

# Initialize the Pydantic AI model
model = OpenAIModel('gpt-3.5-turbo')

# Initialize the agent with dependencies
agent = Agent(
    model,
    deps_type=AgentDeps,
    system_prompt=(
        "You are a helpful assistant with access to various tools. "
        "Use the appropriate tools to answer questions and perform tasks. "
        "Be conversational and helpful in your responses."
    ),
)


# Define tools using Pydantic AI decorators
@agent.tool
def calculate(ctx: RunContext[AgentDeps], expression: str) -> str:
    """Useful for when you need to perform mathematical calculations.
    Input should be a mathematical expression like '2 + 2' or '10 * 5'.
    Supports basic operations: +, -, *, /, **, %"""
    try:
        # Safe evaluation with limited operations
        allowed_chars = "0123456789+-*/()., "
        if all(c in allowed_chars for c in expression):
            result = eval(expression, {"__builtins__": {}}, {})
            return f"The result of {expression} is {result}"
        else:
            return "Invalid expression. Only numbers and basic operators (+, -, *, /, **, %) are allowed."
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@agent.tool
def get_current_time(ctx: RunContext[AgentDeps]) -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


@agent.tool
def word_count(ctx: RunContext[AgentDeps], text: str) -> str:
    """Count the number of words and characters in a given text."""
    words = text.strip().split()
    chars = len(text)
    return f"The text contains {len(words)} words and {chars} characters"


@agent.tool
def reverse_text(ctx: RunContext[AgentDeps], text: str) -> str:
    """Reverse the given text."""
    return f"Reversed text: {text[::-1]}"


@agent.tool
def random_number(ctx: RunContext[AgentDeps], min_val: str, max_val: str) -> str:
    """Generate a random number between min and max (inclusive)."""
    try:
        min_num = int(min_val)
        max_num = int(max_val)
        result = random.randint(min_num, max_num)
        return f"Random number between {min_num} and {max_num}: {result}"
    except ValueError:
        return "Error: Please provide valid integers for min and max values"


@agent.tool
def to_uppercase(ctx: RunContext[AgentDeps], text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


@agent.tool
def to_lowercase(ctx: RunContext[AgentDeps], text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


# Global dependencies instance for maintaining conversation history
global_deps = AgentDeps()


# API Routes
@app.get("/", response_model=AgentInfo)
async def home():
    """Home endpoint with agent information."""
    # Get tool names from the agent
    tool_names = []
    for tool_name in agent._function_toolset.tools:
        tool_names.append(tool_name)

    return AgentInfo(
        name="Simple Pydantic AI Agent",
        type="pydantic-ai-agent",
        tools=tool_names,
        endpoints={
            "chat": "/chat",
            "health": "/health",
            "tools": "/tools",
            "clear_history": "/clear_history"
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for the agent."""
    try:
        message = request.message
        logger.info(f"Processing message: {message}")

        # Add user message to history
        global_deps.chat_history.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })

        # Run the agent
        start_time = datetime.now()
        result = agent.run_sync(message, deps=global_deps)
        end_time = datetime.now()

        response = result.data
        response_time = (end_time - start_time).total_seconds()

        # Add assistant response to history
        global_deps.chat_history.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        # Keep history manageable (last 20 messages)
        if len(global_deps.chat_history.messages) > 20:
            global_deps.chat_history.messages = global_deps.chat_history.messages[-20:]

        return ChatResponse(
            response=response,
            response_time=response_time,
            timestamp=datetime.now().isoformat(),
            usage={
                "total_tokens": result.usage().total_tokens if result.usage() else 0,
                "prompt_tokens": result.usage().request_tokens if result.usage() else 0,
                "completion_tokens": result.usage().response_tokens if result.usage() else 0,
            }
        )

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "An error occurred",
                "details": str(e)
            }
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        # Test the agent with a simple query
        test_result = await agent.run("What is 2+2?", deps=AgentDeps())
        test_response = test_result.data

        return HealthResponse(
            status="healthy",
            agent_responsive=bool(test_response),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=HealthResponse(
                status="unhealthy",
                agent_responsive=False,
                error=str(e),
                timestamp=datetime.now().isoformat()
            ).dict()
        )


@app.get("/tools", response_model=ToolsResponse)
async def list_tools():
    """List available tools."""
    tools_info = []

    for tool_name, tool_func in agent._function_tools.items():
        tools_info.append(ToolInfo(
            name=tool_name,
            description=tool_func.__doc__ or "No description available"
        ))

    return ToolsResponse(tools=tools_info)


@app.post("/clear_history", response_model=MessageResponse)
async def clear_history():
    """Clear conversation history."""
    global global_deps
    global_deps.chat_history.messages.clear()

    return MessageResponse(
        message="Conversation history cleared",
        timestamp=datetime.now().isoformat()
    )


@app.get("/history", response_model=HistoryResponse)
async def get_history():
    """Get conversation history."""
    return HistoryResponse(
        history=global_deps.chat_history.messages,
        count=len(global_deps.chat_history.messages),
        timestamp=datetime.now().isoformat()
    )


if __name__ == '__main__':
    import uvicorn

    port = int(os.getenv('PORT', 8080))
    logger.info(f"Starting Pydantic AI Agent on port {port}...")
    logger.info(f"Available tools: {agent._function_toolset.tools.keys()}")

    uvicorn.run(app, host="0.0.0.0", port=port)