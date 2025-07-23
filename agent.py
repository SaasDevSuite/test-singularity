#!/usr/bin/env python3
"""
Simple LangChain Agent with Tools for AI Agent Platform.
This agent can use tools to answer questions and perform calculations.
"""

import os
import json
import logging
from datetime import datetime
import random
import re

from flask import Flask, request, jsonify
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is required!")
    exit(1)

# Custom Tools
@tool
def calculate(expression: str) -> str:
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

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"

@tool
def word_count(text: str) -> str:
    """Count the number of words and characters in a given text."""
    words = text.strip().split()
    chars = len(text)
    return f"The text contains {len(words)} words and {chars} characters"

@tool
def reverse_text(text: str) -> str:
    """Reverse the given text."""
    return f"Reversed text: {text[::-1]}"

@tool
def random_number(min_val: str, max_val: str) -> str:
    """Generate a random number between min and max (inclusive)."""
    try:
        min_num = int(min_val)
        max_num = int(max_val)
        result = random.randint(min_num, max_num)
        return f"Random number between {min_num} and {max_num}: {result}"
    except ValueError:
        return "Error: Please provide valid integers for min and max values"

@tool
def to_uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()

@tool
def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()

# Initialize tools
tools = [
    calculate,
    get_current_time,
    word_count,
    reverse_text,
    random_number,
    to_uppercase,
    to_lowercase,
]

# Initialize LLM
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY
)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,  # Set to True to see agent's reasoning
    handle_parsing_errors=True,
    max_iterations=5,
)

# API Routes
@app.route('/', methods=['GET'])
def home():
    """Home endpoint with agent information."""
    return jsonify({
        "name": "Simple LangChain Agent",
        "type": "conversational-react-agent",
        "tools": [tool.name for tool in tools],
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "tools": "/tools"
        }
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint for the agent."""
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400

        message = data['message']
        logger.info(f"Processing message: {message}")

        # Run the agent
        start_time = datetime.now()
        response = agent.run(message)
        end_time = datetime.now()

        response_time = (end_time - start_time).total_seconds()

        return jsonify({
            "response": response,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({
            "error": "An error occurred",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        # Test the agent with a simple query
        test_response = agent.run("What is 2+2?")

        return jsonify({
            "status": "healthy",
            "agent_responsive": bool(test_response),
            "tools_count": len(tools),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

@app.route('/tools', methods=['GET'])
def list_tools():
    """List available tools."""
    return jsonify({
        "tools": [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in tools
        ]
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    logger.info(f"Starting LangChain Agent on port {port}...")
    logger.info(f"Available tools: {[tool.name for tool in tools]}")

    app.run(host='0.0.0.0', port=port, debug=False)