#!/usr/bin/env python3
"""
Test script for the StructuredAgent with MCP tools
"""

import asyncio
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from agent.custom_agent import StructuredAgent, make_langchain_tools_from_mcp_schema

# Example MCP tool schema (like what you'd get from an MCP server)
EXAMPLE_MCP_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Perform basic mathematical calculations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "args": {
                    "$ref": "#/$defs/CalculatorArgs"
                }
            },
            "required": ["args"],
            "$defs": {
                "CalculatorArgs": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        }
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "inputSchema": {
            "type": "object",
            "properties": {
                "args": {
                    "$ref": "#/$defs/WeatherArgs"
                }
            },
            "required": ["args"],
            "$defs": {
                "WeatherArgs": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
                    },
                    "required": ["location"]
                }
            }
        }
    }
]

# Mock backend functions (in real use, these would call your actual MCP server)
def create_backend_func(tool_name: str):
    """Create a backend function for a given tool"""
    def backend_func(args):
        print(f"[Backend] Calling {tool_name} with args: {args}")
        
        if tool_name == "calculator":
            # Extract args from MCP format
            calc_args = args.get("args", {})
            operation = calc_args.get("operation")
            a = calc_args.get("a")
            b = calc_args.get("b")
            
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return "Error: Division by zero"
                result = a / b
            else:
                return f"Unknown operation: {operation}"
            
            return f"Result: {result}"
        
        elif tool_name == "get_weather":
            # Extract args from MCP format
            weather_args = args.get("args", {})
            location = weather_args.get("location")
            units = weather_args.get("units", "celsius")
            
            # Mock weather data
            return f"Weather in {location}: 22°{'C' if units == 'celsius' else 'F'}, sunny"
        
        else:
            return f"Unknown tool: {tool_name}"
    
    return backend_func

async def test_agent():
    """Test the StructuredAgent with MCP tools"""
    
    # Initialize LLM (you'll need to set up your API key)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1
    )
    
    # Create tools from MCP schemas
    tools = make_langchain_tools_from_mcp_schema(EXAMPLE_MCP_SCHEMAS, create_backend_func)
    
    # Create agent
    agent = StructuredAgent(llm, tools)
    
    # Debug tool schemas
    print("Tool schemas:")
    for tool_name, info in agent.debug_tool_schemas().items():
        print(f"  {tool_name}: {info}")
    
    # Test queries
    test_queries = [
        "What is 15 + 27?",
        "What's the weather like in New York?",
        "Calculate 100 divided by 5 and then get the weather in Paris"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        try:
            result = await agent.process_request([HumanMessage(content=query)])
            
            print(f"Response: {result['response']}")
            print(f"Tool executions: {len(result['tool_executions'])}")
            
            for i, exec in enumerate(result['tool_executions']):
                print(f"  Tool {i+1}: {exec['tool_name']}")
                print(f"    Args: {exec['arguments']}")
                if 'result' in exec:
                    print(f"    Result: {exec['result']}")
                if 'error' in exec:
                    print(f"    Error: {exec['error']}")
                    
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_agent())
