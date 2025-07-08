# StructuredAgent for MCP Tools

This directory contains a robust, scalable custom agent for tool use in a LangChain-based FastAPI backend, specifically designed for MCP (Model Context Protocol) tools with complex schemas.

## Key Features

- **Dynamic Schema Handling**: Automatically adapts to MCP tool schemas provided at runtime
- **Argument Wrapping**: Properly handles MCP tools that require arguments wrapped in an "args" field
- **LangChain Native**: Uses LangGraph's StateGraph and ToolNode for robust agent orchestration
- **Error Handling**: Comprehensive error handling and debugging capabilities
- **Future-Proof**: No hardcoded argument structures - adapts to schema changes

## Architecture

### StructuredAgent Class

The main agent class that provides:
- LangGraph-based agent execution using StateGraph and ToolNode
- Automatic tool binding to the LLM using LangChain's APIs
- Structured response with tool execution details
- Built-in error handling and recovery

### Utility Functions

#### `make_langchain_tools_from_mcp_schema(tool_schemas, backend_func_factory)`

Dynamically creates LangChain tools from MCP tool schemas:
- Handles "args" wrapping for MCP tools
- Creates proper Pydantic models for validation
- Maps JSON schema types to Python types
- Returns list of LangChain-compatible tools

#### `create_structured_agent(llm, tool_schemas, backend_func_factory)`

Convenience function to create a complete agent from MCP schemas.

## Usage

### Basic Usage

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agent.custom_agent import StructuredAgent, make_langchain_tools_from_mcp_schema

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create tools from MCP schemas
tools = make_langchain_tools_from_mcp_schema(tool_schemas, backend_func_factory)

# Create agent
agent = StructuredAgent(llm, tools)

# Process request
result = await agent.process_request([HumanMessage(content="What's the weather?")])
print(result["response"])
```

### FastAPI Integration

The agent is integrated into the FastAPI backend via the `/llm/structured-agent` endpoint:

```python
@app.post("/llm/structured-agent")
async def llm_structured_agent(req: ChatRequest):
    # ... server discovery and tool schema extraction ...
    
    tools = make_langchain_tools_from_mcp_schema(tool_schemas, create_mcp_backend_func)
    agent = StructuredAgent(llm, tools)
    
    result = await agent.process_request(messages, max_iterations=8)
    return result
```

## MCP Tool Schema Format

The agent automatically handles MCP tool schemas like:

```json
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
                    "operation": {"type": "string", "enum": ["add", "subtract"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
    }
}
```

## Response Format

The agent returns structured responses:

```json
{
    "response": "The result is 42",
    "tool_executions": [
        {
            "tool_name": "calculator",
            "arguments": {"args": {"operation": "add", "a": 20, "b": 22}},
            "result": "42",
            "timestamp": 1234567890
        }
    ],
    "total_messages": 5,
    "success": true,
    "agent_type": "structured"
}
```

## Debugging

### Debug Tool Schemas

```python
debug_info = agent.debug_tool_schemas()
print(debug_info)
```

### Get Tool Usage Example

```python
example = agent.get_tool_usage_example("calculator")
print(example)
```

## Error Handling

The agent includes comprehensive error handling:
- Tool execution errors are captured and reported
- Model calling errors are handled gracefully  
- Schema validation errors are caught
- Recursive execution limits prevent infinite loops

## Testing

Run the test script to see the agent in action:

```bash
python test_agent.py
```

Make sure to set up your OpenAI API key in environment variables.

## Endpoints

- `/llm/structured-agent` - Main chat endpoint using StructuredAgent
- `/debug/tools` - Debug tool schemas and availability
- `/debug/tool-example/{tool_name}` - Get usage example for a tool

## Dependencies

- langchain-core
- langchain-openai  
- langgraph
- pydantic
- fastapi
- fastmcp

The agent is designed to be as maintainable and future-proof as LangChain's built-in ReAct agent while providing better handling of MCP tool schemas.
