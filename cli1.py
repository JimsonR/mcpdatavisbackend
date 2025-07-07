

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import Client
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import asyncio
from langgraph.prebuilt import create_react_agent
import yaml
from threading import Lock


load_dotenv(".env")

# Path to the YAML config file
MCP_SERVERS_YAML = os.path.join(os.path.dirname(__file__), "mcp_servers.yaml")

# Thread-safe lock for config file access
_mcp_servers_lock = Lock()

def load_mcp_servers():
    with _mcp_servers_lock:
        if not os.path.exists(MCP_SERVERS_YAML):
            return {}
        with open(MCP_SERVERS_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Handle nested 'servers' key structure
        if 'servers' in data:
            return data['servers']
        return data

def save_mcp_servers(servers):
    with _mcp_servers_lock:
        with open(MCP_SERVERS_YAML, "w", encoding="utf-8") as f:
            # Maintain nested 'servers' structure
            yaml.safe_dump({"servers": servers}, f)


# In-memory cache of servers (reload on every change)
_cached_servers = None
_cached_servers_mtime = None
def get_mcp_servers():
    global _cached_servers, _cached_servers_mtime
    try:
        mtime = os.path.getmtime(MCP_SERVERS_YAML)
    except Exception:
        mtime = None
    if _cached_servers is None or mtime != _cached_servers_mtime:
        _cached_servers = load_mcp_servers()
        _cached_servers_mtime = mtime
    return _cached_servers

# For backward compatibility, fallback to hardcoded if YAML missing
def get_server_cfg(server):
    servers = get_mcp_servers()
    if server not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{server}' not found.")
    return servers[server]


app = FastAPI()

# Allow CORS for all origins (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health Check Endpoint ---
@app.get("/api/health")
def health_check():
    """Health check endpoint for backend availability."""
    return {"status": "ok"}


# LangChain Azure OpenAI LLM
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_type="azure",
)


from typing import List, Optional

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = None  # Each dict: {"role": "user"/"assistant", "content": "..."}

@app.post("/mcp/call-tool")
async def call_mcp_tool(server: str, tool_name: str, arguments: dict):
    server_cfg = get_server_cfg(server)
    async with Client(server_cfg["url"]) as client:
        result = await client.call_tool(tool_name, arguments)
        return result


# --- Optimized: Only check reachable servers on demand, not on every frontend load ---
@app.get("/mcp/list-tools")
async def list_mcp_tools(server: str):
    server_cfg = get_server_cfg(server)
    async with Client(server_cfg["url"]) as client:
        tools = await client.list_tools()
        return [t.model_dump() if hasattr(t, 'model_dump') else t for t in tools]


@app.get("/mcp/list-resources")
async def list_mcp_resources(server: str):
    server_cfg = get_server_cfg(server)
    async with Client(server_cfg["url"]) as client:
        resources = await client.list_resources()
        return resources

# --- New endpoint: Get resource content by URI ---
@app.get("/mcp/get-resource-content")
async def get_resource_content(server: str, uri: str):
    """Fetch and return the content of a resource by its URI."""
    server_cfg = get_server_cfg(server)
    async with Client(server_cfg["url"]) as client:
        content_list = await client.read_resource(uri)
        result = []
        for item in content_list:
            entry = {"mimeType": getattr(item, "mimeType", None)}
            if hasattr(item, "text") and item.text is not None:
                entry["type"] = "text"
                entry["content"] = item.text
            elif hasattr(item, "blob") and item.blob is not None:
                entry["type"] = "binary"
                entry["content"] = f"<binary: {len(item.blob)} bytes>"
            else:
                entry["type"] = "unknown"
                entry["content"] = None
            result.append(entry)
        return result

@app.get("/mcp/list-prompts")
async def list_mcp_prompts(server: str):
    server_cfg = get_server_cfg(server)
    async with Client(server_cfg["url"]) as client:
        prompts = await client.list_prompts()
        return prompts
@app.post("/mcp/get-prompt-content")
async def get_prompt_content(server: str, prompt_name: str, arguments: dict = {}):
    server_cfg = get_server_cfg(server)
    async with Client(server_cfg["url"]) as client:
        result = await client.get_prompt(prompt_name, arguments)
        # Return all messages as a list of dicts
        return [
            {"role": m.role, "content": getattr(m.content, "text", m.content)}
            for m in result.messages
        ]


from langchain_core.messages import AIMessage

@app.post("/llm/chat")
async def llm_chat(req: ChatRequest):
    # Build message list: history + new message
    messages = []
    if req.history:
        for m in req.history:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=req.message))
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: llm.invoke(messages))
    return {"response": response.content}


@app.post("/llm/agent")
async def llm_agent(req: ChatRequest):
    servers = get_mcp_servers()
    
    # Safety check: ensure we have servers configured
    if not servers:
        return {"response": "No MCP servers configured.", "error": True}
    
    try:
        # Filter out unreachable servers
        reachable_servers = {}
        for server_name, server_config in servers.items():
            try:
                # Quick health check for each server
                async with Client(server_config["url"]) as client:
                    await client.list_tools()  # Test if server is reachable
                    reachable_servers[server_name] = server_config
            except Exception as e:
                print(f"Warning: MCP server '{server_name}' at {server_config['url']} is unreachable: {e}")
                continue
        
        if not reachable_servers:
            return {"response": "No MCP servers are currently reachable. Please check if your MCP servers are running.", "error": True}
        
        print(f"Using {len(reachable_servers)} reachable servers: {list(reachable_servers.keys())}")
        
        client = MultiServerMCPClient(reachable_servers)
        tools = await client.get_tools()
        
        # Safety check: ensure we have tools
        if not tools:
            return {"response": "No tools available from reachable MCP servers.", "error": True}
        
        agent = create_react_agent(llm, tools)
        
        # Build message list: history + new message with context management
        messages = []
        if req.history:
            # Limit history to last 10 messages to prevent context overflow
            recent_history = req.history[-10:] if len(req.history) > 10 else req.history
            for m in recent_history:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    messages.append(AIMessage(content=m["content"]))
        messages.append(HumanMessage(content=req.message))
        
        # Add recursion limit and timeout
        result = await agent.ainvoke(
            {"messages": messages},
            config={
                "recursion_limit": 5,  # Further reduce to 5 to prevent context overflow
                "max_execution_time": 30  # Reduce timeout to 30 seconds
            }
        )
        return {"response": result['messages'][-1].content}
        
    except Exception as e:
        error_msg = str(e)
        print(f"Agent error: {error_msg}")
        if "recursion" in error_msg.lower():
            return {"response": "Agent hit recursion limit. The task may be too complex or tools are failing repeatedly.", "error": True}
        return {"response": f"Agent error: {error_msg}", "error": True}

    
@app.post("/llm/agent-detailed")
async def llm_agent_detailed(req: ChatRequest):
    servers = get_mcp_servers()
    
    # Safety check: ensure we have servers configured
    if not servers:
        return {
            "response": "No MCP servers configured.", 
            "error": True,
            "tool_executions": [],
            "full_conversation": []
        }
    
    try:
        # Filter out unreachable servers
        reachable_servers = {}
        for server_name, server_config in servers.items():
            try:
                # Quick health check for each server
                async with Client(server_config["url"]) as client:
                    await client.list_tools()  # Test if server is reachable
                    reachable_servers[server_name] = server_config
            except Exception as e:
                print(f"Warning: MCP server '{server_name}' at {server_config['url']} is unreachable: {e}")
                continue
        
        if not reachable_servers:
            return {
                "response": "No MCP servers are currently reachable. Please check if your MCP servers are running.", 
                "error": True,
                "tool_executions": [],
                "full_conversation": []
            }
        
        print(f"Using {len(reachable_servers)} reachable servers: {list(reachable_servers.keys())}")
        
        client = MultiServerMCPClient(reachable_servers)
        tools = await client.get_tools()
        
        # Safety check: ensure we have tools
        if not tools:
            return {
                "response": "No tools available from reachable MCP servers.", 
                "error": True,
                "tool_executions": [],
                "full_conversation": []
            }
        
        agent = create_react_agent(llm, tools)
        
        # Build message list: history + new message with context management
        messages = []
        if req.history:
            # Limit history to last 10 messages to prevent context overflow
            recent_history = req.history[-10:] if len(req.history) > 10 else req.history
            for m in recent_history:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    messages.append(AIMessage(content=m["content"]))
        messages.append(HumanMessage(content=req.message))
        
        # Add recursion limit and timeout
        result = await agent.ainvoke(
            {"messages": messages},
            config={
                "recursion_limit": 30,  # Further reduce to 5 to prevent context overflow
                "max_execution_time": 60  # Reduce timeout to 30 seconds
            }
        )
        
        # Extract tool execution details from the conversation
        tool_executions = []
        for message in result['messages']:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_executions.append({
                        "tool_name": tool_call.get("name", "unknown"),
                        "arguments": tool_call.get("args", {}),
                        "id": tool_call.get("id", "unknown")
                    })
            elif hasattr(message, 'type') and message.type == "tool":
                tool_executions.append({
                    "tool_response": getattr(message, 'content', 'No content'),
                    "tool_call_id": getattr(message, 'tool_call_id', 'unknown')
                })
        
        return {
            "response": result['messages'][-1].content,
            "tool_executions": tool_executions,
            "full_conversation": [
                {
                    "type": getattr(m, 'type', 'unknown'),
                    "content": getattr(m, 'content', str(m)),
                    "role": getattr(m, 'role', 'unknown') if hasattr(m, 'role') else None
                }
                for m in result['messages']
            ]
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"Agent detailed error: {error_msg}")
        if "recursion" in error_msg.lower():
            return {
                "response": "Agent hit recursion limit. The task may be too complex or tools are failing repeatedly.", 
                "error": True,
                "tool_executions": [],
                "full_conversation": []
            }
        return {
            "response": f"Agent error: {error_msg}", 
            "error": True,
            "tool_executions": [],
            "full_conversation": []
        }


# --- Optimized: Only check reachable servers when explicitly requested ---
@app.get("/langchain/list-tools")
async def langchain_list_tools(servers: str = None, only_reachable: bool = False):
    all_servers = get_mcp_servers()
    if servers:
        selected = {k: v for k, v in all_servers.items() if k in servers.split(",")}
    else:
        selected = all_servers
    if only_reachable:
        # Filter out unreachable servers
        reachable_servers = {}
        for server_name, server_config in selected.items():
            try:
                async with Client(server_config["url"]) as client:
                    await client.list_tools()  # Test if server is reachable
                    reachable_servers[server_name] = server_config
            except Exception as e:
                print(f"Warning: MCP server '{server_name}' at {server_config['url']} is unreachable: {e}")
                continue
        if not reachable_servers:
            return {"error": "No reachable MCP servers found"}
        client = MultiServerMCPClient(reachable_servers)
    else:
        # Do not check reachability, just return all
        client = MultiServerMCPClient(selected)
    tools = await client.get_tools()
    # Convert each tool to a dict (if needed)
    serializable_tools = []
    for tool in tools:
        if hasattr(tool, "dict"):
            serializable_tools.append(tool.dict())
        elif hasattr(tool, "model_dump"):
            serializable_tools.append(tool.model_dump())
        else:
            serializable_tools.append(tool)
    return serializable_tools

# --- MCP Server Config Management Endpoints ---
class MCPServerConfig(BaseModel):
    url: str
    transport: str

@app.get("/mcp/servers")
def list_mcp_servers():
    """List all MCP server names and configs."""
    return get_mcp_servers()

@app.post("/mcp/servers")
def add_mcp_server(name: str, config: MCPServerConfig):
    """Add a new MCP server."""
    servers = get_mcp_servers()
    if name in servers:
        raise HTTPException(status_code=400, detail=f"Server '{name}' already exists.")
    servers[name] = config.dict()
    save_mcp_servers(servers)
    return {"message": f"Server '{name}' added.", "servers": servers}

@app.put("/mcp/servers/{name}")
def update_mcp_server(name: str, config: MCPServerConfig):
    """Update an existing MCP server."""
    servers = get_mcp_servers()
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found.")
    servers[name] = config.dict()
    save_mcp_servers(servers)
    return {"message": f"Server '{name}' updated.", "servers": servers}

@app.delete("/mcp/servers/{name}")
def delete_mcp_server(name: str):
    """Delete an MCP server."""
    servers = get_mcp_servers()
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found.")
    del servers[name]
    save_mcp_servers(servers)
    return {"message": f"Server '{name}' deleted.", "servers": servers}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cli1:app", host="127.0.0.1", port=8080, reload=True)