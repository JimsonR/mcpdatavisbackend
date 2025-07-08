# import json
# import re
# import time
# from typing import Dict, Any, List, Optional, Callable, Union, Sequence
# from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
# from langchain_core.tools import BaseTool, Tool
# from langchain_core.language_models import BaseChatModel
# from pydantic import create_model, BaseModel
# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode
# from typing_extensions import Annotated, TypedDict

# class AgentState(TypedDict):
#     """State for the custom agent"""
#     messages: Annotated[List[BaseMessage], add_messages]

# class StructuredAgent:
#     def __init__(self, llm: BaseChatModel, tools: List[BaseTool]):
#         self.llm = llm
#         self.tools = tools
#         self.tools_by_name = {tool.name: tool for tool in tools}
        
#         # Validate tools
#         self._validate_tools()
        
#         # Create LangGraph-based agent
#         self.graph = self._create_agent_graph()
    
#     def _validate_tools(self):
#         """Validate that all tools are properly configured"""
#         for tool in self.tools:
#             if not hasattr(tool, 'name') or not tool.name:
#                 raise ValueError(f"Tool missing name: {tool}")
#             if not hasattr(tool, 'description') or not tool.description:
#                 raise ValueError(f"Tool '{tool.name}' missing description")
#             # Accept either a sync func or an async coroutine
#             if not (hasattr(tool, 'func') and callable(tool.func)) and not (hasattr(tool, 'coroutine') and callable(tool.coroutine)):
#                 raise ValueError(f"Tool '{tool.name}' missing or invalid func/coroutine")
    
#     def _create_agent_graph(self):
#         """Create a LangGraph-based agent similar to create_react_agent but with custom error handling"""
        
#         # Create tool node with error handling
#         tool_node = ToolNode(
#             self.tools,
#             name="tools",
#             handle_tool_errors=True  # Let LangChain handle tool errors
#         )
        
#         # Create the graph
#         workflow = StateGraph(AgentState)
        
#         # Add nodes
#         workflow.add_node("agent", self._call_model)
#         workflow.add_node("tools", tool_node)
        
#         # Add edges
#         workflow.set_entry_point("agent")
#         workflow.add_conditional_edges(
#             "agent",
#             self._should_continue,
#             {
#                 "continue": "tools",
#                 "end": END,
#             }
#         )
#         workflow.add_edge("tools", "agent")
        
#         return workflow.compile()
    
#     async def _call_model(self, state: AgentState) -> Dict[str, Any]:
#         """Call the LLM with proper prompting"""
#         messages = state["messages"]
        
#         # Create system message with tool information
#         system_prompt = self._create_system_prompt()
        
#         # Bind tools to model (LangChain will handle the tool schema automatically)
#         model_with_tools = self.llm.bind_tools(self.tools)
        
#         # Add system prompt to messages if this is the first call
#         if not any(isinstance(msg, HumanMessage) and system_prompt in msg.content for msg in messages):
#             full_messages = [HumanMessage(content=system_prompt)] + messages
#         else:
#             full_messages = messages
        
#         try:
#             # Call model
#             response = await model_with_tools.ainvoke(full_messages)
#             return {"messages": [response]}
#         except Exception as e:
#             # Return error message
#             error_msg = AIMessage(content=f"Error calling model: {str(e)}")
#             return {"messages": [error_msg]}
    
#     def _create_system_prompt(self) -> str:
#         """Create system prompt with tool descriptions"""
#         tool_descriptions = []
#         for tool in self.tools:
#             tool_descriptions.append(f"- {tool.name}: {tool.description}")
        
#         return f"""You are a helpful assistant with access to tools. Use tools when needed to help answer questions.

# Available tools:
# {chr(10).join(tool_descriptions)}

# Think step by step and use tools as needed. When you get tool results, use them to provide a comprehensive answer."""
    
#     def _should_continue(self, state: AgentState) -> str:
#         """Determine whether to continue with tools or end"""
#         last_message = state["messages"][-1]
        
#         # If the last message has tool calls, continue to tools
#         if hasattr(last_message, "tool_calls") and last_message.tool_calls:
#             return "continue"
        
#         # Otherwise, we're done
#         return "end"
    
#     async def process_request(self, messages: List[BaseMessage], max_iterations: int = 10) -> Dict[str, Any]:
#         """Process a request using the LangGraph agent"""
#         try:
#             # Validate inputs
#             if not messages:
#                 return {
#                     "response": "No messages provided",
#                     "tool_executions": [],
#                     "error": True
#                 }
            
#             # Create initial state
#             initial_state = {"messages": messages}
            
#             # Run the graph with recursion limit
#             config = {"recursion_limit": max_iterations}
#             result = await self.graph.ainvoke(initial_state, config=config)
            
#             # Extract tool executions from the conversation
#             tool_executions = self._extract_tool_executions(result["messages"])
            
#             # Get final response
#             final_message = result["messages"][-1] if result["messages"] else None
#             if final_message and hasattr(final_message, 'content'):
#                 final_response = final_message.content
#             else:
#                 final_response = "No response generated"
            
#             # Check if the conversation ended with tool calls (incomplete)
#             if (final_message and hasattr(final_message, "tool_calls") and 
#                 final_message.tool_calls and final_response == ""):
#                 final_response = "Processing tools... (conversation may have been cut off due to iteration limit)"
            
#             return {
#                 "response": final_response,
#                 "tool_executions": tool_executions,
#                 "total_messages": len(result["messages"]),
#                 "success": True
#             }
            
#         except Exception as e:
#             return {
#                 "response": f"Agent error: {str(e)}",
#                 "tool_executions": [],
#                 "error": True,
#                 "error_type": type(e).__name__
#             }
    
#     def _extract_tool_executions(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
#         """Extract tool execution details from the message history"""
#         executions = []
        
#         for i, message in enumerate(messages):
#             if hasattr(message, "tool_calls") and message.tool_calls:
#                 # This is an AI message with tool calls
#                 for tool_call in message.tool_calls:
#                     execution = {
#                         "tool_name": tool_call["name"],
#                         "arguments": tool_call["args"],
#                         "tool_call_id": tool_call["id"],
#                         "timestamp": time.time()
#                     }
                    
#                     # Look for corresponding tool result
#                     for j in range(i + 1, len(messages)):
#                         if (isinstance(messages[j], ToolMessage) and 
#                             hasattr(messages[j], "tool_call_id") and
#                             messages[j].tool_call_id == tool_call["id"]):
                            
#                             if hasattr(messages[j], "additional_kwargs") and messages[j].additional_kwargs.get("is_error"):
#                                 execution["error"] = messages[j].content
#                             else:
#                                 execution["result"] = messages[j].content[:500] + "..." if len(messages[j].content) > 500 else messages[j].content
#                             break
                    
#                     executions.append(execution)
        
#         return executions
    
#     # Utility methods for debugging and compatibility
#     def debug_tool_schemas(self) -> Dict[str, Any]:
#         """Debug method to inspect tool schemas"""
#         debug_info = {}
#         for tool in self.tools:
#             debug_info[tool.name] = {
#                 'description': tool.description,
#                 'name': tool.name,
#                 'has_schema': hasattr(tool, 'args_schema') and tool.args_schema is not None
#             }
            
#             # Try to get basic schema info
#             if hasattr(tool, 'args_schema') and tool.args_schema:
#                 try:
#                     if hasattr(tool.args_schema, 'model_json_schema'):
#                         schema = tool.args_schema.model_json_schema()
#                         debug_info[tool.name]['properties'] = list(schema.get('properties', {}).keys())
#                         debug_info[tool.name]['required'] = schema.get('required', [])
#                     elif hasattr(tool.args_schema, 'schema'):
#                         schema = tool.args_schema.schema()
#                         debug_info[tool.name]['properties'] = list(schema.get('properties', {}).keys())
#                         debug_info[tool.name]['required'] = schema.get('required', [])
#                 except Exception as e:
#                     debug_info[tool.name]['schema_error'] = str(e)
        
#         return debug_info
    

# # --- Utility: ensure_async ---
# import asyncio
# import inspect

# def ensure_async(fn):
#     """Wrap a sync function as async if needed."""
#     if inspect.iscoroutinefunction(fn):
#         return fn
#     async def async_wrapper(*args, **kwargs):
#         return await asyncio.to_thread(fn, *args, **kwargs)
#     return async_wrapper

# def make_langchain_tools_from_mcp_schema(tool_schemas: list, backend_func_factory: Callable) -> List[BaseTool]:
#     """
#     Dynamically create LangChain tools from MCP tool schemas.
    
#     Args:
#         tool_schemas: List of tool schema dicts (as provided by your MCP server)
#         backend_func_factory: Callable that takes (tool_name: str) and returns a function to call the backend
    
#     Returns:
#         List of LangChain Tool objects that properly handle MCP "args" wrapping
#     """
#     tools = []
    
#     for tool_info in tool_schemas:
#         name = tool_info["name"]
#         description = tool_info.get("description", "")
#         input_schema = tool_info.get("inputSchema", {})
        
#         # Check if 'args' is required (MCP pattern)
#         if "args" in input_schema.get("required", []):
#             # Build inner schema for 'args'
#             defs = input_schema.get("$defs") or input_schema.get("definitions")
            
#             if defs:
#                 # Use the first $defs key (usually only one)
#                 args_def = defs[list(defs.keys())[0]]
#                 fields = {}
                
#                 for k, v in args_def.get("properties", {}).items():
#                     # Map JSON schema types to Python types
#                     typ = str  # default
#                     if v.get("type") == "integer":
#                         typ = int
#                     elif v.get("type") == "boolean":
#                         typ = bool
#                     elif v.get("type") == "number":
#                         typ = float
#                     elif v.get("type") == "array":
#                         typ = List[str]  # simplified array handling
                    
#                     required = k in args_def.get("required", [])
#                     default = ... if required else v.get("default", None)
#                     fields[k] = (typ, default)
                
#                 # Create Pydantic models for the tool schema
#                 ArgsModel = create_model(f"{name}_Args", **fields)
#                 InputModel = create_model(f"{name}_Input", args=(ArgsModel, ...))
                


#                 import inspect
#                 def make_tool_func(tool_name):
#                     backend_fn = backend_func_factory(tool_name)
#                     if inspect.iscoroutinefunction(backend_fn):
#                         async def tool_func(args):
#                             return await backend_fn({"args": args})
#                     else:
#                         async def tool_func(args):
#                             wrapped = ensure_async(backend_fn)
#                             return await wrapped({"args": args})
#                     return tool_func

#                 tools.append(Tool(
#                     name=name,
#                     description=description,
#                     args_schema=InputModel,
#                     func=None,
#                     coroutine=make_tool_func(name)
#                 ))
#             else:
#                 # Fallback: treat as generic dict with args wrapper


#                 import inspect
#                 def make_tool_func(tool_name):
#                     backend_fn = backend_func_factory(tool_name)
#                     if inspect.iscoroutinefunction(backend_fn):
#                         async def tool_func(**kwargs):
#                             return await backend_fn({"args": kwargs})
#                     else:
#                         async def tool_func(**kwargs):
#                             wrapped = ensure_async(backend_fn)
#                             return await wrapped({"args": kwargs})
#                     return tool_func

#                 tools.append(Tool(
#                     name=name,
#                     description=description,
#                     func=None,
#                     coroutine=make_tool_func(name)
#                 ))
#         else:
#             # No args wrapper needed - direct parameter passing

#             import inspect
#             def make_tool_func(tool_name):
#                 backend_fn = backend_func_factory(tool_name)
#                 if inspect.iscoroutinefunction(backend_fn):
#                     async def tool_func(**kwargs):
#                         return await backend_fn(kwargs)
#                 else:
#                     async def tool_func(**kwargs):
#                         wrapped = ensure_async(backend_fn)
#                         return await wrapped(kwargs)
#                 return tool_func

#             tools.append(Tool(
#                 name=name,
#                 description=description,
#                 func=None,
#                 coroutine=make_tool_func(name)
#             ))
    
#     return tools


# def create_structured_agent(llm: BaseChatModel, tool_schemas: list, backend_func_factory: Callable) -> StructuredAgent:
#     """
#     Convenience function to create a StructuredAgent with MCP tools.
    
#     Args:
#         llm: The language model to use
#         tool_schemas: List of MCP tool schemas
#         backend_func_factory: Function factory for tool execution
    
#     Returns:
#         Configured StructuredAgent ready for use
#     """
#     tools = make_langchain_tools_from_mcp_schema(tool_schemas, backend_func_factory)
#     return StructuredAgent(llm, tools)


import asyncio
from typing import List, Dict, Any, Optional, Union
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel

# class StructuredAgent:
#     def __init__(self, llm: BaseChatModel, tools: List[BaseTool], max_iterations: int = 30):
#         self.llm = llm
#         self.tools = {tool.name: tool for tool in tools} 
#         self.max_iterations = max_iterations

#     async def invoke(self, input: Union[str, List[BaseMessage]]) -> Dict[str, Any]:
#         """Run the agent with input (text or messages)"""
#         # Initialize state
#         if isinstance(input, str):
#             messages = [HumanMessage(content=input)]
#         else:
#             messages = input.copy()
        
#         iterations = 0
#         final_response = None
        
#         # Main loop
#         while iterations < self.max_iterations:
#             iterations += 1
            
#             # Generate AI response (with tool binding)
#             llm_with_tools = self.llm.bind_tools(list(self.tools.values()))
#             ai_message = await llm_with_tools.ainvoke(messages)
#             messages.append(ai_message)
            
#             # Extract tool calls if any
#             if not hasattr(ai_message, 'tool_calls') or not ai_message.tool_calls:
#                 final_response = ai_message.content
#                 break
                
#             # Execute all tool calls
#             tool_messages = await self._execute_tools(ai_message.tool_calls)
#             messages.extend(tool_messages)
        
#         return {
#             "response": final_response or f"Stopped after {iterations} iterations",
#             "messages": messages,
#             "iterations": iterations,
#             "tool_executions": self._extract_tool_executions(messages)
#         }

#     async def _execute_tools(self, tool_calls: List[Dict]) -> List[ToolMessage]:
#         """Execute multiple tool calls asynchronously"""
#         tool_messages = []
        
#         for tool_call in tool_calls:
#             tool_name = tool_call['name']
#             tool_args = tool_call['args']
            
#             if tool_name not in self.tools:
#                 result = f"Error: Unknown tool '{tool_name}'"
#             else:
#                 tool = self.tools[tool_name]
#                 try:
#                     if asyncio.iscoroutinefunction(tool.arun):
#                         result = await tool.arun(tool_args)
#                     elif asyncio.iscoroutinefunction(tool.run):
#                         result = await tool.run(tool_args)
#                     else:
#                         result = await asyncio.to_thread(tool.run, tool_args)
#                 except Exception as e:
#                     result = f"Error executing {tool_name}: {str(e)}"
            
#             tool_messages.append(
#                 ToolMessage(
#                     content=str(result),
#                     tool_call_id=tool_call['id'],
#                     name=tool_name
#                 )
#             )
        
#         return tool_messages

#     def _extract_tool_executions(self, messages: List[BaseMessage]) -> List[Dict]:
#         """Extract tool execution details from message history"""
#         executions = []
        
#         for msg in messages:
#             if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls'):
#                 for tool_call in msg.tool_calls:
#                     execution = {
#                         'tool_name': tool_call['name'],
#                         'arguments': tool_call['args'],
#                         'tool_call_id': tool_call['id']
#                     }
                    
#                     # Find corresponding tool result
#                     for tool_msg in messages:
#                         if (isinstance(tool_msg, ToolMessage) and 
#                             tool_msg.tool_call_id == tool_call['id']):
#                             execution['result'] = (
#                                 tool_msg.content[:500] + '...' 
#                                 if len(tool_msg.content) > 500 
#                                 else tool_msg.content
#                             )
#                             break
                    
#                     executions.append(execution)
        
#         return executions
    

class StructuredAgent:
    def __init__(self, llm: BaseChatModel, tools: List[BaseTool], max_iterations: int = 30):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools} 
        self.max_iterations = max_iterations

    async def invoke(self, input: Union[str, List[BaseMessage]]) -> Dict[str, Any]:
        """Run the agent with input (text or messages)"""
        # Initialize state
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input.copy()
        
        iterations = 0
        final_response = None
        reasoning_steps = []
        
        # Main loop
        while iterations < self.max_iterations:
            iterations += 1
            
            # Generate AI response (with tool binding)
            llm_with_tools = self.llm.bind_tools(list(self.tools.values()))
            ai_message = await llm_with_tools.ainvoke(messages)
            messages.append(ai_message)
            
            # Process the AI response with reasoning
            step_info = self._process_ai_response(ai_message, iterations)
            reasoning_steps.append(step_info)
            
            # Extract tool calls if any
            if not hasattr(ai_message, 'tool_calls') or not ai_message.tool_calls:
                final_response = ai_message.content
                break
                
            # Execute all tool calls and update the step info
            tool_messages = await self._execute_tools(ai_message.tool_calls)
            messages.extend(tool_messages)
            
            # Update step with tool results
            step_info['tool_results'] = self._format_tool_results(ai_message.tool_calls, tool_messages)
        
        return {
            "response": final_response or f"Stopped after {iterations} iterations",
            "reasoning_steps": reasoning_steps,
            "messages": messages,
            "iterations": iterations,
            "formatted_output": self._format_structured_response(reasoning_steps, final_response)
        }

    def _process_ai_response(self, ai_message: AIMessage, iteration: int) -> Dict[str, Any]:
        """Process AI response and extract reasoning + tool calls"""
        step_info = {
            'iteration': iteration,
            'reasoning': ai_message.content,
            'tool_calls': [],
            'tool_results': []
        }
        
        if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
            for tool_call in ai_message.tool_calls:
                step_info['tool_calls'].append({
                    'name': tool_call['name'],
                    'args': tool_call['args'],
                    'id': tool_call['id']
                })
        
        return step_info

    async def _execute_tools(self, tool_calls: List[Dict]) -> List[ToolMessage]:
        """Execute multiple tool calls asynchronously"""
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            if tool_name not in self.tools:
                result = f"Error: Unknown tool '{tool_name}'"
            else:
                tool = self.tools[tool_name]
                try:
                    if asyncio.iscoroutinefunction(tool.arun):
                        result = await tool.arun(tool_args)
                    elif asyncio.iscoroutinefunction(tool.run):
                        result = await tool.run(tool_args)
                    else:
                        result = await asyncio.to_thread(tool.run, tool_args)
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"
            
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call['id'],
                    name=tool_name
                )
            )
        
        return tool_messages

    def _format_tool_results(self, tool_calls: List[Dict], tool_messages: List[ToolMessage]) -> List[Dict]:
        """Format tool results with their corresponding calls"""
        results = []
        
        for tool_call in tool_calls:
            # Find corresponding result
            result_content = None
            for tool_msg in tool_messages:
                if tool_msg.tool_call_id == tool_call['id']:
                    result_content = tool_msg.content
                    break
            
            results.append({
                'tool_name': tool_call['name'],
                'arguments': tool_call['args'],
                'result': result_content
            })
        
        return results

    def _format_structured_response(self, reasoning_steps: List[Dict], final_response: str) -> str:
        """Format the response in a structured, Claude-like manner"""
        formatted_parts = []
        
        for step in reasoning_steps:
            if step['reasoning']:
                formatted_parts.append(step['reasoning'])
            
            if step['tool_calls']:
                formatted_parts.append("\n")
                for i, tool_result in enumerate(step['tool_results']):
                    tool_name = tool_result['tool_name']
                    args = tool_result['arguments']
                    result = tool_result['result']
                    
                    # Format like Claude's tool calls
                    formatted_parts.append(f"<tool_call>")
                    formatted_parts.append(f"<tool_name>{tool_name}</tool_name>")
                    
                    # Format arguments
                    for key, value in args.items():
                        formatted_parts.append(f"<{key}>{value}</{key}>")
                    
                    formatted_parts.append(f"</tool_call>")
                    
                    # Add result if available
                    if result:
                        formatted_parts.append(f"\n{result}\n")
        
        if final_response:
            formatted_parts.append(f"\n{final_response}")
        
        return '\n'.join(formatted_parts)

    def _extract_tool_executions(self, messages: List[BaseMessage]) -> List[Dict]:
        """Extract tool execution details from message history (legacy method)"""
        executions = []
        
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls'):
                for tool_call in msg.tool_calls:
                    execution = {
                        'tool_name': tool_call['name'],
                        'arguments': tool_call['args'],
                        'tool_call_id': tool_call['id']
                    }
                    
                    # Find corresponding tool result
                    for tool_msg in messages:
                        if (isinstance(tool_msg, ToolMessage) and 
                            tool_msg.tool_call_id == tool_call['id']):
                            execution['result'] = (
                                tool_msg.content[:500] + '...' 
                                if len(tool_msg.content) > 500 
                                else tool_msg.content
                            )
                            break
                    
                    executions.append(execution)
        
        return executions

    def get_conversation_summary(self) -> str:
        """Get a summary of the last conversation in a readable format"""
        # This would be called after invoke() to get a nice summary
        pass