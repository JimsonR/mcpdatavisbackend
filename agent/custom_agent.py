import json
import re
import time
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import BaseTool

class StructuredAgent:
    def __init__(self, llm, tools: List[BaseTool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.tool_descriptions = {tool.name: tool.description for tool in tools}
        self.tool_arg_schemas = self._get_arg_schemas(tools)
    
    def _get_arg_schemas(self, tools: List[BaseTool]) -> Dict[str, Dict[str, Any]]:
        """Extracts argument schemas from tools without hardcoding"""
        schemas = {}
        for tool in tools:
            schema = {}
            if hasattr(tool, 'args_schema'):
                args_schema = tool.args_schema
                if hasattr(args_schema, '__annotations__'):
                    schema = args_schema.__annotations__
                elif hasattr(args_schema, 'schema'):
                    schema = args_schema.schema().get('properties', {})
            schemas[tool.name] = schema
        return schemas
    
    async def process_request(self, messages: List[BaseMessage], max_iterations=5) -> Dict[str, Any]:
        """Process request with flexible tool calling"""
        response_parts = []
        tool_executions = []
        current_context = messages[-1].content
        
        for iteration in range(max_iterations):
            decision = await self._make_decision(
                current_context,
                self._format_history(messages[:-1]),
                iteration
            )
            
            if "error" in decision:
                response_parts.append(decision["error"])
                break
                
            if decision.get("explanation"):
                response_parts.append(decision["explanation"])

            if decision.get("final_response"):
                response_parts.append(decision["final_response"])
                break

            if decision.get("tool_name"):
                tool_result = await self._execute_tool(
                    decision["tool_name"],
                    decision.get("tool_args", {}),
                    tool_executions
                )
                current_context = f"Tool result: {tool_result}\nOriginal request: {messages[-1].content}"

        return {
            "response": "\n\n".join(response_parts),
            "tool_executions": tool_executions
        }
    
    async def _make_decision(self, context: str, history: str, iteration: int) -> Dict[str, Any]:
        """Get LLM's decision on next steps"""
        prompt = self._create_decision_prompt(context, history, iteration)
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        try:
            return self._parse_response(response.content)
        except json.JSONDecodeError as e:
            return {
                "error": f"⚠️ Failed to parse response: {str(e)}",
                "original_content": response.content
            }
    
    def _create_decision_prompt(self, context: str, history: str, iteration: int) -> str:
        """Create prompt for structured decision making"""
        tools_info = []
        for name, desc in self.tool_descriptions.items():
            schema = self.tool_arg_schemas.get(name, {})
            if schema:
                args_desc = ", ".join([f"{k}: {v.__name__}" for k, v in schema.items()])
                tools_info.append(f"- {name}: {desc} (args: {args_desc})")
            else:
                tools_info.append(f"- {name}: {desc}")

        tools_info_str = "\n".join(tools_info)

        return f"""You are a structured assistant that uses tools precisely.

Current request: {context}

Conversation history: {history}

Available tools:
{tools_info_str}

On iteration {iteration + 1}, respond with JSON in this exact format:
{{
    "explanation": "Brief reasoning about your next step",
    "tool_name": "name_of_tool_or_'none'",
    "tool_args": {{"arg1": value, "arg2": value}},
    "final_response": "Only if you have a complete answer",
    "completed": boolean
}}

Guidelines:
- ALWAYS use the exact tool names shown above
- For tool arguments, use the exact names shown or what makes sense
- If a tool isn't needed, set "tool_name": "none"
- Include all required arguments - don't make up new ones
- Be precise with argument types (string, number, etc)"""
    

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON decision from response content"""
        json_str = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_str:
            return {
                "error": "⚠️ No valid JSON found in response",
                "original_content": content
            }
        
        try:
            decision = json.loads(json_str.group())
            # Validate minimum required fields
            if not isinstance(decision, dict):
                raise ValueError("Decision must be a JSON object")
            
            if "tool_name" not in decision:
                decision["tool_name"] = "none"
                
            return decision
        except (json.JSONDecodeError, ValueError) as e:
            return {
                "error": f"⚠️ Invalid decision format: {str(e)}",
                "original_content": content
            }

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any], executions: List[Dict]) -> Any:
        """Execute tool with flexible argument handling"""
        if tool_name not in self.tools:
            executions.append({
                "tool_name": tool_name,
                "error": f"Tool not found",
                "timestamp": time.time()
            })
            return f"Error: Tool '{tool_name}' not available"

        tool = self.tools[tool_name]
        args = self._validate_args(tool_name, args)
        
        try:
            result = await tool.ainvoke(args)
            executions.append({
                "tool_name": tool_name,
                "arguments": args,
                "result": str(result),
                "timestamp": time.time()
            })
            return result
        except Exception as e:
            error_msg = f"Tool error: {str(e)}"
            executions.append({
                "tool_name": tool_name,
                "arguments": args,
                "error": error_msg,
                "timestamp": time.time()
            })
            return error_msg
    
    def _validate_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate arguments against schema without hardcoding"""
        schema = self.tool_arg_schemas.get(tool_name, {})
        validated = {}
        
        for arg, value in args.items():
            # Convert types based on schema if available
            if arg in schema:
                expected_type = schema[arg]
                try:
                    if expected_type == str:
                        validated[arg] = str(value)
                    elif expected_type == int:
                        validated[arg] = int(value)
                    elif expected_type == float:
                        validated[arg] = float(value)
                    elif expected_type == bool:
                        validated[arg] = bool(value)
                    else:
                        validated[arg] = value
                except (ValueError, TypeError):
                    validated[arg] = value  # Fallback to original if conversion fails
            else:
                validated[arg] = value
                
        return validated

    def _format_history(self, messages: List[BaseMessage]) -> str:
        """Format conversation history"""
        if not messages:
            return "No previous messages"
        return "\n".join([
            f"{getattr(msg, 'type', 'unknown')}: {msg.content}"
            for msg in messages[-3:]  # Last 3 messages for context
        ])
