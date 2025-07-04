from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Weather")

@mcp.tool()
async def get_weather(location:str)->str:
    """Get the weather location."""
    return "It's always raining in California"

@mcp.tool()
async def my_name(location:str)->str:
    """Say my name"""
    return "Jimmy"

if __name__=="__main__":
    mcp.run(transport="streamable-http")