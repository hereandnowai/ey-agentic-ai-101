import sys
from pathlib import Path
from agent_framework import MCPStdioTool
from _maf import POLICY, banner, get_client, run

banner("File 3 - MAF - This is an MCP client that calls MCP Server (another process)")

SERVER = str(Path(__file__).resolve().parent / "_mcp_server.py")

async def main():
    orders = MCPStdioTool(
        name="sheldon-orders",
        command=sys.executable,
        args=[SERVER],
    )
    async with orders:
        print("WHAT THE SERVER ADVERTISED (we did not write this list)")
        for tool in orders.functions:
            print(f" {tool.name}: {tool.description}")
        print()

        agent = get_client().as_agent(
            name="support",
            tools=[orders],
            instructions=f"You are Sheldon Retail Support. {POLICY} Use the tools.")
        result = await agent.run("Is order SC-90455 returnable?")
        print(f" agent answered: {result.text}")

run(main())