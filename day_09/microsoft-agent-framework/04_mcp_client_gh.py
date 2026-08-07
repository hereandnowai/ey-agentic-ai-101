import os
import gradio as gr
from httpx import AsyncClient
from agent_framework import FunctionInvocationContext, MCPStreamableHTTPTool, Message
from _maf import MODEL, banner, get_client

banner("File 4 - MAF - This is an MCP client that calls GitHub MCP Server (another process) over HTTP")

URL = "https://api.githubcopilot.com/mcp/"
TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
if not TOKEN:
    raise SystemExit("Please set the GITHUB_PERSONAL_ACCESS_TOKEN environment variable.")

READ = []
WRITE = []
ALLOWED = READ + WRITE
INSTRUCTIONS = ("")