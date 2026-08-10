import asyncio
import json
import os
import urllib.request
from pathlib import Path
from typing import Annotated

from agent_framework import Agent, Message
from agent_framework.observability import enable_instrumentation
from agent_framework.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from langfuse import Langfuse
from pydantic import Field
