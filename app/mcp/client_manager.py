"""
MCP Client Manager for Research Paper Assistant.

Manages connections to external MCP servers (GitHub, Gmail, Slack, Apify)
via Stdio transport. Converts their tools into schemas compatible with LiteLLM/OpenAI,
and routes executed tool calls to the correct subprocess.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("uvicorn")

class MCPClientManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MCPClientManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        # Store user-specific contexts keying by clerk_id
        # Schema: { clerk_id: { "exit_stack": AsyncExitStack, "sessions": {}, "tool_mappings": {}, "initialized": bool, "github_username": str } }
        self.user_contexts: Dict[str, Dict[str, Any]] = {}
        # Backwards compatibility fields
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.tool_mappings: Dict[str, tuple] = {}
        self.github_username = None
        self.initialized = True

    def get_user_context(self, clerk_id: Optional[str] = None) -> Dict[str, Any]:
        """Resolves or initializes the context map for a given user clerk_id."""
        cid = clerk_id or "__global__"
        if cid not in self.user_contexts:
            self.user_contexts[cid] = {
                "exit_stack": AsyncExitStack(),
                "sessions": {},
                "tool_mappings": {},
                "initialized": False,
                "github_username": None
            }
        return self.user_contexts[cid]

    async def ensure_user_initialized(self, clerk_id: Optional[str], user_doc: Optional[dict] = None):
        """Ensures that the MCP child processes are spawned and connected for a specific user."""
        ctx = self.get_user_context(clerk_id)
        if ctx["initialized"]:
            return

        cid_label = clerk_id if clerk_id else "global"
        logger.info(f"🔌 Initializing MCP Client Connectors for user {cid_label}...")

        # Resolve credentials. Fallback to global settings if user_doc/connector_doc is not provided
        from app.core.config import settings
        rag_config = settings.load_rag_config()

        connector_doc = None
        if clerk_id:
            try:
                from app.db.mongodb import get_connector_collection
                conn_coll = get_connector_collection()
                connector_doc = await conn_coll.find_one({"clerk_id": clerk_id})
            except Exception as db_err:
                logger.error(f"⚠️ Failed to query connectors collection for user {clerk_id}: {db_err}")

        # Fallback to passed user_doc if query returned nothing
        if not connector_doc and user_doc:
            connector_doc = user_doc

        if connector_doc:
            github_token = connector_doc.get("github_token") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
            slack_token = connector_doc.get("slack_token") or os.getenv("SLACK_BOT_TOKEN")
            slack_team_id = connector_doc.get("slack_team_id") or os.getenv("SLACK_TEAM_ID") or "T00000000"
            gmail_id = connector_doc.get("gmail_client_id") or os.getenv("GMAIL_CLIENT_ID")
            gmail_secret = connector_doc.get("gmail_client_secret") or os.getenv("GMAIL_CLIENT_SECRET")
            gmail_refresh = connector_doc.get("gmail_refresh_token") or os.getenv("GMAIL_REFRESH_TOKEN")
            apify_token = connector_doc.get("apify_token") or os.getenv("APIFY_TOKEN")
        else:
            github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN") or rag_config.get("github_token")
            slack_token = os.getenv("SLACK_BOT_TOKEN") or rag_config.get("slack_token")
            slack_team_id = os.getenv("SLACK_TEAM_ID") or rag_config.get("slack_team_id") or "T00000000"
            gmail_id = os.getenv("GMAIL_CLIENT_ID") or rag_config.get("gmail_client_id")
            gmail_secret = os.getenv("GMAIL_CLIENT_SECRET") or os.getenv("GMAIL_CLIENT_SECRET") or rag_config.get("gmail_client_secret")
            gmail_refresh = os.getenv("GMAIL_REFRESH_TOKEN") or os.getenv("GMAIL_REFRESH_TOKEN") or rag_config.get("gmail_refresh_token")
            apify_token = os.getenv("APIFY_TOKEN") or rag_config.get("apify_token")

        configs = {}

        # 1. GitHub Configuration
        if github_token:
            configs["github"] = {
                "package": "@modelcontextprotocol/server-github",
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": github_token
                }
            }
            # Fetch GitHub username dynamically so LLM prompts can refer to it
            import httpx
            try:
                headers = {
                    "Authorization": f"Bearer {github_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
                async with httpx.AsyncClient() as client:
                    response = await client.get("https://api.github.com/user", headers=headers, timeout=5.0)
                    if response.status_code == 200:
                        ctx["github_username"] = response.json().get("login")
                        # For backwards compatibility if global
                        if not clerk_id:
                            self.github_username = ctx["github_username"]
                        logger.info(f"✨ Authenticated GitHub user for {cid_label}: {ctx['github_username']}")
                    else:
                        logger.warning(f"⚠️ Failed to fetch GitHub user info for {cid_label}: {response.status_code} {response.text}")
            except Exception as e:
                logger.error(f"⚠️ Error fetching GitHub user info for {cid_label}: {e}")
        else:
            ctx["github_username"] = None
            if not clerk_id:
                self.github_username = None
            logger.info(f"ℹ️ GitHub MCP connector disabled for {cid_label} (GITHUB_PERSONAL_ACCESS_TOKEN missing)")

        # 2. Slack Configuration
        if slack_token:
            configs["slack"] = {
                "package": "@modelcontextprotocol/server-slack",
                "env": {
                    "SLACK_BOT_TOKEN": slack_token,
                    "SLACK_TEAM_ID": slack_team_id
                }
            }
        else:
            logger.info(f"ℹ️ Slack MCP connector disabled for {cid_label} (SLACK_BOT_TOKEN missing)")

        # 3. Gmail Configuration
        if gmail_id and gmail_secret and gmail_refresh:
            configs["gmail"] = {
                "package": "@modelcontextprotocol/server-gmail",
                "env": {
                    "GMAIL_CLIENT_ID": gmail_id,
                    "GMAIL_CLIENT_SECRET": gmail_secret,
                    "GMAIL_REFRESH_TOKEN": gmail_refresh
                }
            }
        else:
            logger.info(f"ℹ️ Gmail MCP connector disabled for {cid_label} (GMAIL OAuth credentials missing)")

        # 4. Apify Configuration
        if apify_token:
            configs["apify"] = {
                "package": "@apify/actors-mcp-server",
                "env": {
                    "APIFY_TOKEN": apify_token
                }
            }
        else:
            logger.info(f"ℹ️ Apify MCP connector disabled for {cid_label} (APIFY_TOKEN missing)")

        # Initialize transports
        exit_stack = ctx["exit_stack"]
        sessions_dict = ctx["sessions"]
        tool_mappings_dict = ctx["tool_mappings"]

        for name, config in configs.items():
            package = config["package"]
            env = {**os.environ, **config["env"]}
            label = f"[{cid_label}:{name}]"

            # Windows-compatible command launching
            if sys.platform == "win32":
                command = "cmd.exe"
                args = ["/c", "npx", "-y", package]
            else:
                command = "npx"
                args = ["-y", package]

            params = StdioServerParameters(command=command, args=args, env=env)

            try:
                logger.info(f"🚀 Starting MCP server subprocess for {label}...")
                transport = await exit_stack.enter_async_context(stdio_client(params))
                session = await exit_stack.enter_async_context(ClientSession(transport[0], transport[1]))
                await session.initialize()
                
                sessions_dict[name] = session
                # If global, mirror backwards compatibility
                if not clerk_id:
                    self.sessions[name] = session
                logger.info(f"✅ Connected {label} to MCP server: {name}")

                # Retrieve and register tools
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    tool_mappings_dict[tool.name] = (name, tool)
                    if not clerk_id:
                        self.tool_mappings[tool.name] = (name, tool)
                    logger.info(f"   Registered {label} tool: {tool.name} (from {name})")

            except Exception as e:
                logger.error(f"❌ Failed to connect to {label} MCP server {name}: {e}", exc_info=True)

        ctx["initialized"] = True

    async def get_all_tools(self, clerk_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all registered MCP tools in OpenAI/LiteLLM tool schema format."""
        openai_tools = []
        ctx = self.get_user_context(clerk_id)
        for tool_name, (server_name, tool) in ctx["tool_mappings"].items():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema
                }
            })
        return openai_tools

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], clerk_id: Optional[str] = None) -> str:
        """Executes a tool on the matching MCP server and returns the result string."""
        ctx = self.get_user_context(clerk_id)
        if tool_name not in ctx["tool_mappings"]:
            raise ValueError(f"Tool {tool_name} not found in registered MCP tools")

        server_name, _ = ctx["tool_mappings"][tool_name]
        session = ctx["sessions"].get(server_name)

        if not session:
            raise RuntimeError(f"Session for server {server_name} is not active")

        try:
            logger.info(f"🔌 Executing MCP tool: {tool_name} on {server_name} with args: {arguments}")
            result = await session.call_tool(tool_name, arguments)
            
            # Format results as string
            content_parts = []
            for item in result.content:
                if hasattr(item, "text") and item.text:
                    content_parts.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    content_parts.append(item["text"])
                else:
                    content_parts.append(str(item))
            
            return "\n".join(content_parts)
        except Exception as e:
            logger.error(f"❌ Failed executing tool {tool_name} on {server_name}: {e}")
            return f"Error executing tool {tool_name}: {str(e)}"

    async def initialize(self):
        """Standard global initialization for fallback/backwards compatibility."""
        await self.ensure_user_initialized(clerk_id=None)

    async def shutdown(self):
        """Cleans up all subprocesses and exits cleanly."""
        logger.info("🔌 Shutting down all MCP Client Connectors...")
        await self.exit_stack.aclose()
        self.exit_stack = AsyncExitStack()
        self.sessions.clear()
        self.tool_mappings.clear()
        self.github_username = None
        
        # Shutdown user specific contexts
        for clerk_id in list(self.user_contexts.keys()):
            await self.shutdown_for_user(clerk_id)
            
        logger.info("🔒 All MCP Client subprocesses terminated.")

    async def shutdown_for_user(self, clerk_id: str):
        if clerk_id in self.user_contexts:
            ctx = self.user_contexts[clerk_id]
            logger.info(f"🔌 Shutting down MCP Client Connectors for user {clerk_id}...")
            try:
                await ctx["exit_stack"].aclose()
            except Exception as e:
                logger.error(f"Error closing exit stack for user {clerk_id}: {e}")
            ctx["sessions"].clear()
            ctx["tool_mappings"].clear()
            ctx["github_username"] = None
            ctx["initialized"] = False

    async def reload(self):
        """Re-initializes all MCP connectors safely."""
        await self.shutdown()
        await self.initialize()

    async def reload_for_user(self, clerk_id: str, user_doc: dict):
        """Re-initializes MCP connectors for a specific user."""
        await self.shutdown_for_user(clerk_id)
        await self.ensure_user_initialized(clerk_id, user_doc)

# Global instance
mcp_client_manager = MCPClientManager()
