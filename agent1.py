import os
import json
import time
import httpx
import traceback
import asyncio
import logging
import random
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, ChatContext, AgentSession, JobContext, WorkerOptions, cli, mcp, RoomInputOptions
from livekit.plugins import (
    openai,
    sarvam,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

# Setup logging
logger = logging.getLogger("sop-voice-agent")

# Get environment variables
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8040/sse")

class SOPVoiceAgent(Agent):
    def __init__(self, chat_ctx: ChatContext) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions='''
            You are an AI agent that follows Standard Operating Procedures (SOPs) and has access to employee database through MCP tools.
            
            WORKFLOW:
            1. For any user query, ALWAYS start by retrieving relevant SOPs using the MCP sop_retriever_tool
            2. Analyze the SOP response to determine if database access is required
            3. Only call database_analyzer_tool if the SOP indicates you need to:
               - Check employee records
               - Verify payroll/salary data
               - Analyze shift information
               - Review employee performance
               - Access any employee-related data
            4. Provide comprehensive responses based on SOP guidance and database findings
            
            Remember: SOP dictates when to access the database, not the user query directly.
            
            Always be conversational and helpful in your responses. You're speaking to users via voice, so keep responses natural and engaging.
            '''
        ),
        # self.chat_ctx = chat_ctx

    async def on_enter(self):
        logging.info("SOPVoiceAgent entering room")
        
        # Step 1: Copy the read-only context
        chat_ctx = self.chat_ctx.copy()
        
        # Step 2: Modify the copy
        chat_ctx.add_message(
            role="system",
            content="Welcome to the SOP assistant."
        )
        
        # Step 3: Update the session context
        await self.update_chat_ctx(chat_ctx)


    async def on_message(self, message):
        logging.info(f"User message received: {message}")
        self.chat_ctx.add_message("user", message)
        
        # Generate initial reply
        await self.session.generate_reply()


# --------------------------
# Entrypoint
# --------------------------
async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent"""
    await ctx.connect()
    logger.info("Connected to LiveKit room")
    
    # Parse metadata
    try:
        metadata_str = ctx.job.metadata or "{}"
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError:
        metadata = {}

    user_name = metadata.get("user_name", "user")
    logger.info(f"Starting session for user: {user_name}")

    # Create agent session with MCP servers
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=sarvam.TTS(
            target_language_code="hi-IN", 
            speaker="manisha", 
            api_key=SARVAM_API_KEY
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        mcp_servers=[
            # Your local MCP server for SOP and database tools
            mcp.MCPServerHTTP(
                url=MCP_SERVER_URL,
                timeout=10,
                client_session_timeout_seconds=30,
            ),
            # Add more MCP servers if needed
            # mcp.MCPServerHTTP(
            #     url="http://localhost:8001/sse",
            #     timeout=5,
            #     client_session_timeout_seconds=10,
            # ),
        ],
    )
    chat_ctx = ChatContext()
    chat_ctx.add_message(
        role="assistant", 
        content=f"Hello {user_name}! I'm your AI assistant with access to employee database..."
    )

    # Start the session
    await session.start(
        room=ctx.room,
        agent=SOPVoiceAgent(chat_ctx=chat_ctx),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting SOP Voice Agent with MCP support")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))