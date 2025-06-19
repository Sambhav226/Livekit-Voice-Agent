import os
import json
import time
import httpx
import traceback
import asyncio
import logging
from typing import Optional, List, Dict

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, ChatContext, AgentSession, function_tool, RunContext, RoomInputOptions

from livekit.plugins import (
    openai,
    sarvam,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from VectorSearch import VectorSearch 
# from json_embedder import retrieve_json  

load_dotenv()

# --------------------------
# Initialize Retrieval Service
# --------------------------
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "your-index-name")
vector_service = VectorSearch(index_name=PINECONE_INDEX)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

class SOPVoiceAgent(Agent):
    def __init__(self, chat_ctx: ChatContext) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions='''
            You are a multimodal AI agent that follows Standard Operating Procedures (SOPs) as behavioral guidelines.

            - When a user makes a request, determine the intent.
            - Retrieve relevant SOPs using the `sop_retriever_tool`.
            - Follow the SOP as if it were an execution plan.
            - Use your available tools (function calls or other agents) to carry out actions as described in the SOP.
            - Do not explain or recite the SOP to the user unless they specifically ask.
            - Your job is to act based on the SOP, not to describe it.
            '''
                    )

    @function_tool()
    async def sop_retriever_tool(
        self,
        context: RunContext,
        query: str,
        namespace: str,
        doc_category: str,
        topn: int = 5,
    ) -> dict[str, any]:
        """Retrieve relevant SOP instructions for execution, not for display."""
        try:
            docs = await vector_service.retrieval(
                query=query,
                namespace="test_namespace",
                doc_category="test",
                rerank_threshold=0.7,
                topn=topn
            )

            instructions = "\n\n".join([doc.text for doc in docs])
            clean = instructions.replace('*', '').replace('#', '').replace('- ', '')
            clean = ' '.join(clean.split())

            return {
                "sop_instructions": clean,
                "usage": "Use these SOP instructions to determine the next actions to take. Do not display to the user."
            }
        except Exception as e:
            return {"error": str(e)}


# --------------------------
#        Entrypoint
# --------------------------
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    try:
        metadata_str = ctx.job.metadata or "{}"
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError:
        metadata = {}

    user_name = metadata.get("user_name", "user")
    namespace = metadata.get("namespace", "default")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=sarvam.TTS(target_language_code="hi-IN", speaker="manisha", api_key=SARVAM_API_KEY),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="assistant", content=f"Hello {user_name}, I'm your AI assistant. Ask me anything about your SOP.")

    await session.start(
        room=ctx.room,
        agent=SOPVoiceAgent(chat_ctx=chat_ctx),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
