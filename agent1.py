import os
import json
import logging
import time
from loguru import logger
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, ChatContext, AgentSession, function_tool, RunContext, RoomInputOptions
from livekit.plugins import (
    openai,
    deepgram,
    noise_cancellation,
    silero,
    elevenlabs,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from VectorSearch import VectorSearch
from DynamicUserDatabase import DynamicUserDatabase

load_dotenv()

# --------------------------
# Initialize Retrieval Service
# --------------------------
PINECONE_HOST = os.getenv("PINECONE_HOST", "your-pinecone-host")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "your-index-name")
vector_service = VectorSearch(index_name=PINECONE_INDEX)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")


# Initialize the dynamic database

class SOPVoiceAgent(Agent):
    def __init__(self, chat_ctx: ChatContext) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions='''
            You are an AI agent that follows Standard Operating Procedures (SOPs) and has access to employee database.
            
            WORKFLOW:
            1. For any user query, ALWAYS start by retrieving relevant SOPs using `sop_retriever_tool`
            2. Analyze the SOP response and always treat it as set of instructions that you need to follow
            3. If there nothing for the resonse or no SOP than just say "I need more information". Don't give any generic response or knowledge.
            4. Provide comprehensive responses based on SOP guidance
            5. If it is not a query or question than you can be generic and answer as per suitable.
            
            Remember: SOP dictates what to do and what actions to take.
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
        """
        Retrieve relevant SOP instructions to determine workflow and database access needs.
        """
        start = time.time()
        try:
            print(f"📋 RETRIEVING SOP for query: '{query}'")
            
            docs = await vector_service.retrieval(
                query=query,
                namespace="SmaJ76_679UMK",
                doc_category="kb",
                rerank_threshold=0.1,
                topn=topn
            )
            
            instructions = "\n\n".join([doc.text for doc in docs])
            clean_instructions = instructions.replace('*', '').replace('#', '').replace('- ', '')
            clean_instructions = ' '.join(clean_instructions.split())
        
            end = time.time()
            logger.debug(f"✅ SOP Retrieval Time: {round(end - start, 2)} seconds")
            return {
                "sop_instructions": clean_instructions,
                "retrieved_docs_count": len(docs),
            }
            
        except Exception as e:
            print(f"❌ SOP Retrieval Error: {str(e)}")
            return {
                "error": str(e),
            }


# --------------------------
# Entrypoint
# --------------------------
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    try:
        metadata_str = ctx.job.metadata or "{}"
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError:
        metadata = {}

    user_name = metadata.get("user_name", "user")
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts = elevenlabs.TTS(
            voice_id="FIIBqolBA6JRqu2Lzpd7",
            model="eleven_multilingual_v2",  # ✅ correct key name
            api_key=ELEVEN_API_KEY,
        ),

        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    chat_ctx = ChatContext()
    # Add greeting message when agent initializes
    chat_ctx.add_message(
        role="assistant", 
        content=f"🎉 Welcome {user_name}! I'm now initializing and ready to assist you. I'm your SOP-guided AI assistant with access to comprehensive employee data including profiles, payroll, shifts, and performance metrics. I follow standard operating procedures to ensure accurate and consistent responses. How may I help you today?"
    )

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