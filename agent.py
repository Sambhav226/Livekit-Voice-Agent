import os
import json
import logging
import time
from loguru import logger
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, ChatContext, AgentSession, RunContext, RoomInputOptions, ModelSettings, stt
from livekit.plugins import (
    openai,
    deepgram,
    noise_cancellation,
    silero,
    elevenlabs,
    sarvam
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from VectorSearch import VectorSearch
from DynamicUserDatabase import DynamicUserDatabase
from typing import AsyncIterable, Optional

load_dotenv()

# --------------------------
# Initialize Retrieval Service
# --------------------------
PINECONE_HOST = os.getenv("PINECONE_HOST", "your-pinecone-host")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "your-index-name")
vector_service = VectorSearch(index_name=PINECONE_INDEX)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")


class SOPVoiceAgent(Agent):
    def __init__(self, chat_ctx: ChatContext) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions='''
            You are an AI agent that follows Standard Operating Procedures (SOPs) and has access to employee database.
            
            WORKFLOW:
            1. You will receive user queries along with relevant SOP context that has been automatically retrieved
            2. Analyze the SOP context and treat it as instructions that you need to follow
            3. If the SOP context is empty or irrelevant, say "I need more information". Don't give any generic response or knowledge.
            4. Provide comprehensive responses based on SOP guidance
            5. If it is not a query or question, you can be generic and answer as suitable.
            
            Remember: SOP dictates what to do and what actions to take.
            '''
        )

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        """
        Custom STT node that retrieves SOP context after transcription
        """
        async def process_transcription():
            # Get the default STT transcription
            async for event in Agent.default.stt_node(self, audio, model_settings):
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    # Get the transcribed text
                    user_query = event.alternatives[0].text
                    logger.info(f"📝 User Query: {user_query}")
                    
                    # Retrieve SOP context using VectorSearch
                    try:
                        start_time = time.time()
                        logger.info(f"📋 RETRIEVING SOP for query: '{user_query}'")
                        
                        docs = await vector_service.retrieval(
                            query=user_query,
                            namespace="SmaJ76_679UMK",
                            doc_category="kb",
                            rerank_threshold=0.1,
                            topn=5
                        )
                        
                        # Process retrieved documents
                        if docs:
                            instructions = "\n\n".join([doc.text for doc in docs])
                            clean_instructions = instructions.replace('*', '').replace('#', '').replace('- ', '')
                            clean_instructions = ' '.join(clean_instructions.split())
                            
                            # Modify the transcribed text to include SOP context
                            enhanced_query = f"""
                            User Query: {user_query}
                            
                            Relevant SOP Context:
                            {clean_instructions}
                            
                            Please respond based on the SOP context provided above.
                            """
                            
                            logger.info(f"✅ SOP Retrieved: {len(docs)} documents")
                        else:
                            enhanced_query = f"""
                            User Query: {user_query}
                            
                            Relevant SOP Context: No relevant SOP found.
                            
                            Please respond that you need more information as no relevant SOP was found.
                            """
                            logger.warning("⚠️ No SOP documents found")
                        
                        end_time = time.time()
                        logger.debug(f"✅ SOP Retrieval Time: {round(end_time - start_time, 2)} seconds")
                        
                        # Create a new speech event with enhanced text
                        enhanced_event = stt.SpeechEvent(
                            type=event.type,
                            alternatives=[
                                stt.SpeechData(
                                    text=enhanced_query,
                                    confidence=event.alternatives[0].confidence,
                                    language=event.alternatives[0].language
                                )
                            ]
                        )
                        
                        yield enhanced_event
                        
                    except Exception as e:
                        logger.error(f"❌ SOP Retrieval Error: {str(e)}")
                        # Fallback: yield original event if retrieval fails
                        fallback_query = f"""
                        User Query: {user_query}
                        
                        Relevant SOP Context: Error retrieving SOP context.
                        
                        Please respond that there was an error retrieving information.
                        """
                        
                        enhanced_event = stt.SpeechEvent(
                            type=event.type,
                            alternatives=[
                                stt.SpeechData(
                                    text=fallback_query,
                                    confidence=event.alternatives[0].confidence,
                                    language=event.alternatives[0].language
                                )
                            ]
                        )
                        
                        yield enhanced_event
                else:
                    # For non-final transcripts, pass through unchanged
                    yield event
        
        return process_transcription()


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
        tts=sarvam.TTS(target_language_code="hi-IN", speaker="manisha", api_key=SARVAM_API_KEY),
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