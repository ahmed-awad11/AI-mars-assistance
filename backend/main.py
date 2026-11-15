"""
Marswalk AI Assistant - Main FastAPI Application
Entry point for the backend API server.
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from config import settings
from models import (
    init_db, get_db, Conversation, Message, MissionContext,
    Document, DocumentChunk
)
from schemas import (
    ChatRequest, ChatResponse, ChatMessage,
    ConversationCreate, ConversationResponse, MessageResponse,
    MissionContextResponse, MissionContextUpdate,
    DocumentIngestRequest, DocumentResponse,
    SearchRequest, SearchResponse,
    ModelInfo, ModelSwitchRequest,
    HealthResponse,
    ToolDefinitionResponse, ToolExecuteRequest, ToolExecuteResponse,
    ToolResultResponse, ToolCallRequest
)
from llm import llm_client, LLMResponse, ToolCall
from tools import tool_registry, ToolResult, init_tools
from rag import knowledge_base

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Marswalk AI Assistant...")
    init_db()
    init_tools()
    await knowledge_base.initialize()
    await llm_client.list_models()
    logger.info("Marswalk AI Assistant started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Marswalk AI Assistant...")


app = FastAPI(
    title="Marswalk AI Assistant API",
    description="Intelligent mission control system for Mars exploration",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    ollama_ok = False
    try:
        models = await llm_client.list_models()
        ollama_ok = len(models) > 0
    except Exception:
        pass

    kb_ready = knowledge_base._initialized

    db_ok = False
    try:
        db = next(get_db())
        db.execute("SELECT 1")
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if (ollama_ok and db_ok) else "degraded",
        version=settings.APP_VERSION,
        ollama_connected=ollama_ok,
        model=llm_client.model,
        knowledge_base_ready=kb_ready,
        database_connected=db_ok
    )


# ============================================================================
# Model Management
# ============================================================================

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available Ollama models."""
    models = await llm_client.list_models()
    return [ModelInfo(name=m) for m in models]


@app.post("/models/switch", response_model=ModelInfo)
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model."""
    success = await llm_client.switch_model(request.model)
    if not success:
        raise HTTPException(status_code=400, detail=f"Model not available: {request.model}")
    return ModelInfo(name=llm_client.model)


@app.post("/models/pull", response_model=ModelInfo)
async def pull_model(request: ModelSwitchRequest):
    """Pull a model from Ollama registry."""
    success = await llm_client.pull_model(request.model)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to pull model: {request.model}")
    return ModelInfo(name=request.model)


# ============================================================================
# Conversation Management
# ============================================================================

@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationCreate, db: Session = Depends(get_db)):
    """Create a new conversation."""
    import uuid
    session_id = request.session_id or str(uuid.uuid4())[:8]

    conv = Conversation(
        session_id=session_id,
        title=request.title
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

    # Create empty mission context
    mission_ctx = MissionContext(conversation_id=conv.id)
    db.add(mission_ctx)
    db.commit()

    return ConversationResponse(
        id=conv.id,
        session_id=conv.session_id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        is_active=conv.is_active,
        message_count=0
    )


@app.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(limit: int = 20, db: Session = Depends(get_db)):
    """List conversations."""
    conversations = db.query(Conversation).filter(
        Conversation.is_active == True
    ).order_by(Conversation.updated_at.desc()).limit(limit).all()

    return [
        ConversationResponse(
            id=c.id,
            session_id=c.session_id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            is_active=c.is_active,
            message_count=len(c.messages)
        )
        for c in conversations
    ]


@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Get conversation details."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=conv.id,
        session_id=conv.session_id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        is_active=conv.is_active,
        message_count=len(conv.messages)
    )


@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(conversation_id: int, limit: int = 50, db: Session = Depends(get_db)):
    """Get messages for a conversation."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at).limit(limit).all()

    return [
        MessageResponse(
            id=m.id,
            conversation_id=m.conversation_id,
            role=m.role,
            content=m.content,
            tool_calls=m.tool_calls,
            tool_results=m.tool_results,
            metadata=m.metadata,
            created_at=m.created_at
        )
        for m in messages
    ]


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Delete a conversation."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.is_active = False
    db.commit()
    return {"success": True, "message": "Conversation archived"}


# ============================================================================
# Chat Endpoint
# ============================================================================

def _build_messages_for_llm(conversation_id: int, new_message: str, db: Session) -> List[Dict[str, str]]:
    """Build message history for LLM."""
    messages = []

    # Get recent messages
    recent = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.desc()).limit(settings.MAX_HISTORY_MESSAGES).all()

    # Reverse to chronological order
    for msg in reversed(recent):
        messages.append({"role": msg.role, "content": msg.content})

    # Add new user message
    messages.append({"role": "user", "content": new_message})

    return messages


def _save_message(db: Session, conversation_id: int, role: str, content: str,
                  tool_calls: List = None, tool_results: List = None, metadata: Dict = None) -> Message:
    """Save a message to the database."""
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_results=tool_results,
        metadata=metadata
    )
    db.add(msg)
    return msg


def _update_mission_context(db: Session, conversation_id: int, updates: Dict):
    """Update mission context from tool results."""
    ctx = db.query(MissionContext).filter(
        MissionContext.conversation_id == conversation_id
    ).first()

    if ctx:
        for key, value in updates.items():
            if hasattr(ctx, key) and value is not None:
                setattr(ctx, key, value)
        db.commit()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Main chat endpoint with tool calling."""
    # Get or create conversation
    if request.conversation_id:
        conv = db.query(Conversation).filter(
            Conversation.id == request.conversation_id
        ).first()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
    elif request.session_id:
        conv = db.query(Conversation).filter(
            Conversation.session_id == request.session_id
        ).first()
        if not conv:
            # Create new conversation with this session_id
            conv = Conversation(session_id=request.session_id, title="New Conversation")
            db.add(conv)
            db.commit()
            db.refresh(conv)

            # Create mission context
            mission_ctx = MissionContext(conversation_id=conv.id)
            db.add(mission_ctx)
            db.commit()
    else:
        # Create new conversation
        import uuid
        conv = Conversation(session_id=str(uuid.uuid4())[:8], title="New Conversation")
        db.add(conv)
        db.commit()
        db.refresh(conv)

        mission_ctx = MissionContext(conversation_id=conv.id)
        db.add(mission_ctx)
        db.commit()

    # Save user message
    user_msg = _save_message(db, conv.id, "user", request.message)
    db.commit()

    # Build message history
    messages = _build_messages_for_llm(conv.id, request.message, db)

    # Get available tools
    tools = tool_registry.get_tools_for_llm() if request.use_tools else []

    # Call LLM
    llm_response: LLMResponse = await llm_client.chat(
        messages=messages,
        tools=[type('ToolDef', (), {'name': t['name'], 'description': t['description'], 'parameters': t['parameters']})() for t in tools],
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens or settings.MAX_CONTEXT_TOKENS
    )

    # Execute tool calls
    tool_results = []
    frontend_actions = []

    for tool_call in llm_response.tool_calls:
        result: ToolResult = await tool_registry.execute(tool_call.name, tool_call.arguments)

        tool_results.append(ToolResultResponse(
            success=result.success,
            data=result.data,
            error=result.error,
            metadata=result.metadata
        ))

        # Collect frontend actions
        if result.metadata.get("frontend_action") and result.data:
            frontend_actions.append(result.data)

        # Update mission context from tool results
        if result.success and result.data:
            if tool_call.name == "generate_route":
                _update_mission_context(db, conv.id, {"current_route": result.data})
            elif tool_call.name == "compare_routes":
                _update_mission_context(db, conv.id, {"route_comparisons": result.data.get("routes", [])})
            elif tool_call.name == "highlight_map_location":
                _update_mission_context(db, conv.id, {"selected_point": {"lat": result.data["lat"], "lon": result.data["lon"]}})
            elif tool_call.name == "zoom_map":
                if "lat" in result.data and "lon" in result.data:
                    _update_mission_context(db, conv.id, {"map_center": {"lat": result.data["lat"], "lon": result.data["lon"], "zoom": result.data.get("zoom", 15)}})

    # Save assistant message
    assistant_msg = _save_message(
        db, conv.id, "assistant", llm_response.content,
        tool_calls=[tc.model_dump() for tc in llm_response.tool_calls],
        tool_results=[tr.model_dump() for tr in tool_results]
    )
    db.commit()

    # Update conversation timestamp
    conv.updated_at = datetime.utcnow()
    db.commit()

    # Convert ToolCall to ToolCallRequest for response
    tool_call_requests = [
        ToolCallRequest(name=tc.name, arguments=tc.arguments, id=tc.id)
        for tc in llm_response.tool_calls
    ]

    return ChatResponse(
        message=llm_response.content,
        conversation_id=conv.id,
        session_id=conv.session_id,
        tool_calls=tool_call_requests,
        tool_results=tool_results,
        actions=frontend_actions,
        metadata={"model": llm_response.model, "usage": llm_response.usage}
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, db: Session = Depends(get_db)):
    """Streaming chat endpoint (simplified - returns full response)."""
    # For now, delegate to regular chat
    return await chat(request, db)


# ============================================================================
# Mission Context
# ============================================================================

@app.get("/conversations/{conversation_id}/context", response_model=MissionContextResponse)
async def get_mission_context(conversation_id: int, db: Session = Depends(get_db)):
    """Get mission context for a conversation."""
    ctx = db.query(MissionContext).filter(
        MissionContext.conversation_id == conversation_id
    ).first()

    if not ctx:
        raise HTTPException(status_code=404, detail="Mission context not found")

    return MissionContextResponse(
        current_mission=ctx.current_mission,
        selected_area=ctx.selected_area,
        current_route=ctx.current_route,
        route_comparisons=ctx.route_comparisons,
        selected_layers=ctx.selected_layers,
        scientific_analyses=ctx.scientific_analyses,
        mission_objectives=ctx.mission_objectives,
        map_view=ctx.map_view,
        map_center=ctx.map_center,
        selected_point=ctx.selected_point
    )


@app.patch("/conversations/{conversation_id}/context", response_model=MissionContextResponse)
async def update_mission_context(
    conversation_id: int,
    updates: MissionContextUpdate,
    db: Session = Depends(get_db)
):
    """Update mission context."""
    ctx = db.query(MissionContext).filter(
        MissionContext.conversation_id == conversation_id
    ).first()

    if not ctx:
        raise HTTPException(status_code=404, detail="Mission context not found")

    update_data = updates.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(ctx, key, value)

    db.commit()
    db.refresh(ctx)

    return MissionContextResponse(
        current_mission=ctx.current_mission,
        selected_area=ctx.selected_area,
        current_route=ctx.current_route,
        route_comparisons=ctx.route_comparisons,
        selected_layers=ctx.selected_layers,
        scientific_analyses=ctx.scientific_analyses,
        mission_objectives=ctx.mission_objectives,
        map_view=ctx.map_view,
        map_center=ctx.map_center,
        selected_point=ctx.selected_point
    )


# ============================================================================
# Knowledge Base
# ============================================================================

@app.post("/knowledge/ingest", response_model=DocumentResponse)
async def ingest_document(request: DocumentIngestRequest):
    """Ingest a document into the knowledge base."""
    try:
        doc = await knowledge_base.ingest_document(
            filepath=request.filepath,
            title=request.title,
            source_type=request.source_type,
            metadata=request.metadata
        )
        return DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            filepath=doc.filepath,
            title=doc.title,
            source_type=doc.source_type,
            chunk_count=doc.chunk_count,
            is_processed=doc.is_processed,
            created_at=doc.created_at
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/ingest-directory", response_model=List[DocumentResponse])
async def ingest_directory(directory: Optional[str] = None):
    """Ingest all documents from a directory."""
    docs = await knowledge_base.ingest_directory(directory)
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            filepath=d.filepath,
            title=d.title,
            source_type=d.source_type,
            chunk_count=d.chunk_count,
            is_processed=d.is_processed,
            created_at=d.created_at
        )
        for d in docs
    ]


@app.get("/knowledge/documents", response_model=List[DocumentResponse])
async def list_documents(limit: int = 50):
    """List ingested documents."""
    docs = await knowledge_base.list_documents(limit)
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            filepath=d.filepath,
            title=d.title,
            source_type=d.source_type,
            chunk_count=d.chunk_count,
            is_processed=d.is_processed,
            created_at=d.created_at
        )
        for d in docs
    ]


@app.delete("/knowledge/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete a document from the knowledge base."""
    success = await knowledge_base.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"success": True, "message": "Document deleted"}


@app.post("/knowledge/search", response_model=SearchResponse)
async def search_knowledge(request: SearchRequest):
    """Search the knowledge base."""
    results = await knowledge_base.search(
        query=request.query,
        top_k=request.top_k,
        filter=request.filter
    )
    return SearchResponse(
        query=request.query,
        results=[SearchResult(**r) for r in results]
    )


# ============================================================================
# Tools
# ============================================================================

@app.get("/tools", response_model=List[ToolDefinitionResponse])
async def list_tools():
    """List all available tools."""
    tools = tool_registry.list_tools()
    return [
        ToolDefinitionResponse(
            name=t.name,
            description=t.description,
            parameters=t.to_json_schema()["parameters"],
            returns=t.returns,
            requires_permission=t.requires_permission,
            category=t.category
        )
        for t in tools
    ]


@app.post("/tools/execute", response_model=ToolExecuteResponse)
async def execute_tool(request: ToolExecuteRequest):
    """Execute a tool directly."""
    result = await tool_registry.execute(request.name, request.arguments)
    return ToolExecuteResponse(
        success=result.success,
        data=result.data,
        error=result.error,
        metadata=result.metadata
    )


# ============================================================================
# Run server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )