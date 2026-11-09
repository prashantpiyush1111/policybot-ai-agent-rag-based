"""
FastAPI routes for the AI Agent API.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging

from app.models.schemas import (
    AskRequest,
    AskResponse,
    DetailedAskResponse,
    Source,
    HealthResponse,
    ErrorResponse,
    SessionInfo
)
from app.agents.agent import agent
from app.config import settings
from app.api.security import verify_api_key

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, enabled=settings.rate_limit_enabled)
router = APIRouter(dependencies=[Depends(verify_api_key)])
public_router = APIRouter()


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question",
    description="Submit a query to the AI agent. The agent will decide whether to answer directly or retrieve from documents.",
    responses={
        200: {"model": AskResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def ask_question(request: Request, body: AskRequest) -> AskResponse:
    """
    Main endpoint for asking questions to the AI agent.
    
    The agent will:
    1. Classify the query (general knowledge vs policy-specific)
    2. Use RAG if policy-specific, direct answer otherwise
    3. Maintain conversation context if session_id is provided
    """
    try:
        logger.info(f"Received query: {body.query[:100]}")
        
        # Process query with agent
        response = await agent.process_query(
            query=body.query,
            session_id=body.session_id
        )
        
        return AskResponse(
            answer=response["answer"],
            source=response["source"],
            session_id=response["session_id"],
            metadata=response.get("metadata")
        )
    
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.post(
    "/ask/detailed",
    response_model=DetailedAskResponse,
    summary="Ask a question with detailed response",
    description="Get a detailed response including processing time and method used"
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def ask_question_detailed(request: Request, body: AskRequest) -> DetailedAskResponse:
    """Get detailed response with more metadata."""
    try:
        response = await agent.process_query(
            query=body.query,
            session_id=body.session_id
        )
        
        metadata = response.get("metadata", {})
        
        # Convert source strings to Source objects
        sources = [
            Source(
                document=source,
                relevance_score=None
            )
            for source in response["source"]
        ]
        
        return DetailedAskResponse(
            answer=response["answer"],
            sources=sources,
            session_id=response["session_id"],
            method=metadata.get("method", "unknown"),
            processing_time=metadata.get("processing_time", 0),
            metadata=metadata
        )
    
    except Exception as e:
        logger.error(f"Error processing detailed request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@public_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and configuration"
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        environment=settings.environment,
        llm_provider="gemini" if settings.use_gemini else "azure",
        vector_store="faiss" if settings.use_faiss else "azure_search"
    )


@public_router.get(
    "/ready",
    summary="Readiness check",
    description="Check if API is ready to serve requests (for Kubernetes)"
)
async def readiness_check():
    """Readiness check for deployment orchestration."""
    try:
        # Could add more sophisticated checks here
        # e.g., check vector store connection, LLM availability
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@router.get(
    "/session/{session_id}",
    response_model=SessionInfo,
    summary="Get session info",
    description="Get information about a conversation session"
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def get_session(request: Request, session_id: str) -> SessionInfo:
    """Get session information."""
    session_info = await agent.get_session_info(session_id)
    
    if not session_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    return SessionInfo(**session_info)


@router.get(
    "/stats",
    summary="Get agent statistics",
    description="Get statistics about the agent and memory usage"
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def get_stats(request: Request):
    """Get agent statistics."""
    return agent.get_stats()
