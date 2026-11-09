"""
Main FastAPI application.
"""

import logging
import sys
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.security import validate_security_config, verify_api_key
from app.api.routes import limiter, public_router, router
from app.config import settings
from app.agents.memory import memory

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the application."""
    # Startup
    validate_security_config()
    logger.info("Starting AI Agent API...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"LLM Provider: {'Google Gemini' if settings.use_gemini else 'Azure OpenAI'}")
    logger.info(f"Vector Store: {'FAISS' if settings.use_faiss else 'Azure AI Search'}")
    
    # Start memory cleanup task
    import asyncio
    cleanup_task = asyncio.create_task(memory.start_cleanup_task())
    
    logger.info("✅ Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    cleanup_task.cancel()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="AI Agent with RAG",
    description="Production-ready AI agent that intelligently answers questions using LLM and document retrieval",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.environment == "local" else "An error occurred"
        }
    )


# Include routers
app.include_router(router, prefix="", tags=["AI Agent"])
app.include_router(public_router, prefix="", tags=["Health"])


# Root endpoint
@app.get("/", tags=["Root"], dependencies=[Depends(verify_api_key)])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Agent with RAG",
        "version": "1.0.0",
        "environment": settings.environment,
        "provider": "gemini" if settings.use_gemini else "azure",
        "vector_store": "faiss" if settings.use_faiss else "azure_search",
        "endpoints": {
            "ask": "/ask",
            "health": "/health",
            "docs": "/docs"
        }
    }


def main():
    """Run the application."""
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
