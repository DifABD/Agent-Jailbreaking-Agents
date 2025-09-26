"""FastAPI application entry point."""

from fastapi import FastAPI
from src.config import get_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Agent Jailbreak Research API",
        description="API for managing and analyzing LLM agent jailbreaking experiments",
        version="0.1.0",
        debug=settings.debug,
    )
    
    # TODO: Add routers and middleware
    
    @app.get("/")
    async def root():
        return {"message": "Agent Jailbreak Research API", "version": "0.1.0"}
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    return app


app = create_app()


def main():
    """Entry point for running the server."""
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()