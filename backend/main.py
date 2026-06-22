import logging
import asyncio
import httpx
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import(
    ALLOWED_ORIGINS,
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    SENTENCE_TRANSFORMER_MODEL
)
from backend.api.routes import router

logger = logging.getLogger('ats_resume_scorer')

def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

async def _keep_alive():
    url = os.getenv('RENDER_EXTERNAL_URL', '')
    if not url:
        return
    await asyncio.sleep(60)
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await client.get(f'{url}/api/v1/health', timeout=10)
                logger.info('keep-alive ping sent')
            except Exception:
                pass
            await asyncio.sleep(600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.nlp      = None
    app.state.embedder = None
    asyncio.create_task(_keep_alive())
    logger.info('API started — models will load on first request')
    yield
    logger.info('Shutting down the API')

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url='/docs',
    redoc_url='/redoc'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router)

@app.get('/')
async def root():
    return {
        'name':      'ATS Resume Analyzer API',
        'version':   '2.0.0',
        'endpoints': {
            'POST   /api/v1/analyze-resume': 'Analyze a resume',
            'GET    /api/v1/history':        'Get user history',
            'DELETE /api/v1/history/:id':    'Delete a history entry',
            'GET    /api/v1/health':         'Health check',
            'POST   /api/v1/generate-pdf':   'Generate PDF report from data',
        },
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'backend.main:app',
        host   = '0.0.0.0',
        port   = 8000,
        reload = True,
    )
