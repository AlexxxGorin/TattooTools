from fastapi import FastAPI
from api.routes.generate_route import router as generate_router

app = FastAPI(title="Tattoo Generator API", description="API для генерации эскизов татуировок.")
app.include_router(generate_router, prefix="/api", tags=["Generation"])

@app.get("/", summary="Статус API")
async def health_check():
    return {"status": "ok", "message": "Tattoo Generator API is running!"}
