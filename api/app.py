from fastapi import FastAPI
from api.routes.generate_route import router as generate_router

# Инициализация FastAPI
app = FastAPI(title="Tattoo Generator API", description="API для генерации эскизов татуировок.")

# Подключение маршрутов
app.include_router(generate_router, prefix="/api", tags=["Generation"])

# Маршрут для проверки статуса API
@app.get("/", summary="Статус API")
async def health_check():
    return {"status": "ok", "message": "Tattoo Generator API is running!"}
