from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from diffusers import StableDiffusionPipeline
import torch
import uuid
import os

# Создаём маршрутизатор для генерации
router = APIRouter()

# Путь для сохранения сгенерированных изображений
OUTPUT_DIR = "./generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Загрузка модели (замените на вашу модель)
MODEL_PATH = "path_to_your_model"
try:
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")  # Используем GPU, если доступно
except Exception as e:
    raise RuntimeError(f"Ошибка при загрузке модели: {e}")

# Модель данных для запроса
class GenerateRequest(BaseModel):
    prompt: str

# Маршрут для генерации изображения
@router.post("/generate", response_class=FileResponse, summary="Генерация изображения")
async def generate_image(request: GenerateRequest):
    try:
        # Генерация уникального имени файла
        file_name = f"{uuid.uuid4()}.png"
        output_path = os.path.join(OUTPUT_DIR, file_name)
        
        # Генерация изображения
        prompt = request.prompt
        image = pipeline(prompt).images[0]
        
        # Сохранение изображения
        image.save(output_path)

        # Возвращаем файл пользователю
        return FileResponse(output_path, media_type="image/png", filename=file_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации изображения: {e}")
