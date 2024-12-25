# Указываем базовый образ с поддержкой PyTorch и CUDA
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Указываем порт для API (если используется FastAPI или Flask)
EXPOSE 8000

# Команда по умолчанию для запуска контейнера (например, запускаем FastAPI)
CMD ["python", "scripts/generate.py"]


# docker build -t tattoo-generator .
# docker run --rm -it -p 8000:8000 tattoo-generator
# docker run --rm -it --gpus all -p 8000:8000 tattoo-generator