import os
import json
import torch
from torchvision import transforms
from PIL import Image
import random
from transformers import CLIPTokenizer

# === Функция для загрузки текстовых запросов и путей к изображениям ===
def load_prompts_and_images(data_path):
    """
    Загрузка текстовых запросов и путей к изображениям из файла.
    :param data_path: Путь к файлу с данными (JSON или PyTorch).
    :return: Списки текстовых запросов и путей к изображениям.
    """
    if data_path.endswith(".json"):
        with open(data_path, "r") as f:
            data = json.load(f)
        prompts = [entry["prompt"] for entry in data]
        image_paths = [entry.get("image_path", None) for entry in data]
    elif data_path.endswith(".pt"):
        data = torch.load(data_path)
        prompts = data["prompts"]
        image_paths = data["image_paths"]
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {data_path}")
    return prompts, image_paths

# === Функция для предобработки изображений ===
def preprocess_image(image_path, image_size=(256, 256)):
    """
    Загрузка и предобработка изображения.
    :param image_path: Путь к изображению.
    :param image_size: Размер изображения для изменения.
    :return: Предобработанное изображение (torch.Tensor).
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

# === Функция для токенизации текстовых запросов ===
def tokenize_prompts(prompts, tokenizer_name="openai/clip-vit-base-patch32", max_length=77):
    """
    Токенизация текстовых запросов с использованием CLIPTokenizer.
    :param prompts: Список текстовых запросов.
    :param tokenizer_name: Имя токенизатора (по умолчанию CLIP).
    :param max_length: Максимальная длина токенизированного текста.
    :return: Токенизированные запросы в формате PyTorch.
    """
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return tokenized

# === Функция для создания DataLoader ===
def create_dataloader(prompts, image_paths, batch_size=16, shuffle=True, image_size=(256, 256)):
    """
    Создание PyTorch DataLoader из текстовых запросов и изображений.
    :param prompts: Список текстовых запросов.
    :param image_paths: Список путей к изображениям.
    :param batch_size: Размер батча.
    :param shuffle: Перемешивание данных.
    :param image_size: Размер изображений для изменения.
    :return: DataLoader для обучения или оценки.
    """
    data = []

    for prompt, image_path in zip(prompts, image_paths):
        if image_path and os.path.exists(image_path):
            image = preprocess_image(image_path, image_size)
            data.append((prompt, image))
        else:
            print(f"Пропущено: {image_path}")

    def collate_fn(batch):
        prompts, images = zip(*batch)
        tokenized_prompts = tokenize_prompts(list(prompts))
        return tokenized_prompts, torch.stack(images)

    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# === Функция для разделения данных ===
def split_data(prompts, image_paths, split_ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Разделение данных на обучающую, валидационную и тестовую выборки.
    :param prompts: Список текстовых запросов.
    :param image_paths: Список путей к изображениям.
    :param split_ratios: Доли для разделения (по умолчанию 80%/10%/10%).
    :param seed: Зерно для воспроизводимости.
    :return: Кортеж из трёх частей данных: (train, val, test).
    """
    random.seed(seed)
    data = list(zip(prompts, image_paths))
    random.shuffle(data)

    train_size = int(len(data) * split_ratios[0])
    val_size = int(len(data) * split_ratios[1])

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data


# from utils.data_utils import load_prompts_and_images

# prompts, image_paths = load_prompts_and_images("data/processed_data.pt")
# print(prompts[:5])
# print(image_paths[:5])


# from utils.data_utils import create_dataloader

# dataloader = create_dataloader(prompts, image_paths, batch_size=8)
# for batch in dataloader:
#     tokenized_prompts, images = batch
#     print(tokenized_prompts["input_ids"].shape)  # (batch_size, max_length)
#     print(images.shape)  # (batch_size, 3, 256, 256)

# from utils.data_utils import split_data

# train_data, val_data, test_data = split_data(prompts, image_paths)
# print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
