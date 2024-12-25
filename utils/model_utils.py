import os
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# === Функция для загрузки UNet модели ===
def load_unet_model(model_path, device="cuda"):
    """
    Загрузка UNet модели.
    :param model_path: Путь к обученной модели или предварительно обученному чекпойнту.
    :param device: Устройство (CPU или GPU).
    :return: UNet модель.
    """
    model = UNet2DConditionModel.from_pretrained(model_path)
    model = model.to(device)
    print(f"UNet модель загружена из {model_path}.")
    return model

# === Функция для загрузки текстового энкодера CLIP ===
def load_clip_text_encoder(encoder_path, tokenizer_path, device="cuda"):
    """
    Загрузка текстового энкодера и токенизатора CLIP.
    :param encoder_path: Путь к модели CLIPTextModel.
    :param tokenizer_path: Путь к токенизатору CLIPTokenizer.
    :param device: Устройство (CPU или GPU).
    :return: Кортеж (текстовый энкодер, токенизатор).
    """
    text_encoder = CLIPTextModel.from_pretrained(encoder_path).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    print(f"CLIPTextModel и CLIPTokenizer загружены из {encoder_path}.")
    return text_encoder, tokenizer

# === Функция для загрузки планировщика DDPMScheduler ===
def load_scheduler(scheduler_path):
    """
    Загрузка планировщика шума.
    :param scheduler_path: Путь к DDPMScheduler.
    :return: Планировщик.
    """
    scheduler = DDPMScheduler.from_pretrained(scheduler_path)
    print(f"Планировщик шума загружен из {scheduler_path}.")
    return scheduler

# === Функция для сохранения чекпойнта модели ===
def save_model_checkpoint(model, output_path):
    """
    Сохранение модели в указанную директорию.
    :param model: Модель для сохранения.
    :param output_path: Путь для сохранения модели.
    """
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    print(f"Модель сохранена в {output_path}.")

# === Функция для загрузки чекпойнта ===
def load_model_checkpoint(model, checkpoint_path):
    """
    Загрузка весов модели из чекпойнта.
    :param model: Модель для загрузки весов.
    :param checkpoint_path: Путь к файлу чекпойнта.
    """
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Чекпойнт модели загружен из {checkpoint_path}.")
    else:
        print(f"Чекпойнт {checkpoint_path} не найден.")
    return model

# === Функция для вывода информации о модели ===
def get_model_summary(model):
    """
    Вывод информации о модели (количество параметров).
    :param model: Модель для анализа.
    :return: Количество параметров.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Общее количество параметров: {total_params:,}")
    print(f"Обучаемые параметры: {trainable_params:,}")
    return total_params, trainable_params


# Загрузка UNet модели:
# from utils.model_utils import load_unet_model

# unet_model = load_unet_model("path_to_unet", device="cuda")


# Загрузка текстового энкодера CLIP:
# from utils.model_utils import load_clip_text_encoder

# text_encoder, tokenizer = load_clip_text_encoder(
#     "openai/clip-vit-base-patch32", 
#     "openai/clip-vit-base-patch32", 
#     device="cuda"
# )


# Сохранение и загрузка чекпойнта:
# from utils.model_utils import save_model_checkpoint, load_model_checkpoint

# # Сохранение модели
# save_model_checkpoint(unet_model, "models/checkpoints/unet_epoch_10")

# # Загрузка модели из чекпойнта
# unet_model = load_model_checkpoint(unet_model, "models/checkpoints/unet_epoch_10/pytorch_model.bin")



# Вывод информации о модели:
# from utils.model_utils import get_model_summary

# total_params, trainable_params = get_model_summary(unet_model)
