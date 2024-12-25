import os
import yaml
import torch
from torchvision.transforms import ToPILImage
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from PIL import Image

# === Загрузка конфигурации ===
def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# === Генерация изображений ===
def generate_images(prompts, model, text_encoder, tokenizer, scheduler, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx, prompt in enumerate(tqdm(prompts, desc="Генерация изображений")):
            # Токенизация текста
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

            # Генерация случайного шума
            noise = torch.randn((1, 3, 256, 256)).to(device)

            # Декодирование изображения
            image = scheduler.step(noise, model, text_embeds=text_encoder(**inputs).last_hidden_state)["prev_sample"]

            # Постобработка
            image = (image[0].cpu().clamp(-1, 1) + 1) / 2  # Масштабирование от -1..1 в 0..1
            image = ToPILImage()(image)

            # Сохранение изображения
            output_path = os.path.join(output_dir, f"generated_{idx + 1}.png")
            image.save(output_path)
            print(f"Сохранено: {output_path}")

# === Основная функция ===
def main():
    # Загрузка конфигурации
    config = load_config()

    # Установка устройства
    device = config["training"]["device"]

    # Загрузка модели
    text_encoder = CLIPTextModel.from_pretrained(config["model"]["text_encoder"]).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(config["model"]["text_encoder"])
    unet = UNet2DConditionModel.from_pretrained(config["model"]["unet"]).to(device)
    scheduler = DDPMScheduler.from_pretrained(config["model"]["scheduler"])

    # Загрузка текстовых запросов
    prompts_file = "data/prompts.json"
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Файл {prompts_file} не найден.")
    with open(prompts_file, "r") as f:
        prompts = [entry["prompt"] for entry in yaml.safe_load(f)]

    # Папка для сохранения результатов
    output_dir = "generated_images"

    # Генерация изображений
    generate_images(prompts, unet, text_encoder, tokenizer, scheduler, output_dir, device)

if __name__ == "__main__":
    main()
