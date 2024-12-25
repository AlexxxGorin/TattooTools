import os
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# === Загрузка конфигурации ===
def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# === Загрузка данных ===
def load_data(data_path, batch_size):
    data = torch.load(data_path)
    prompts, image_paths = data["prompts"], data["image_paths"]
    return prompts, image_paths

# === Функция для генерации изображений ===
def generate_images(prompts, model, text_encoder, tokenizer, scheduler, device):
    generated_images = []
    model.eval()

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Генерация изображений"):
            # Токенизация текста
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

            # Генерация случайного шума
            noise = torch.randn((1, 3, 256, 256)).to(device)

            # Декодирование изображения
            image = scheduler.step(noise, model, text_embeds=text_encoder(**inputs).last_hidden_state)["prev_sample"]

            # Постобработка
            image = (image[0].cpu().clamp(-1, 1) + 1) / 2  # Масштабирование от -1..1 в 0..1
            generated_images.append(ToPILImage()(image))
    return generated_images

# === Оценка качества ===
def calculate_metrics(real_images, generated_images):
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    ssim_scores = []
    psnr_scores = []

    for real, generated in zip(real_images, generated_images):
        real = np.array(real).astype(np.float32) / 255.0
        generated = np.array(generated).astype(np.float32) / 255.0

        # Вычисление SSIM и PSNR
        ssim_scores.append(ssim(real, generated, multichannel=True))
        psnr_scores.append(psnr(real, generated))

    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    return avg_ssim, avg_psnr

# === Основная функция ===
def main():
    # Загрузка конфигурации
    config = load_config()

    # Установка устройства
    device = config["training"]["device"]

    # Загрузка данных
    prompts, image_paths = load_data(config["data"]["data_path"], config["data"]["batch_size"])

    # Загрузка модели
    text_encoder = CLIPTextModel.from_pretrained(config["model"]["text_encoder"]).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(config["model"]["text_encoder"])
    unet = UNet2DConditionModel.from_pretrained(config["model"]["unet"]).to(device)
    scheduler = DDPMScheduler.from_pretrained(config["model"]["scheduler"])

    # Генерация изображений
    generated_images = generate_images(prompts, unet, text_encoder, tokenizer, scheduler, device)

    # Загрузка реальных изображений
    real_images = [Image.open(path) for path in image_paths]

    # Оценка качества
    avg_ssim, avg_psnr = calculate_metrics(real_images, generated_images)
    print(f"Средний SSIM: {avg_ssim:.4f}")
    print(f"Средний PSNR: {avg_psnr:.4f}")

    fid = FrechetInceptionDistance(feature=2048).to(device)
    for real, generated in zip(real_images, generated_images):
        fid.update(real.unsqueeze(0), real=True)
        fid.update(generated.unsqueeze(0), real=False)

    print(f"FID: {fid.compute().item():.4f}")

    # Сохранение примеров изображений
    os.makedirs("evaluation_results", exist_ok=True)
    for i, img in enumerate(generated_images[:5]):  # Сохраним первые 5 изображений
        img.save(f"evaluation_results/generated_{i}.png")
        real_images[i].save(f"evaluation_results/real_{i}.png")

    print("Результаты сохранены в папке evaluation_results.")

if __name__ == "__main__":
    main()
