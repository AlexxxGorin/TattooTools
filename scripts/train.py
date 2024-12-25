import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from tqdm import tqdm

# === Чтение конфигурации из config.yaml ===
def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# === Загрузка данных ===
def load_data(data_path, batch_size):
    data = torch.load(data_path)
    prompts, image_paths = data["prompts"], data["image_paths"]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = [
        (prompts[i], transform(torch.tensor(np.array(Image.open(image_paths[i])))))
        for i in range(len(prompts))
    ]
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Петля обучения ===
def train_one_epoch(model, text_encoder, tokenizer, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for prompts_batch, images_batch in tqdm(dataloader, desc="Training"):
        # Токенизация текстов
        inputs = tokenizer(prompts_batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Подготовка изображений
        images_batch = images_batch.to(device)

        # Генерация шума
        noise = torch.randn_like(images_batch).to(device)
        noisy_images = scheduler.add_noise(images_batch, noise)

        # Прямой проход
        optimizer.zero_grad()
        noise_pred = model(noisy_images, text_embeds=text_encoder(**inputs).last_hidden_state).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Обратное распространение и оптимизация
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# === Основная функция ===
def main():
    # Загрузка конфигурации
    config = load_config()

    # Создание директорий
    os.makedirs(config["model"]["model_save_path"], exist_ok=True)

    # Загрузка данных
    dataloader = load_data(
        data_path=config["data"]["data_path"],
        batch_size=config["data"]["batch_size"]
    )

    # Инициализация модели и компонентов
    text_encoder = CLIPTextModel.from_pretrained(config["model"]["text_encoder"]).to(config["training"]["device"])
    tokenizer = CLIPTokenizer.from_pretrained(config["model"]["text_encoder"])
    unet = UNet2DConditionModel.from_pretrained(config["model"]["unet"]).to(config["training"]["device"])
    scheduler = DDPMScheduler.from_pretrained(config["model"]["scheduler"])

    optimizer = Adam(unet.parameters(), lr=config["training"]["learning_rate"])

    # Обучение
    for epoch in range(config["training"]["num_epochs"]):
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        epoch_loss = train_one_epoch(unet, text_encoder, tokenizer, dataloader, optimizer, scheduler, config["training"]["device"])
        print(f"Loss: {epoch_loss:.4f}")

        # Сохранение чекпойнта
        checkpoint_path = os.path.join(config["model"]["model_save_path"], f"unet_epoch_{epoch + 1}.pt")
        torch.save(unet.state_dict(), checkpoint_path)
        print(f"Сохранён чекпойнт: {checkpoint_path}")

if __name__ == "__main__":
    main()
