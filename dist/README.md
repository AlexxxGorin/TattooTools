# Примеры использования utils
# Загрузка данных:
```python
from tattoolib.utils.data_utils import load_prompts_and_images

prompts, image_paths = load_prompts_and_images("data/processed_data.pt")
print(prompts[:5])
print(image_paths[:5])
```

## Создание DataLoader:
```python
from tattoolib.utils.data_utils import create_dataloader

dataloader = create_dataloader(prompts, image_paths, batch_size=8)
for batch in dataloader:
    tokenized_prompts, images = batch
    print(tokenized_prompts["input_ids"].shape)  # (batch_size, max_length)
    print(images.shape)  # (batch_size, 3, 256, 256)
```

## Разделение данных:
```python
from tattoolib.utils.data_utils import split_data

train_data, val_data, test_data = split_data(prompts, image_paths)
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
```


# Визуализация изображений:
```python
from tattoolib.utils.visualisation import display_images

# Пример изображений (в формате PIL)
images = [Image.open(f"generated_images/generated_{i + 1}.png") for i in range(5)]
display_images(images, titles=[f"Image {i + 1}" for i in range(5)])
```
## Построение графика потерь:
```python
from tattoolib.utils.visualisation import plot_loss

# Пример потерь
losses = [0.9, 0.8, 0.7, 0.6, 0.5]
plot_loss(losses, title="Training Loss", save_path="logs/loss_curve.png")
```
## Сохранение сгенерированных изображений:
```python
from tattoolib.utils.visualisation import save_generated_images

# Пример изображений (в формате PIL)
images = [Image.open(f"generated_images/generated_{i + 1}.png") for i in range(5)]
save_generated_images(images, output_dir="results", prefix="tattoo")
```

## Сравнение реальных и сгенерированных изображений:
```python
from tattoolib.utils.visualisation import compare_images

# Пример реальных и сгенерированных изображений
real_images = [Image.open(f"real_images/real_{i + 1}.png") for i in range(3)]
generated_images = [Image.open(f"generated_images/generated_{i + 1}.png") for i in range(3)]
compare_images(real_images, generated_images, titles=["Example 1", "Example 2", "Example 3"])
```

# Загрузка моделей
## Загрузка UNet модели:
```python
from tattoolib.utils.model_utils import load_unet_model
unet_model = load_unet_model("path_to_unet", device="cuda")
```

## Загрузка текстового энкодера CLIP:
```python
from tattoolib.utils.model_utils import load_clip_text_encoder
text_encoder, tokenizer = load_clip_text_encoder(
    "openai/clip-vit-base-patch32", 
    "openai/clip-vit-base-patch32", 
    device="cuda"
)
```

## Сохранение и загрузка чекпойнта:
```python
from tattoolib.utils.model_utils import save_model_checkpoint, load_model_checkpoint

# Сохранение модели
save_model_checkpoint(unet_model, "models/checkpoints/unet_epoch_10")

# Загрузка модели из чекпойнта
unet_model = load_model_checkpoint(unet_model, "models/checkpoints/unet_epoch_10/pytorch_model.bin")
```


## Вывод информации о модели:
```python
from tattoolib.utils.model_utils import get_model_summary
total_params, trainable_params = get_model_summary(unet_model)
```