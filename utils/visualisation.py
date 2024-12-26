import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def display_images(images, titles=None, cols=3, figsize=(15, 10)):
    """
    Отображение списка изображений.
    :param images: Список изображений (в формате PIL или numpy).
    :param titles: Список заголовков для изображений.
    :param cols: Количество столбцов для отображения.
    :param figsize: Размер фигуры.
    """
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=figsize)
    
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if isinstance(img, Image.Image):  # Если изображение в формате PIL
            img = np.array(img)
        plt.imshow(img)
        if titles:
            plt.title(titles[i], fontsize=12)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_loss(losses, title="Loss Curve", save_path=None):
    """
    Построение графика потерь.
    :param losses: Список значений потерь.
    :param title: Заголовок графика.
    :param save_path: Путь для сохранения графика (если None, график не сохраняется).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", label="Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"График сохранён: {save_path}")
    plt.show()

def save_generated_images(images, output_dir, prefix="generated"):
    """
    Сохранение сгенерированных изображений в указанную папку.
    :param images: Список изображений (в формате PIL).
    :param output_dir: Папка для сохранения изображений.
    :param prefix: Префикс для имён файлов.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        file_path = os.path.join(output_dir, f"{prefix}_{i + 1}.png")
        img.save(file_path)
        print(f"Сохранено изображение: {file_path}")

def compare_images(real_images, generated_images, titles=None, cols=2, figsize=(15, 10)):
    """
    Сравнение реальных и сгенерированных изображений.
    :param real_images: Список реальных изображений (в формате PIL).
    :param generated_images: Список сгенерированных изображений (в формате PIL).
    :param titles: Список заголовков для изображений (опционально).
    :param cols: Количество столбцов (по умолчанию 2: реальное и сгенерированное).
    :param figsize: Размер фигуры.
    """
    num_images = len(real_images)
    plt.figure(figsize=figsize)
    
    for i in range(num_images):
        plt.subplot(num_images, cols, i * cols + 1)
        plt.imshow(np.array(real_images[i]))
        if titles:
            plt.title(f"Real: {titles[i]}", fontsize=12)
        plt.axis("off")
        
        plt.subplot(num_images, cols, i * cols + 2)
        plt.imshow(np.array(generated_images[i]))
        if titles:
            plt.title(f"Generated: {titles[i]}", fontsize=12)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

