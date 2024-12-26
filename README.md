# TattooTools
## Для запуска использовать команды:
```bash
docker build -t tattoo-generator .
docker run --rm -it -p 8000:8000 tattoo-generator
docker run --rm -it --gpus all -p 8000:8000 tattoo-generator
```