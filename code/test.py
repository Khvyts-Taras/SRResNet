import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import Places365
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from model import *


transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = "cuda"
# Загрузка изображения из файла
image_path = "32.png"
image = Image.open(image_path)

# Применение трансформаций к изображению
image_tensor = transform(image).unsqueeze(0).to(device)



model = Model().to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()


# Проход изображения через модель
with torch.no_grad():
    output = model(image_tensor)
    #output = model(output)
    #output = F.interpolate(output, size=(32, 32))
    #output = model(output)
    #output = F.interpolate(output, size=(64, 64))

# Преобразование вывода модели к изображению
output_image = (output.squeeze().cpu().numpy() / 2.0 + 0.5).clip(0, 1)
output_image = (output_image * 255).astype('uint8')
output_image = Image.fromarray(output_image.transpose((1, 2, 0)))

# Сохранение обработанного изображения
output_image.save('result.png')