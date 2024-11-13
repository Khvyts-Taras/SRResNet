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

from model import *


transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((128, 128)),
	transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


batch_size = 32
dataloader = torch.utils.data.DataLoader(Places365("/data/Places365", split='val', small=True, download=False, transform=transform), batch_size=batch_size, shuffle=True)

device = "cuda"
model = Model().to(device)

vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
fe_net = vgg16.features[0:3].to(device)

def rec_loss(x, x_hat):
	loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
	loss += nn.functional.mse_loss(fe_net(x_hat), fe_net(x), reduction='mean')/2

	return loss


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
epochs = 500
for epoch in range(epochs):
	for i, (img, _) in enumerate(dataloader):
		optimizer.zero_grad()
		img = img.to(device)
		small_img  = F.interpolate(img, size=(32, 32), mode='nearest')

		rec = model(small_img)
		loss = rec_loss(rec, img)

		loss.backward()
		optimizer.step()

		if i%100 == 0:
			print(loss.item())

			save_image(rec/2+0.5, f'images/{epoch}_{i}_rec.png')
			torch.save(model.state_dict(), f'model.pt')
