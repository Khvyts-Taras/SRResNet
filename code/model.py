import torch.nn as nn

class ResBlock(nn.Module):
	def __init__(self, chanels):
		super(ResBlock, self).__init__()
		
		self.conv1 = nn.Sequential(
			nn.Conv2d(chanels, chanels, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1)
			)

		self.conv2 = nn.Sequential(
			nn.Conv2d(chanels, chanels, kernel_size=5, stride=1, padding=2),
			nn.LeakyReLU(0.1)
			)

	def forward(self, inp):
		x = self.conv1(inp)+inp
		x = self.conv2(x)+inp

		return x


class ChenBlock(nn.Module):
	def __init__(self, inp_chanels, out_chanels):
		super(ChenBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(inp_chanels, out_chanels, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1)
			)

	def forward(self, inp):
		x = self.conv(inp)

		return x


class UpBlock(nn.Module):
	def __init__(self, chanels):
		super(UpBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.ConvTranspose2d(chanels, chanels, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.1),
			nn.ConvTranspose2d(chanels, chanels, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.1)
			)

	def forward(self, inp):
		x = self.conv(inp)

		return x


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.model = nn.Sequential(
			ResBlock(3),
			ChenBlock(3, 24),
			ResBlock(24),
			ChenBlock(24, 48),
			ResBlock(48),

			nn.Dropout2d(0.2),

			UpBlock(48),
			
			ResBlock(48),
			ChenBlock(48, 24),
			ResBlock(24),
			ChenBlock(24, 3),
			ResBlock(3),
			nn.Tanh()
			)

	def forward(self, inp):
		x = self.model(inp)

		return x
