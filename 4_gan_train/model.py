import torch
import torch.nn as nn
import math

class Generator(nn.Module):

	def __init__(self):
		super(Generator, self).__init__()

		self.nc = 3
		self.ngf = 96
		self.ndf = 96

		## Activation 
		self.leakyrelu = nn.LeakyReLU(0.2,inplace=True)

		self.conv1 = nn.Conv2d(self.nc, self.ngf, 4, 2, 1, bias=False)

		self.conv2 = nn.Conv2d(self.ngf, self.ngf*2, 4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(self.ngf*2)

		self.conv3 = nn.Conv2d(self.ngf*2, self.ngf*4, 4, 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.ngf*4)	

		self.conv4 = nn.Conv2d(self.ngf*4, self.ngf*8, 4, 2, 1, bias=False)
		self.bn4 = nn.BatchNorm2d(self.ngf*8)	


		## Activation
		self.relu = nn.ReLU(inplace=True)

		self.deconv1 = nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 4, 2, 1, bias=False)
		self.bn_deconv1 = nn.BatchNorm2d(self.ngf*4)	

		self.deconv2 = nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 4, 2, 1, bias=False)
		self.bn_deconv2 = nn.BatchNorm2d(self.ngf*2)

		self.deconv3 = nn.ConvTranspose2d(self.ngf*2, self.ngf, 4, 2, 1, bias=False)
		self.bn_deconv3 = nn.BatchNorm2d(self.ngf)

		self.deconv4 = nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False)

		## Output b/w -1 and 1
		self.tanh = nn.Tanh()

	
	def forward(self, X):

		X = self.leakyrelu(self.conv1(X))
		X = self.leakyrelu(self.bn2(self.conv2(X)))
		X = self.leakyrelu(self.bn3(self.conv3(X)))
		X = self.leakyrelu(self.bn4(self.conv4(X)))

		X = self.relu(self.bn_deconv1(self.deconv1(X)))
		X = self.relu(self.bn_deconv2(self.deconv2(X)))
		X = self.relu(self.bn_deconv3(self.deconv3(X)))
		X = self.tanh(self.deconv4(X))
		#print(X)
		#print(X.shape)
		#brak
		return X



class DiscriminatorD(nn.Module):

	def __init__(self):
		super(DiscriminatorD, self).__init__()

		self.nc = 3
		self.ngf = 96
		self.ndf = 96

		## Activation 
		self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
		self.sigmoid = nn.Sigmoid()

		self.conv1 = nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False)

		self.conv2 = nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(self.ndf*2)

		self.conv3 = nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.ndf*4)	

		self.conv4 = nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias=False)
		self.bn4 = nn.BatchNorm2d(self.ndf*8)	

		self.conv5 = nn.Conv2d(self.ndf*8, 1, 4, 4, 0, bias=False)
	
	def forward(self, X):

		X = self.leakyrelu(self.conv1(X))
		X = self.leakyrelu(self.bn2(self.conv2(X)))
		X = self.leakyrelu(self.bn3(self.conv3(X)))
		X = self.leakyrelu(self.bn4(self.conv4(X)))
		X = self.conv5(X)
		return X.view(-1,1)


class DiscriminatorA(nn.Module):

	def __init__(self):
		super(DiscriminatorA, self).__init__()

		self.nc = 3
		self.ngf = 96
		self.ndf = 96

		## Activation 
		self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
		self.sigmoid = nn.Sigmoid()

		self.conv1 = nn.Conv2d(self.nc*2, self.ndf, 4, 2, 1, bias=False)

		self.conv2 = nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(self.ndf*2)

		self.conv3 = nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.ndf*4)	

		self.conv4 = nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias=False)
		self.bn4 = nn.BatchNorm2d(self.ndf*8)	

		self.conv5 = nn.Conv2d(self.ndf*8, 1, 4, 4, 0, bias=False)
	
	def forward(self, X):

		X = self.leakyrelu(self.conv1(X))
		X = self.leakyrelu(self.bn2(self.conv2(X)))
		X = self.leakyrelu(self.bn3(self.conv3(X)))
		X = self.leakyrelu(self.bn4(self.conv4(X)))
		X = self.conv5(X)
		return X.view(-1,1)























		






