import torch
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image
import glob
import os
import random


def populateTrainList(dir):

	clothes = glob.glob(dir+'/*CLEAN1*')

	tmp = sorted(glob.glob(dir+'/*CLEAN0*'))
	models = {}
	for path in tmp:
		PID = path.split("/")[-1].split("_")[0].split("ID")[-1]
		if PID in models:
			models[PID].append(path)
		else:
			models[PID] = []
			models[PID].append(path)

	return clothes, models


class domainTransferLoader(data.Dataset):


	def __init__(self, dataPath):
		self.clothes, self.models = populateTrainList(dataPath)
		self.index = -1
		print("Unique Clothes:",len(self.clothes))
		

	def __getitem__(self, index):

		#idx1 = np.random.randint(0, len(self.clothes))
		self.index = self.index + 1
		idx1 = self.index
		PID1 = self.clothes[idx1].split("/")[-1].split("_")[0].split("ID")[-1]

		idx2 = np.random.randint(0, len(self.clothes))
		PID2 = self.clothes[idx2].split("/")[-1].split("_")[0].split("ID")[-1]

		#print("This:",len(self.models[PID1]))
		idx3 = np.random.randint(0, len(self.models[PID1]))

		if idx2 == idx1:
			idx2 = np.random.randint(1, len(self.clothes))
			PID2 = self.clothes[idx2].split("/")[-1].split("_")[0].split("ID")[-1]

		imgPath1 = self.clothes[idx1]
		imgPath2 = self.clothes[idx2]
		imgPath3 = self.models[PID1][idx3]

		with open("testfile.txt", "a") as myfile:
			myfile.write(imgPath1+"\n")

		img1 = np.array(cv2.resize(cv2.imread(imgPath1),(64,64))[:,:,(2,1,0)],dtype=np.float32)
		img2 = np.array(cv2.resize(cv2.imread(imgPath2),(64,64))[:,:,(2,1,0)],dtype=np.float32)
		img3 = np.array(cv2.resize(cv2.imread(imgPath3),(64,64))[:,:,(2,1,0)],dtype=np.float32)

		img1 = (img1/127.5) - 1
		img2 = (img2/127.5) - 1
		img3 = (img3/127.5) - 1

		#print(np.min(img1))

		return torch.from_numpy(img1.transpose((2,0,1))), torch.from_numpy(img2.transpose((2,0,1))), torch.from_numpy(img3.transpose((2,0,1)))


	def __len__(self):
		ct = 0
		for key in list(self.models.keys()):
			ct += len(self.models[key])
		return ct#len(self.clothes)
			



