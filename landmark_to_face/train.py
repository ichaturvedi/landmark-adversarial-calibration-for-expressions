import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import data
import model
import numpy as np
from numpy import genfromtxt
import scipy.linalg as la


def get_lips(lossG1, lossD1, lossA1):

    my_data = genfromtxt('popt_low.txt', delimiter=',')

    error = np.array([[lossG1, lossD1, lossA1]])

    v1 = np.matmul(error,my_data)
    v1 = np.matmul(v1,error.transpose())
    #print(v1[0,0])

    v2 = np.matmul(error.transpose(),error)
    eigvals, eigvecs = la.eig(v2)

    v2 = eigvals[0].real
    #print(str(v1[0,0])+" "+str(v2))

    check = 0
    if v2 > v1[0,0]:
    	check = 1
    else:
    	check = 0
    	#print("not updating")

    return check

def train_val(datadir, epochs, models):

	generator = model.Generator().cuda() # Generator
	discriminatorD = model.DiscriminatorD().cuda() # Real-Fake Discriminator
	discriminatorA = model.DiscriminatorA().cuda() # Domain Discriminator

	#generator = torch.load('saved_models_happycn/generator.pt')
	#discriminatorA = torch.load('saved_models_happycn/discriminatora.pt')
	#discriminatorD = torch.load('saved_models_happycn/discriminatord.pt')

	dataFeeder = data.domainTransferLoader(datadir)
	train_loader = torch.utils.data.DataLoader(dataFeeder, batch_size=1, shuffle=True,
											   num_workers=1, pin_memory=True)

	criterion = nn.BCEWithLogitsLoss().cuda()

	optimizerD = torch.optim.Adam(discriminatorD.parameters(), lr=0.002) 
	optimizerA = torch.optim.Adam(discriminatorA.parameters(), lr=0.002)
	optimizerG = torch.optim.Adam(generator.parameters(), lr=0.002)

	generator.train()
	discriminatorD.train()
	discriminatorA.train()
	os.system("rm samples/*")	
	
	for epoch in range(epochs):
		os.system("rm testfile.txt")
		for i, (image1, image2, image3) in enumerate(train_loader):

			I1_var = image1.to(torch.float32).cuda() #Image of cloth being worn by model in image3
			I2_var = image2.to(torch.float32).cuda() #Image of cloth unassociated with model in image3
			I3_var = image3.to(torch.float32).cuda() #Image of Model
			
			real_label_var = torch.ones((I1_var.shape[0],1), requires_grad=False).cuda()
			fake_label_var = torch.zeros((I1_var.shape[0],1), requires_grad=False).cuda()			

			# ----------
			# Train DiscriminatorD
			# ----------
			
			#optimizerD.zero_grad()

			out_associated = discriminatorD(I1_var)
			lossD_real_1 = criterion(out_associated, real_label_var)

			out_not_associated = discriminatorD(I2_var)
			lossD_real_2 = criterion(out_not_associated, real_label_var)

			fake = generator(I3_var).detach()
			out_fake = discriminatorD(fake)
			lossD_fake = criterion(out_fake, fake_label_var)

			lossD = (lossD_real_1 + lossD_real_2 + lossD_fake)/3

			#lossD.backward()
			#optimizerD.step()
			
			# ----------
			# Train DiscriminatorA
			# ----------
			
			#optimizerA.zero_grad()

			associated_pair_var = torch.cat((I3_var, I1_var),1)
			not_associated_pair_var = torch.cat((I3_var, I2_var),1)

			fake = generator(I3_var).detach()
			fake_pair_var = torch.cat((I3_var, fake),1)

			out_associated = discriminatorA(associated_pair_var)
			lossA_ass = criterion(out_associated, real_label_var)

			out_not_associated = discriminatorA(not_associated_pair_var)
			lossA_not_ass = criterion(out_not_associated, fake_label_var)

			out_fake = discriminatorA(fake_pair_var)
			lossA_fake = criterion(out_fake, fake_label_var)

			lossA = (lossA_ass + lossA_not_ass + lossA_fake)/3

			#lossA.backward()
			#optimizerA.step()

			# ----------
			# Train Generator
			# ----------
			
			#optimizerG.zero_grad()
			
			fake = generator(I3_var)
			outputD = discriminatorD(fake)
			lossGD = criterion(outputD,real_label_var)

			fake_pair_var = torch.cat((I3_var, fake),1)
			outputA = discriminatorA(fake_pair_var)
			lossGA = criterion(outputA,real_label_var)

			lossG = (lossGD + lossGA)/2
			check = 0
			check = get_lips(lossG.item(), lossD.item(), lossA.item())

			if check == 1:
				optimizerG.zero_grad()
				lossG.backward()
				optimizerG.step()

				optimizerD.zero_grad()
				lossD.backward()
				optimizerD.step()

				optimizerA.zero_grad()
				lossA.backward()
				optimizerA.step()
			

			if((i+1) % 100) == 0:
				print("Iter:", i+1, "/", len(train_loader))
				print("LossG:", lossG.item(), "LossD:", lossD.item(), "LossA:", lossA.item())
			if((epoch+1) % 500) == 0:
                                #print("saving file")
                                output_pair = torch.cat((I1_var, I2_var, I3_var, fake),3)
                                torchvision.utils.save_image((output_pair+1)/2, 'samples/'+str(i+1)+'.jpg')
                                torch.save(generator,'models/generator.pt')
                                torch.save(discriminatorA,'models/discriminatora.pt')
                                torch.save(discriminatorD,'models/discriminatord.pt')


if __name__ == '__main__':
	os.system('mkdir -p samples')
        os.system('mkdir -p models')
	train_val(sys.argv[1], int(sys.argv[2]), sys.argv[3])















