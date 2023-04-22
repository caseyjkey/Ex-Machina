import torch
import os
import argparse

from torchvision.utils import save_image
from model.model import pSp, condi
from model.DNAnet import DNAnet
from utils.utils import align_face, totensor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import ElasticNet
from sklearn.cluster import KMeans

import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt


def main(args):
	device = torch.device(args.device)

	# Load model weights
	DNA = DNAnet().to(device)
	DNA.load_state_dict(torch.load(args.dna), strict=True)
	DNA.eval()

	net = nn.DataParallel(pSp(3, args.enc, args.gan)).to(device)
	net.eval()

	mapper = nn.DataParallel(condi()).to(device)
	mapper.load_state_dict(torch.load(args.map), strict=True)
	mapper.eval()

	# Inference
	mImg = align_face(args.mom_path).convert('RGB')
	dImg = align_face(args.dad_path).convert('RGB')

	testAge =  torch.ones((1,1)) * (args.target_age) /100 
	if args.target_gender == 'male':
		testGen = torch.ones(1, 1).to(device)
	else:
		testGen = torch.zeros(1, 1).to(device)
	"""
	    - What effects do different parts of the latent code have on different parts of the reconstructed image:
        - mask diffrent parts of the latent vector and generate and look at results and observe effects
        - given results of above suggest and try diffrent ways of combining. Look at KinStyle paper, they did something similar
        - maybe try pca of latent vectors like StyleDNA baseline
		project or interpolate of 2 parents 
	"""
	with torch.no_grad():
		#print(mImg)
		#print(dImg)
		mImg = totensor(mImg).unsqueeze(0).to(device)
		dImg = totensor(dImg).unsqueeze(0).to(device)

		sW_hat = DNA(net.module.encoder(mImg), net.module.encoder(dImg))
		# Mask first half of  sw_hat 
		"""mask = torch.ones_like(sW_hat)
		mask[:, :sW_hat.shape[1]//2] = 1

		# Mask second half of  sw_hat 
		mask = torch.ones_like(sW_hat)
		mask[:, sW_hat.shape[1]//2] = 1
		
		# Randomly mask features
		mask = (torch.rand_like(sW_hat) < 0.5).float()
		
		# Use L1 regularization to determine most important features
		l1_loss = F.l1_loss(sW_hat, torch.zeros_like(sW_hat), reduction='none')
		l1_loss = l1_loss.mean(0)  # Take mean across batch
		sorted_indices = torch.argsort(l1_loss, descending=True)

		# Create binary mask where only the top k features are kept
		#print(sW_hat.shape)
		k = 50  # Number of features to keep
		mask = torch.zeros_like(sW_hat)
		mask[:, sorted_indices[:k]] = 1
		"""

		"""alpha = 0.1 
		l1_ratio = 0.5  # Ratio  L1/L2 
		n_samples, n_features = sW_hat.shape[0], sW_hat.shape[1]
		X = sW_hat.cpu().numpy()
		y = np.zeros(n_samples)
		enet = ElasticNet(alpha=alpha, max_iter=100, l1_ratio=l1_ratio, fit_intercept=False)
		enet.fit(X, y)
		coef = enet.coef_
		threshold = 0.1  
		mask = torch.from_numpy(np.abs(coef) > threshold).to(device).unsqueeze(0).repeat(n_samples, 1)"""
		sW_hat_freq = np.fft.fft(sW_hat)

		# Set high-frequency coefficients to zero
		cutoff_freq = 0.1 # adjust this as needed
		cutoff_idx = int(np.ceil(cutoff_freq * len(sW_hat_freq)))
		sW_hat_freq[cutoff_idx:-cutoff_idx] = 0

		# Inverse Fourier transform to get masked sW_hat
		masked_sW_hat = np.fft.ifft(sW_hat_freq).real

		#masked_sW_hat = sW_hat * mask
		#print("MASK VECTOR", masked_sW_hat)
		
		sW_hat_expand = masked_sW_hat.repeat(18, 1, 1).permute(1, 0, 2)
		sW_hat_delta = mapper(sW_hat_expand, testAge,  testGen)
		sImg_hat = net(sW_hat_expand + sW_hat_delta)

		save_image((sImg_hat+1)/2, f'{args.outputs}/result.png', nrow = 1)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with StyleDNA')

    # Image Path and Saving Path
    parser.add_argument('-m', '--mom_path', help='Path of mom image', default="styledna/result/assets/mom.png")
    parser.add_argument('-d', '--dad_path', help='Path of dad image', default="styledna/result/assets/dad.png")
    parser.add_argument('-a', '--target_gender', help='Predicted child age', default= 'male')
    parser.add_argument('-g', '--target_age', help='Predicted child gender', default = 15)
    parser.add_argument('-o', '--outputs', help='Output directory', default='./result')

    # Model weights Path
    parser.add_argument('-E', '--enc', help='Path of pretrained encoder model', default='./pretrained_model/enc_4_2.pth')
    parser.add_argument('-M', '--map', help='Path of pretrained mapper model', default='./pretrained_model/condi_4_2.pth')
    parser.add_argument('-G', '--gan', help='Path of pretrained stylegan2 model', default='./pretrained_model/stylegan2-ffhq-config-f.pt')
    parser.add_argument('-D', '--dna', help='Path of pretrained DNA model', default='./pretrained_model/10.pth')

    # Device
    parser.add_argument('--device', help='Device to be used by the model (default=cuda:0)', default="cuda")
    args = parser.parse_args()

    ### RUN
    main(args)



