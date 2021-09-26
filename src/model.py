import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from src.encoder import VGGEncoder
from src.decoder import Decoder

from src.segment import run_segmentation
from src.cluster import run_clustering

class IMST(nn.Module):
	def __init__(self, hdbscan_cluster_size, alpha, device):
		super().__init__()
		
		self.alpha = alpha
		self.device = device
		self.hdbscan_cluster_size = hdbscan_cluster_size

		self.vgg_encoder = VGGEncoder().to(device)
		self.decoder = Decoder(4).to(device)

	@staticmethod
	def calc_content_loss(out_features, content_features):
		return F.mse_loss(out_features, content_features)

	@staticmethod
	def calc_style_loss(out_middle_features, style_middle_features):
		loss = 0
		for c, s in zip(out_middle_features, style_middle_features):
			c_mean, c_std = calc_mean_std(c)
			s_mean, s_std = calc_mean_std(s)
			loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
		return loss

	def match_clusters(self, cf, sf, cl, sl):
		pass
	
	def run_ST(self, content_image, style_image, randomize=False):
		c_tensor = transforms.ToTensor()(content_image).unsqueeze(0).to(self.device)
		s_tensor = transforms.ToTensor()(style_image).unsqueeze(0).to(self.device)
		
		cs = []

		# Get content/style encoder feaures for WCT and Matching procedures
		_, cf_match, _, cf = self.vgg_encoder(c_tensor.to(self.device))
		_, sf_match, _, sf = self.vgg_encoder(s_tensor.to(self.device))

		# Segment content image
		content_labels = run_segmentation(content_image, device=self.device)
		
		# Cluster style image
		style_labels = run_clustering(style_image, min_cluster_size=self.hdbscan_cluster_size, device=self.device)

		"""
		content_k = int(content_label.max().item() + 1)
		style_k = int(style_label.max().item() + 1)
		start = time()
		match, match_full = self.match_clusters(cf_match[0, :, ::4, ::4], sf_match[0, :, ::4, ::4], content_label, style_label)
		end = time()
		print("Matching: ", end-start)
		#print("MATCH:\n")

		P = torch.randperm(style_k)
		for k in match.keys():
			if randomize:
				match[k] = match[k][P]
			#print(k, "||", match[k], "\n")
		content_label = content_label.to(self.device)
		style_label = style_label.to(self.device)

		cs_feature = torch.zeros_like(cf)
		#cs_feature_full = torch.zeros_like(cf)

		for i, j in match.items():
			cl = (content_label == i).unsqueeze(dim=0).expand_as(cf).to(torch.float)
			sub_sfs = []
			for jj, _ in enumerate(j):
				sl = (style_label == jj).unsqueeze(dim=0).expand_as(sf).to(torch.bool)
				sub_sf = sf[sl].reshape(sf.shape[0], -1)
				sub_sfs.append(sub_sf)
			cs_feature += labeled_whiten_and_color2(cf, sub_sfs, alpha, cl, j)

		cs.append(cs_feature.unsqueeze(dim=0))
		#cs_full.append(cs_feature_full.unsqueeze(dim=0))
			
		cs = torch.cat(cs, dim=0)
		#cs_full = torch.cat(cs_full, dim=0)
		out = self.decoder(cs).cpu()
		out_full = None#self.decoder(cs_full).cpu()
		"""

		return c_tensor, content_labels, style_labels