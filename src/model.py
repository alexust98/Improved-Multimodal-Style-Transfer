import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from src.encoder import VGGEncoder
from src.decoder import Decoder
from src.utils import fit_shape

from src.segment import run_segmentation
from src.cluster import run_clustering
from src.matching import match_clusters

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
		
	def weighted_WCT(self, cf_cl, style_scores):
		cf_cl_init = cf_cl.clone()
		
		c_mean = torch.mean(cf_cl, 2)
		cf_cl = cf_cl - c_mean.unsqueeze(-1)
		
		c_cov = torch.bmm(cf_cl, cf_cl.transpose(1, 2)) / (cf_cl.shape[2] - 1)
		c_u, c_e, c_v = torch.svd(c_cov)
		eps = 1e-6
		c_d = (c_e+eps).pow(-0.5)

		whitened = torch.bmm(torch.bmm(c_v, torch.diag_embed(c_d)), (c_u.transpose(1, 2)))
		whitened = torch.bmm(whitened, cf_cl)

		s_cov = torch.zeros((self.b, self.c, self.c)).to(self.device)
		s_mean = torch.zeros((self.b, self.c, 1)).to(self.device)
		total_pixels = 0.0
		
		for style_ind, sf in enumerate(self.sf_per_label):
			sb, sc, swh = sf.shape
			s_mean += style_scores[style_ind]*torch.sum(sf, 2, keepdim=True)
			total_pixels += style_scores[style_ind]*swh
		s_mean /= total_pixels
		
		for style_ind, sf in enumerate(self.sf_per_label):
			sf = sf - s_mean
			s_cov += style_scores[style_ind]*torch.bmm(sf, sf.transpose(1, 2))
			
		s_cov /= (total_pixels - 1)
		s_u, s_e, s_v = torch.svd(s_cov)
		s_d = s_e.pow(0.5)
		
		colored = torch.bmm(torch.bmm(s_u, torch.diag_embed(s_d)), s_v.transpose(1, 2))
		colored = torch.bmm(colored, whitened)
		
		colored = colored + s_mean
		colored_feature = self.alpha * colored + (1 - self.alpha) * cf_cl_init
		return colored_feature
	
	def run_ST(self, content_image, style_image, randomize_matching=False):
		c_tensor = transforms.ToTensor()(content_image).unsqueeze(0).to(self.device)
		s_tensor = transforms.ToTensor()(style_image).unsqueeze(0).to(self.device)

		# Get content/style encoder feaures for WCT and Matching procedures
		_, cf_match, _, cf = self.vgg_encoder(c_tensor.to(self.device))
		_, sf_match, _, sf = self.vgg_encoder(s_tensor.to(self.device))
		self.b, self.c, self.w, self.h = cf.shape

		# Segment content image
		content_labels = run_segmentation(content_image, device=self.device)
		# Cluster style image
		style_labels = run_clustering(style_image, min_cluster_size=self.hdbscan_cluster_size, device=self.device)
		# Generate matching map between content segments and style clusters
		matching_map = match_clusters(cf_match, sf_match, content_labels, style_labels, (self.w, self.h))

		style_k = len(matching_map[0])
		if (randomize_matching):
			style_perm = torch.randperm(style_k)
			for content_ind in matching_map.keys():
					matching_map[content_ind] = matching_map[content_ind][style_perm]

		# Select style features for each style cluster
		self.sf_per_label = []
		# Fit masks to features shape
		cl_fit = fit_shape(content_labels.unsqueeze(0), (self.w, self.h), renumerate=True)
		sl_fit = fit_shape(style_labels.unsqueeze(0), (self.w, self.h), renumerate=True)
		
		for style_ind in range(style_k):
			sl_ind = (sl_fit == style_ind).unsqueeze(dim=0).expand_as(sf).bool()
			sf_sl = sf[sl_ind].reshape(sf.shape[0], sf.shape[1], -1)
			self.sf_per_label.append(sf_sl)

		cs_feature = cf.clone()
		for content_ind, style_scores in matching_map.items():
			cl_ind = (cl_fit == content_ind)
			cs_feature[:, :, cl_ind] = self.weighted_WCT(cf[:, :, cl_ind], style_scores)

		out = self.decoder(cs_feature)

		return out, content_labels, style_labels