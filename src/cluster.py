from skimage.segmentation import slic
import numpy as np
import hdbscan
from src.utils import renumerate_mask, reshape_arr
import torch

def run_superpixel(image, mask, scale, n_segments=None):
	if (n_segments is None):
		n_segments = image.shape[0]//2
	segment_slic = slic(image, n_segments=n_segments)[::scale, ::scale]
	
	w, h = mask.shape
	mask_flat = mask.reshape(-1).clone()
	segment_slic_flat = segment_slic.reshape(-1)
	
	for segment in range(len(np.unique(segment_slic_flat))):
		seg_mask = segment_slic_flat  == segment
		mask_flat[seg_mask] = mask_flat[seg_mask].median()
		
	mask_superpixeled = mask_flat.reshape(w, h)
	return mask_superpixeled

def run_clustering(image, min_cluster_size, device, hdbscan_scale_factor = 4):
	# Feel free to change hdbscan_scale_factor, it mainly affects computational time
	image_scaled = image[::hdbscan_scale_factor, ::hdbscan_scale_factor]
	w, h, c = image_scaled.shape
	image_flat = image_scaled.reshape(w * h, -1)

	# Run HDBSCAN clustering procedure
	clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(image_flat)
	mask = clusters.labels_.reshape(w, h)
	mask = renumerate_mask(mask).to(device)
	
	# Run superpixelization
	mask_superpixeled = run_superpixel(image, mask, hdbscan_scale_factor)
	mask_superpixeled = reshape_arr(mask_superpixeled[None, :, :], (w*hdbscan_scale_factor, h*hdbscan_scale_factor))

	return mask_superpixeled