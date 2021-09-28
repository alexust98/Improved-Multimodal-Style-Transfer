import numpy as np
import torch
from torch.nn.functional import interpolate

def renumerate_mask(mask):
	if (isinstance(mask, np.ndarray)):
		mask = torch.from_numpy(mask)

	labels = torch.unique(mask)
	mask_renum = mask.clone()
	for ind, label in enumerate(labels):
		mask_renum[mask == label] = ind
	return mask_renum
	
def fit_shape(arr, target_shape, renumerate=False):
	arr = reshape_arr(arr, target_shape)
	if (renumerate):
		arr = renumerate_mask(arr)
	return arr

def reshape_arr(arr, target_size):
	if (isinstance(arr, np.ndarray)):
		arr = torch.from_numpy(arr).float()
		
	arr = interpolate(
				arr.unsqueeze(1).float(),
				size=target_size,
				mode="nearest",
			).squeeze()
	return arr
	
def normalize_arr(arr):
	arr = arr.float().cpu().numpy()
	arr -= arr.min()
	arr /= arr.max()

	return np.array(arr*255, dtype=np.uint8)