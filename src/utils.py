import numpy as np
import torch

def renumerate_mask(mask):
	if (isinstance(mask, np.ndarray)):
		mask = torch.from_numpy(mask)

	labels = torch.unique(mask)
	mask_renum = mask.clone()
	for ind, label in enumerate(labels):
		mask_renum[mask == label] = ind
	return mask_renum

def normalize_mask(mask):
	mask = mask.float().cpu().numpy()
	mask -= mask.min()
	mask /= mask.max()

	return np.array(mask*255, dtype=np.uint8)
	
def tensor_to_array(t):
	return np.array(t.cpu().squeeze(0).transpose(0, 1).transpose(1, 2)*255, dtype=np.uint8)