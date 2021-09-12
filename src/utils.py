import numpy as np

def normalize_mask(mask):
	"""
		c_mask = c_mask.float().cpu()
		c_mask /= c_mask.max()
		c_mask = c_mask.numpy()
		
		s_mask = s_mask.float().cpu()
		s_mask /= s_mask.max()
		s_mask = s_mask.numpy()
	"""
	return mask
	
def tensor_to_array(t):
	return np.array(t.cpu().squeeze(0).transpose(0, 1).transpose(1, 2)*255, dtype=np.uint8)