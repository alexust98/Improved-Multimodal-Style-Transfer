import os
import argparse
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.io import imsave
from time import time

from src.model import IMST
from src.utils import normalize_arr
import matplotlib.pyplot as plt

def main():
	parser = argparse.ArgumentParser(description='Improved-Multimodal-Style-Transfer')
	parser.add_argument('--content', '-c', type=str, default=None,
						help='Content image path e.g. contents/content.jpg.')
	parser.add_argument('--style', '-s', type=str, default=None,
						help='Style image path e.g. styles/style.jpg.')
	parser.add_argument('--output_name', '-o', type=str, default="results/result.jpg",
						help='Output path for generated image, e.g. out.jpg.')
	parser.add_argument('--imsize', type=int, default=512,
						help='Size of content and style images to be scaled to, e.g. 512x512. Non-square shapes are not supported yet.')
	parser.add_argument('--WCT_alpha', '-alpha', type=float, default=1.0,
						help='WCT procedure content/style fusion proportion, e.g. 1.0 means full stylization, 0.0 would return content image.')
	parser.add_argument('--save_masks', type=bool, default=False,
						help='Boolean flag whether save segmentation/clustering masks or not.')
	parser.add_argument('--randomize_matching', type=bool, default=False,
						help='Applies random shuffling to matching map.')
	parser.add_argument('--HDBSCAN_cluster_size', "-cluster_size", type=int, default=1500,
							help='HDBSCAN cluster size hyperparameter, e.g 1500.')
	parser.add_argument('--gpu', type=int, default=0,
						help='GPU device id. -1 means cpu.')
	parser.add_argument('--model_path', type=str, default='model/model_state.pth',
						help='Pretrained model path (encoder + decoder).')

	args = parser.parse_args()

	# Set device on GPU if available, else CPU
	if (torch.cuda.is_available()) and (args.gpu >= 0):
		device = torch.device(f'cuda:{args.gpu}')
		print(f'Working on CUDA: {torch.cuda.get_device_name(0)}\n')
	else:
		device = 'cpu'
		print("Working on CPU\n")
	
	model = IMST(hdbscan_cluster_size=args.HDBSCAN_cluster_size, alpha=args.WCT_alpha, device=device)
	model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))
	model = model.to(device)
	
	c = Image.open(args.content).convert('RGB')
	s = Image.open(args.style).convert('RGB')
	c = np.array(transforms.Resize((args.imsize, args.imsize))(c))
	s = np.array(transforms.Resize((args.imsize, args.imsize))(s))

	# Run ST
	with torch.no_grad():
		t_start = time()
		out, c_mask, s_mask = model.run_ST(c, s, args.randomize_matching)
		t_end = time()
		print("ST time: ", t_end-t_start)
		if (args.save_masks):
			c_mask = normalize_arr(c_mask)
			s_mask = normalize_arr(s_mask)
			
			ind = args.output_name.rfind(".")
			imsave(args.output_name[:ind] + "_c_mask" + args.output_name[ind:], c_mask, quality=100)
			imsave(args.output_name[:ind] + "_s_mask" + args.output_name[ind:], s_mask, quality=100)
		
	clip=False # Change to True if you want to clip decoder output
	if (clip):
		out = normalize_arr(out.clip(min=0.0, max=1.0).squeeze(0).transpose(0, 1).transpose(1, 2))
	else:
		out = normalize_arr(out.squeeze(0).transpose(0, 1).transpose(1, 2))
		
	imsave(args.output_name, out, quality=100)
	"""
	f, ax = plt.subplots(1, 3, figsize=(18, 6))
	ax[0].imshow(c)
	ax[0].axis('off')
	ax[1].imshow(s)
	ax[1].axis('off')
	ax[2].imshow(out)
	ax[2].axis('off')
	plt.tight_layout()
	plt.savefig('git_readme.jpg')
	"""
	return;
	
if __name__ == '__main__':
	main()