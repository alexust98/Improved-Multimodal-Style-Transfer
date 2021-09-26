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
from src.utils import normalize_mask, tensor_to_array

def main():
	parser = argparse.ArgumentParser(description='Improved-Multimodal-Style-Transfer')
	parser.add_argument('--content', '-c', type=str, default=None,
						help='Content image path e.g. content.jpg.')
	parser.add_argument('--style', '-s', type=str, default=None,
						help='Style image path e.g. image.jpg.')
	parser.add_argument('--output_name', '-o', type=str, default="results/result.jpg",
						help='Output path for generated image, no need to add ext, e.g. out.')
	parser.add_argument('--imsize', type=int, default=512,
						help='Size of content and style images to be scaled to. E.g. 512x512. Non-square shapes are not supported yet.')
	parser.add_argument('--WCT_alpha', '-alpha', default=1.0,
						help='WCT procedure content/style fusion proportion, e.g. 1.0 means full stylization, 0.0 would return content image.')
	parser.add_argument('--save_masks', type=bool, default=False,
						help='Boolean flag whether save segmentation/clustering masks or not.')
	parser.add_argument('--gpu', type=int, default=0,
						help='GPU device id.')
	parser.add_argument('--model_path', type=str, default='model/model_state.pth',
						help='pretrained model path.')

	args = parser.parse_args()

	# Set device on GPU if available, else CPU
	if (torch.cuda.is_available()) and (args.gpu >= 0):
		device = torch.device(f'cuda:{args.gpu}')
		print(f'Working on CUDA: {torch.cuda.get_device_name(0)}\n')
	else:
		device = 'cpu'
		print("Working on CPU\n")
	
	model = IMST(alpha=args.WCT_alpha, device=device)
	model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))
	model = model.to(device)
	
	c = Image.open(args.content).convert('RGB')
	s = Image.open(args.style).convert('RGB')
	c = np.array(transforms.Resize((args.imsize, args.imsize))(c))
	s = np.array(transforms.Resize((args.imsize, args.imsize))(s))

	# Run ST
	with torch.no_grad():
		t_start = time()
		out, c_mask, s_mask = model.run_ST(c, s)
		t_end = time()
		print("ST time: ", t_end-t_start)
		if (args.save_masks):
			c_mask = normalize_mask(c_mask)
			s_mask = normalize_mask(s_mask)
			
			ind = args.output_name.rfind(".")
			imsave(args.output_name[:ind] + "_c_mask" + args.output_name[ind:], c_mask, quality=100)
			imsave(args.output_name[:ind] + "_s_mask" + args.output_name[ind:], s_mask, quality=100)
		

	out = tensor_to_array(out)
	imsave(args.output_name, out, quality=100)
	return;
	
if __name__ == '__main__':
	main()