import torch
from torch.nn.functional import interpolate
from torchvision import transforms
from src.utils import renumerate_mask, reshape_arr

def segmentator_preprocess(image):
	preproc = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	return preproc(image).unsqueeze(0)
	
def merge_masks(main_mask, add_mask):
	assert main_mask.shape == add_mask.shape, "Segmentation masks shapes are not coherent."
	
	# Check if add_mask has detected something
	merged_mask = main_mask.clone()
	if (add_mask.min() != add_mask.max()):
		merged_mask[add_mask.bool()] = main_mask.max() + 1
		
	return merged_mask
	
def run_segmentation(image, device):
	# Extract sky mask
	sky_mask = run_midas(image, device=device)
	
	# Load segmentator
	segmentator = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True).to(device)
	segmentator.eval()
	
	image_preproc = segmentator_preprocess(image).to(device)
	prediction = segmentator(image_preproc)['out'][0].argmax(0)
	
	# Renumerate labels to 0, 1, 2, ...
	seg_mask = renumerate_mask(prediction)
	mask = merge_masks(seg_mask, sky_mask)
	
	return mask
	
def run_midas(image, device, threshold=300.0):
	# Load MiDaS
	midas = torch.hub.load("intel-isl/MiDaS", "MiDaS").to(device)
	midas.eval();

	# Load MiDaS transform
	midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform
	image_trans = midas_transform(image).to(device)

	# Run MiDaS sky detection
	with torch.no_grad():
		prediction = midas(image_trans)
		prediction = reshape_arr(prediction, image.shape[:2])
		sky_mask = (prediction < threshold)*1.0

	return sky_mask