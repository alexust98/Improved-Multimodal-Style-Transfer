from skimage.segmentation import slic
import hdbscan
from src.utils import renumerate_mask

def run_superpixel(labels, img):
	segment_slic = slic(img, n_segments=2000, compactness=5, sigma=1)[::4, ::4]
	w, h = labels.shape
	labels_ = labels.reshape(-1)
	segment_slic_ = segment_slic.reshape(-1)
	for segment in range(len(np.unique(segment_slic_))):
		mask = segment_slic_  == segment
		labels_[mask] = int(np.median(labels_[mask]))
	labels_superpixeled = np.array(labels_.reshape(w, h), dtype=int)
	return labels_superpixeled, segment_slic

def run_clustering(image,
		   min_cluster_size,
		   device):
	image = image[::4, ::4]
	w, h, c = image.shape
	image_flat = image.reshape(w * h, -1)

	# Run HDBSCAN clustering procedure
	clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(image_flat)
	mask = clusters.labels_.reshape(w, h)
	
	mask = renumerate_mask(mask).to(device)
	#labels_filtered, segments = superpixel(labels.copy(), img)
	#labels = torch.from_numpy(labels)
	return mask