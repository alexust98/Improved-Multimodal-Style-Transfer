import torch
from src.utils import fit_shape

def calc_stats(features, labels):
	grams = []
	for k_cluster in range(0, int(labels.max().item() + 1)):
		arr = features[:, labels == k_cluster]
		grams.append(torch.mm(arr, arr.T)/arr.shape[1])

	return grams

def calc_dist(content_grams, style_grams):
	dist = torch.zeros((len(content_grams), len(style_grams)))
	
	for i in range(len(content_grams)):
		for j in range(len(style_grams)):
			dist[i, j] = torch.nn.functional.mse_loss(content_grams[i], style_grams[j])
	return dist
	
def calc_penalty(arr, mode="max"):
	assert mode in {"max", "mean"}, "Active penalty mode is not supported."
	
	if (mode == "mean"):
		arr_sorted = arr.sort().values
		penalty = (arr_sorted[:, 1:] - arr_sorted[:, :-1]).mean(dim=1)
	elif (mode == "max"):
		penalty = arr.amax(dim=1)
		
	return penalty

def softmax(vec, T):
	softmax_score = torch.exp(-T*(vec - vec.min()))
	softmax_score = softmax_score/softmax_score.sum()
	return softmax_score

def generate_style_scores(dist_style, T=None, target_max_score=0.85):
	if not(T is None):
		# Softmax with temperature T
		return softmax(dist_style, T)
		
	# Auto Temperature Tuning
	T = 0.0
	max_score = 0.0
	diff_criterion = 1.0
	while ((max_score <= target_max_score) and (diff_criterion > 1e-3)):
		T += 0.1
		style_scores = softmax(dist_style, T)
		max_score_new = style_scores.max()
		diff_criterion = torch.abs(max_score_new - max_score)
		max_score = max_score_new
		
	return style_scores
	
def match_clusters(cf, sf, cl, sl, target_shape):
	cf_fit = fit_shape(cf.squeeze(0), target_shape)
	sf_fit = fit_shape(sf.squeeze(0), target_shape)
	
	cl_fit = fit_shape(cl.unsqueeze(0), target_shape, renumerate=True)
	sl_fit = fit_shape(sl.unsqueeze(0), target_shape, renumerate=True)
	
	content_grams = calc_stats(cf_fit, cl_fit)
	style_grams = calc_stats(sf_fit, sl_fit)
		
	# Calculate L2-distance between content and style clusters along spatial dim
	dist = calc_dist(content_grams, style_grams)
	# Order content clusters by their distance to style clusters, the lower distance the higher priority
	content_priority_order = torch.argsort(dist.amin(dim=1)).tolist()
	
	W = {}
	penalty = calc_penalty(dist)
	for content_ind in content_priority_order:
		style_ind = int(torch.argmin(dist[content_ind, :]))
		style_scores = generate_style_scores(dist[content_ind, :])

		W[content_ind] = style_scores
		dist[:, style_ind] += penalty
		
	return W