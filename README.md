# Improved-Multimodal-Style-Transfer
Cluster-based style transfer approach, that combines content image segmentation and style image clustering techniques via enhanced matching procedure to generate realistic stylizations.

Command to run IMST: <br>
<hr>
-- content: Content image path e.g. contents/content.jpg <br>
-- style: Style image path e.g. styles/style.jpg <br>
-- output_name: Output path for generated image, e.g. out.jpg <br>
-- imsize: Size of content and style images to be scaled to, e.g. 512x512 <br>
-- WCT_alpha: WCT procedure content/style fusion proportion, e.g. 1.0 means full stylization, 0.0 would return content image <br>
-- save_masks: Boolean flag whether save segmentation/clustering masks or not <br>
-- randomize_matching: Applies random shuffling to matching map <br>
-- HDBSCAN_cluster_size: HDBSCAN cluster size hyperparameter, e.g 1500 <br>
-- gpu: GPU device id. -1 means cpu <br>
-- model_path: Pretrained model path (encoder + decoder)
<hr>
Example: >> python main.py --content "contents/content3.jpg" --style "styles/style3.jpg" --imsize 512 --save_masks True --HDBSCAN_cluster_size 1000 <br>

Stylization samples: <br>
<img src="https://user-images.githubusercontent.com/45120679/135097024-04e01d53-6f87-4f9f-b11f-855b11ae6f5d.jpg" width="600"><br>
<img src="https://user-images.githubusercontent.com/45120679/135097313-3c359985-f0de-4849-b4dc-ae4d26278bd3.jpg" width="600"><br>
<img src="https://user-images.githubusercontent.com/45120679/135097482-9424ef63-b996-441c-95eb-2fd09fb438ab.jpg" width="600"><br>
