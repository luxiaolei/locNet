
import os
import skimage

import numpy as np

from scipy.misc import imread, imresize
from skimage import color

from numpy.random import randint



class InputProducer:
	def __init__(self, imgs_path, gt_path, live=False):
		"""

		"""
		self.imgs_path_list = [os.path.join(imgs_path, fn) for fn in sorted(os.listdir(imgs_path))]
		self.gts_list = self.gen_gts(gt_path)
		self.gen_img = self.get_image()

		self.roi_params = {
		'roi_size': 224, 
		'roi_scale': 3,
		'l_off': [0,0]
		}

	def get_image(self):
		idx = -1
		for img_path, gt in zip(self.imgs_path_list, self.gts_list):
			img = imread(img_path, mode='RGB')

			assert min(img.shape[:2]) >= 224

			# Gray to color. RES??
			#if len(img.shape) < 3:
			#img = skimage.color.gray2rgb(img)
			assert len(img.shape) == 3

			idx += 1
			if idx == 0: 
				self.first_gt = gt
				self.first_img = img
			yield img, gt, idx


	def gen_gts(self, gt_path):
		"""
		Each row in the ground-truth files represents the bounding box 
		of the target in that frame. (tl_x, tl_y, box-width, box-height)
		"""
		f = open(gt_path, 'r')
		lines = f.readlines()

		try:
			gts_list = [[int(p) for p in i[:-1].split(',')] 
			                   for i in lines]
		except Exception as e:
			gts_list = [[int(p) for p in i[:-1].split('\t')] 
			                   for i in lines]
		return gts_list


	def _random_place(self, img, gt, eadge, override_img=0):
		# scale ratio distortion
		tlx, tly, w, h = gt
		target = img[tly:tly+h, tlx:tlx+w, :]

		w_rd, h_rd = randint(w, 2*w), randint(h, 2*h)
		new_shape = [h_rd, w_rd]
		target_rz = imresize(target, new_shape)

		# put back to img with a random location
		tlx_rand = randint(1, eadge-w_rd)
		tly_rand = randint(1, eadge-h_rd)
		
		if not isinstance(override_img, np.ndarray) :
			# place generated target onto img
			img[tly:tly+h, tlx:tlx+w, :] = img.mean()
			img[tly_rand:tly_rand+h_rd, tlx_rand:tlx_rand+w_rd] = target_rz
			return img.astype(np.uint8)
		else:
			# place generated target onto img
			override_img[tly:tly+h, tlx:tlx+w, :] = override_img.mean()
			override_img[tly_rand:tly_rand+h_rd, tlx_rand:tlx_rand+w_rd] = target_rz
			return override_img.astype(np.uint8), [tlx_rand, tly_rand, w_rd, h_rd]

	def _gen_distorted_sample(self, img, gt):
		convas = np.zeros((max(img.shape), max(img.shape), 3))
		convas[:img.shape[0], :img.shape[1]] = img
		eadge = convas.shape[0]

		# randomly replace a box region
		for _ in range(50):
			w_rd, h_rd = randint(10, 20), randint(10, 20)
			tlx_rand = randint(1, eadge-w_rd)
			tly_rand = randint(1, eadge-h_rd)
			gt_rand = [tlx_rand, tly_rand, w_rd, h_rd]
			convas = self._random_place(convas, gt_rand, eadge)
		# randomly replace the target region with a random scale
		convas, loc_gen = self._random_place(img, gt, eadge, convas)

		# generate ground truth map
		tlx_gen, tly_gen, w_gen, h_gen = loc_gen
		gt_M = np.zeros(convas.shape[:2])
		gt_M[tly_gen:tly_gen+h_gen, tlx_gen:tlx_gen+w_gen] = 1

		return  imresize(convas,(224,224)), imresize(gt_M, (224,224)), loc_gen


	def gen_batches(self, img, gt, num_samples=100, batch_sz=10):
		input_batch, gt_M_batch, loc_batch = [], [], []
		for step in range(num_samples):
			img_distored, gt_M, loc = self._gen_distorted_sample(img, gt)
			input_batch += [img_distored]
			gt_M_batch += [gt_M[...,np.newaxis]]
			loc_batch += [loc]

		input_batch = [np.array(input_batch[i:i+batch_sz]) for i in range(num_samples) if i % batch_sz==0]
		gt_M_batch = [np.array(gt_M_batch[i:i+batch_sz]) for i in range(num_samples) if i % batch_sz==0]
		loc_batch = [np.array(loc_batch[i:i+batch_sz]) for i in range(num_samples) if i % batch_sz==0]
		return input_batch, gt_M_batch, loc_batch