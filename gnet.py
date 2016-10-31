
import tensorflow as tf
import numpy as np 


class GNet:

	def __init__(self, scope, conv_tensor):
		"""
		Base calss for SGNet, defines the network structure
		"""
		self.scope = scope
		self.params = {'wd': 0.05} # L2 regulization coefficien
		
		self.variables = []
		with tf.variable_scope(scope) as scope:
			self._build_graph(conv_tensor)
			self.gt_M = tf.placeholder(dtype=tf.float32, shape=(None,224,224,1),name='gt_M')


	def _build_graph(self, conv_tensor):
		"""
		Define Structure. 
		The first additional convolutional
		layer has convolutional kernels of size 9×9 and outputs
		36 feature maps as the input to the next layer. The second
		additional convolutional layer has kernels of size 5 × 5
		and outputs the foreground heat map of the input image.
		ReLU is chosen as the nonlinearity for these two layers.

		Args:
		    vgg_conv_shape: 
		Returns:
		    conv2: 
		"""
		self.variables = []
		self.kernel_weights = []

		
		self.input_maps = conv_tensor    
		with tf.name_scope('conv1') as scope:
			kernel = tf.Variable(tf.truncated_normal([9,9,512,36], dtype=tf.float32,stddev=1e-1), name='weights')

			conv = tf.nn.conv2d(self.input_maps, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[36], dtype=tf.float32), name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv1 = tf.nn.relu(out, name=scope)
			self.variables += [kernel, biases]
			self.kernel_weights += [kernel]


		with tf.name_scope('conv2') as scope:
			kernel = tf.Variable(tf.truncated_normal([5,5,36,1], dtype=tf.float32, stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(conv1, kernel , [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32), name='biases')
			self.pre_M = tf.nn.bias_add(conv, biases)
			self.variables += [kernel, biases]
			self.kernel_weights += [kernel]

			# Turn pre_M to a rank2 tensor within range 0-1.
			#pre_M = tf.squeeze(pre_M)
			self.pre_M /= tf.reduce_max(self.pre_M)
			self.out_layer = tf.nn.relu(self.pre_M)


	def loss(self):
		"""Returns Losses for the current network.

		Args:
		    gt_M: np.ndarry, ground truth heat map.
		Returns:
		    Loss: 
		"""

		with tf.name_scope(self.scope) as scope:
			beta = tf.constant(self.params['wd'], name='beta')
			loss_rms = tf.reduce_max(tf.squared_difference(self.gt_M, self.pre_M))
			loss_wd = [tf.reduce_mean(tf.square(w)) for w in self.kernel_weights]
			loss_wd = beta * tf.add_n(loss_wd)
			total_loss = loss_rms + loss_wd
		return total_loss




