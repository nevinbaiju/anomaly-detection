from .C3D_model import C3D

import torch
import torch.nn as nn

class C3D_features(nn.Module):
	"""
	The C3D network as described in [1] with features extracted from fc6.
	Pytorch implementation by DavideA at https://github.com/DavideA/c3d-pytorch.
	"""

	def __init__(self, weights_path="../weights/c3d.pickle"):
		super(C3D_features, self).__init__()

		layers = self.load_layers(weights_path)

		self.conv1 = layers[0]
		self.pool1 = layers[1]

		self.conv2 = layers[2]
		self.pool2 = layers[3]

		self.conv3a = layers[4]
		self.conv3b = layers[5]
		self.pool3 = layers[6]

		self.conv4a = layers[7]
		self.conv4b = layers[8]
		self.pool4 = layers[9]

		self.conv5a = layers[10]
		self.conv5b = layers[11]
		self.pool5 = layers[12]
		self.fc6 = layers[13]

		self.relu = nn.ReLU()

	def load_layers(self, weights):
		model = C3D()
		model.load_state_dict(torch.load(weights))
		C3D_CNN_LIST = list(model.children())[:-5]

		return C3D_CNN_LIST


	def forward(self, x):

		h = self.relu(self.conv1(x))
		h = self.pool1(h)

		h = self.relu(self.conv2(h))
		h = self.pool2(h)

		h = self.relu(self.conv3a(h))
		h = self.relu(self.conv3b(h))
		h = self.pool3(h)

		h = self.relu(self.conv4a(h))
		h = self.relu(self.conv4b(h))
		h = self.pool4(h)

		h = self.relu(self.conv5a(h))
		h = self.relu(self.conv5b(h))
		h = self.pool5(h)

		h = h.view(-1, 8192)
		h = self.relu(self.fc6(h))

		return h

if __name__ == '__main__':
	model = C3D_features()
	print(model(torch.rand((1, 3, 32, 112, 112))).shape)

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""