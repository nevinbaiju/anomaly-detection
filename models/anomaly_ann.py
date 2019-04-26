import torch
from scipy.io import loadmat
import torch.nn as nn

def conv_dict(weights_mat):
    """
    Function to convert weights from mat format to dictionary with arrays.

    Parameters
    ----------
    weights_mat :dict
                 Dictionary of the weights in mat format.

    Returns
    -------
    dict        :dict
                 Dictionary of the weights with weights as list.
    """
    i = 0
    dict = {}
    for i in range(len(weights_mat)):
        if str(i) in weights_mat:
            if weights_mat[str(i)].shape == (0, 0):
                dict[str(i)] = weights_mat[str(i)]
            else:
                weights = weights_mat[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict

def get_weight(weight_path):
    """
    Function to convert the weights and biases of original implementation to Torch Tensors

    Parameters
    ----------
    weight_path :str
                 Path of the weights file.

    Returns
    -------
    weights     :dict
                 Dictionary of the tensors of Weights and Biases referenced by the layer.
    """
    dict2 = loadmat(weight_path)
    weights = conv_dict(dict2)

    weights['0'][0] = torch.transpose(torch.tensor(weights['0'][0]), 0, -1)
    weights['0'][1] = torch.tensor(weights['0'][1])

    weights['2'][0] = torch.transpose(torch.tensor(weights['2'][0]), 0, -1)
    weights['2'][1] = torch.tensor(weights['2'][1])

    weights['4'][0] = torch.transpose(torch.tensor(weights['4'][0]), 0, -1)
    weights['4'][1] = torch.tensor(weights['4'][1])
    return weights

class anomaly_ann(nn.Module):
    """
    The anomaly detection network which takes in features from the fc6 layer
    of a C3D network and predicts the anomaly score as described in [1].
    """
    def __init__(self, weights_path='../weights/weights_L1L2.mat', no_sigmoid=False):
        super(anomaly_ann, self).__init__()

        weights = get_weight(weights_path)
        self.no_sigmoid = no_sigmoid

        self.layer1 = nn.Linear(4096, 512)
        self.layer1.weight.data = weights['0'][0]
        self.layer1.bias.data = weights['0'][1]

        self.layer2 = nn.Linear(512, 32)
        self.layer2.weight.data = weights['2'][0]
        self.layer2.bias.data = weights['2'][1]

        self.layer3 = nn.Linear(32, 1)
        self.layer3.weight.data = weights['4'][0]
        self.layer3.bias.data = weights['4'][1]

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)

        out = self.layer2(out)

        out = self.layer3(out)
        if(self.no_sigmoid):
            return out
        return self.sigmoid(out)

if __name__ == '__main__':
    net = anomaly_ann()
    print(net)
    print(net(torch.rand(4096)))

"""
References
----------
[1] Waqas Sultani, Chen Chen, Mubarak Shah, "Real-world Anomaly Detection in Surveillance Videos".
The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
"""
