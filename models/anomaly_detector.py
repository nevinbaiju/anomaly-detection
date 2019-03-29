import torch

from .C3D_features import C3D_features
from .anomaly_ann import anomaly_ann

class anomaly_detector():
    """
	This class combines the C3D feature extracter and the anomaly detection
    ANN and provides a framework for prediction.
	"""
    def __init__(self, weights_dict):
        """
        __init__ function.

        Parameters
        ----------
        weights_dict : dict
            Dictionary that holds the weights path of c3d and ann.
        """
        print("Loading C3D weights...")
        self.c3d = C3D_features(weights_dict['c3d'])
        print("Done!")
        print("Loading anomaly ANN weights...")
        self.ann = anomaly_ann(weights_dict['ann'])
        print("Done!")

    def predict(self, block):
        """
        Function that predicts the score for a block of frames.

        Parameters
        ----------
        block : torch.tensor
            Tensor of the block of the frames.
        """
        features = self.c3d(block)
        score = self.ann(features).item()
        del features
        return score

if __name__ == '__main__':
    print("Testing...")
    weights_dict = {'c3d' : '../weights/c3d.pickle', 'ann' : '../weights/weights_L1L2.mat'}
    detector = anomaly_detector(weights_dict)
    print(detector.predict(torch.rand((1, 3, 16, 112, 112))))
    print("Anomaly detector set!")
