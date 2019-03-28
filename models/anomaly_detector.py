import torch

from C3D_features import C3D_features
from anomaly_ann import anomaly_ann

class anomaly_detector():
    """
	This class combines the C3D feature extracter and the anomaly detection
    ANN and provides a framework for prediction.
	"""
    def __init__(self):
        print("Loading C3D weights...")
        self.c3d = C3D_features('../weights/c3d.pickle')
        print("Done!")
        print("Loading anomaly ANN weights...")
        self.ann = anomaly_ann('../weights/weights_L1L2.mat')
        print("Done!")
    def predict(self, block):
        features = self.c3d(block)
        score = self.ann(features).item()
        del features
        return score

if __name__ == '__main__':
    detector = anomaly_detector()
    print(detector.predict(torch.rand((1, 3, 16, 112, 112))))
