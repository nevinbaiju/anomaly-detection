# Anomaly Detection

### [Link to paper](http://ijaecs.iraj.in//paper_detail.php?paper_id=18269&nameA_Comprehensive_Framework_for_Road_Accident_Detection_and_Response_using_Intelligent_Visual_Surveillance)

The projects implements the work of \[1\] [Real-world Anomaly Detection in Surveillance Videos](https://arxiv.org/abs/1801.04264) in pytorch. The project implemented and covered the following things:
- Modifying the \[2\] C3D implementation of pytorch by @DavideA as the feature extractor in the anomaly detection pipeline.
- Converted the weights of original implementation of the anomaly detection ANN to pytorch weights.
- Trained the anomaly detection ANN from the C3D feature extractor in pytorch.
- Modified \[3\] 3D Resnets for feature extraction in the pipeline.
- Training scripts for various 3D Resnet features.

The project was able to study the advantages and shortcomings of the novel approach for tackling weakly labelled data for anomaly detection. This project was completed with very short time and with beginner level understanding of concepts, so if there is any corrections, please let me know in the issues.

The original implementation of the codes I had used to learn are listed below.

- Original implementation of [\[1\]](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018)
- Tensorflow implementation of [\[1\]](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018) by @vantage-vision-vv.
- Pytorch implementation of [\[2\]](https://github.com/DavideA/c3d-pytorch) by @DavideA.
- Original implementation of [\[3\]](https://github.com/kenshohara/3D-ResNets-PyTorch)

### Setting up

1. Set up the libraries as per requirements.txt.
2. Download the C3D weights (Sports 1m) from [here](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle).
3. Download the weights for Anomaly detection framework from [here](https://raw.githubusercontent.com/WaqasSultani/AnomalyDetectionCVPR2018/master/weights_L1L2.mat).

### Script details
- detector.py - Script for reading a video and predicting its anomaly scores.
- detector_laptop.py - Script for reading the C3D features of a video and predicting the scores, this is suitable for demonstration on a hardware limited pc.
- gen_features_c3d.py - Script for reading videos from a folder and generating its c3d fetures.
- gen_features_resnet.py - Script for reading videos from a folder and generating its ResNet fetures according to the type of resnet chosen.
- plot_scores_folder.py - Script for reading the video files in a folder and plot the anomaly scores for the videos.
- The scripts in utils/ contains various scripts for reading video, generating segments, database acccess, alert by mail, etc.

Further details about the scripts can be found in the docstrings of various functions.
detector.py and detector_laptop.py contains various cmd line arguments. Please refer the script for more details.

### References

- [1] Waqas Sultani, Chen Chen, Mubarak Shah, "Real-world Anomaly Detection in Surveillance Videos".
The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- [2] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
Proceedings of the IEEE international conference on computer vision. 2015.
- [3]  Kensho Hara, Hirokatsu Kataoka, Yutaka Satoh. "Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?"
The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 6546-6555 

