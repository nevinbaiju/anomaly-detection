### Manual for running the modified project for feature extraction.

- input file stores the name(s) of the video(s) used for processing
  videos folder stores the input video(s)

- output.json stores the feature vectors

- Features are being generated for a set of 16 frames in the video sequence

Command to extract features in this machine :

```python main.py --input ./input --video_root /home/ubuntu/resnet/3DResnet/videos --output ./output.json --model_name --model_depth --model path-to-weight --mode feature```
