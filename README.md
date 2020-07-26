# Egocentric Gaze Prediction

### Two neural networks to learn to predict gaze in egocentric videos:
- saliency.py - Outputs a saliency map where higher value means more likely to gaze at that position. The target is a saliency map of the same size (Gaussian blur around true gaze location), and the loss function is MSE.
- saliency_shifted_grids.py - Outputs 5 saliency maps shfited in different ways. The targets are 5 class labels representing the true gaze location, and the loss function is cross entropy.

### To train the neural networks:
- Put data in "data" folder (see "Expected data format" below).
- Edit the training scripts to include the videos for training.
  - train.py trains the network defined in saliency.py
  - train_shifted_grids.py trains the network defined in saliency_shifted_grids.py

### Expected data format:
1. Data should go in a folder named "data" at the root directory.
2. For each video, create a directory with the video name.
3. Within each video directory, the raw video file should be named "video.mp4" and the gaze positions csv file should be named "gaze_positions.csv".
Data comes from egocentric video recorded by Pupil glasses. Data files are exported from Pupil Player.
