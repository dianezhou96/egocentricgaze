# Egocentric Gaze Prediction

### Two convolutional neural networks to learn to predict gaze in egocentric videos based on saliency maps:
- `saliency.py` - Regression model that outputs a saliency map where higher value means more likely to gaze at that position. For training, the target is a saliency map with a Gaussian blur around true gaze location, and the loss function is MSE.
- `saliency_shifted_grids.py` - Classification model with shifted grids that outputs 5 saliency maps shfited in different ways. For training, the targets are 5 class labels representing the true gaze location, and the loss function is cross entropy.

### To train the neural networks:
- Put data in `data` folder (see "Expected data format" below).
- Put the list of names of videos to train on in a file called `train.txt` in the root directory.
- To train the regression model:
  `python train.py --gaussian_blur_size <blur size for the target>`
- To train the classification model with shifted grids:
  `python train.py --shifted_grids True --N <size of grid>`
- Optional parameters (see `parser.py` for default values)
  - `learning_rate`
  - `num_epochs`
  - `batch_size`
  - `model` - Specify model file to start training from a previous model, else model will train from random initialization

### To evaluate a model:
- To get the error of a model on training set:
  `python evaluate.py [--gaussian_blur_size <blur size for the target>] [--shifted_grids True --N <size of grid>] --model <path to model file>`
- To get the AUROC and average distance between predicted and actual gaze locations of a model on validation set:
  1. Put the list of names of videos to evaluate on in a file called `validate.txt` in the root directory.
  2. Run `python validate.py [--gaussian_blur_size <blur size for the target>] [--shifted_grids True --N <size of grid>] --model <path to model file>`

### Expected data format:
1. Data should go in a folder named `data` at the root directory.
2. For each video, create a directory with the video name.
3. Within each video directory, the raw video file should be named `world.mp4` and the gaze positions csv file should be named `gaze_positions.csv`.
4. Resize the videos to be 227 x 227 pixels and name the resized videos `world_resized.mp4`. You can do so by running the script `resize_videos.sh` in the "data" folder.

Data comes from egocentric video recorded by Pupil glasses. Data files are exported from Pupil Player.
