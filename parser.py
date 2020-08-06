import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--train_file", default="train.txt", help="File with list of videos for training")
parser.add_argument("--validation_file", default="validate.txt", help="File with list of videos for validation")
parser.add_argument("--test_file", default="test.txt", help="File with list of videos for testing")

parser.add_argument("--shifted_grids", default=False, help="True if using model with shifted grids")
parser.add_argument("--gaussian_blur_size", default=3, help="Size of Gaussian blur")
parser.add_argument("--N", default=5, help="Size of shifted grids")

parser.add_argument("--learning_rate", default=0.01, help="Learning rate for training")
parser.add_argument("--num_epochs", default=10, help="Number of epochs to train")
