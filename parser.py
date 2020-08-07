import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--train_file", default="train.txt", help="File with list of videos for training")
parser.add_argument("--validation_file", default="validate.txt", help="File with list of videos for validation")
parser.add_argument("--test_file", default="test.txt", help="File with list of videos for testing")

parser.add_argument("--shifted_grids", type=bool, default=False, help="True if using model with shifted grids")
parser.add_argument("--gaussian_blur_size", type=int, default=3, help="Size of Gaussian blur")
parser.add_argument("--N", default=5, type=int, help="Size of shifted grids")

parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

parser.add_argument("--model", default="model_blur_3_lr_0.01_epochs_5_batch_32_08-06-2020_15-45.tar", 
	help="Path to model file")

