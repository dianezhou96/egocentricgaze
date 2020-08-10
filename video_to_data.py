import cv2
import mmcv
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms



# multiple videos
class GazeFrameDataset(IterableDataset):
	"""
	An dataset to iterate through frames of video with gaze data
	data_path: folder where the videos are located
	videos_list: list of names of videos to include
	transform: transform to apply to data
	"""

	def __init__(self, data_path, videos_list, transform=None, shuffle=False):

		self.data_path = data_path
		self.transform = transform

		self.video_readers = [self.get_video_reader(video_name) for video_name in videos_list]
		self.gaze_dfs = [self.get_gaze_positions(video_name) for video_name in videos_list]

		self.video_gaze_tuples = []
		for i, gaze_df in enumerate(self.gaze_dfs):
			self.video_gaze_tuples.extend([(i, j) for j in range(len(gaze_df))])
		if shuffle:
			random.shuffle(self.video_gaze_tuples)

		self.video_gaze_idx = 0

	def __iter__(self):
		return self

	def __next__(self):

		# Find next frame
		# Sometimes last frames are empty so need to set new video
		found_frame = False
		while not found_frame:

			if self.video_gaze_idx == len(self.video_gaze_tuples):
				raise StopIteration

			video_idx, gaze_idx = self.video_gaze_tuples[self.video_gaze_idx]
			video = self.video_readers[video_idx]
			gaze_df = self.gaze_dfs[video_idx]
			frame_idx = gaze_df.loc[gaze_idx, 'world_index']
			frame = video[frame_idx]

			self.video_gaze_idx += 1
			if frame is not None:
				found_frame = True

		# Extract gaze info for frame
		frame = frame.transpose(1, 0, 2) # make shape H x W
		gaze_position = (gaze_df.loc[gaze_idx, 'norm_pos_x'], 
						 gaze_df.loc[gaze_idx, 'norm_pos_y'])
		sample = (frame, gaze_position)

		if self.transform:
			sample = self.transform(sample)

		return sample

	def get_video_reader(self, video_name):
		path = self.data_path + video_name + '/world.mp4'
		video = mmcv.VideoReader(path)
		return video

	def get_gaze_positions(self, video_name):
		path = self.data_path + video_name + '/gaze_positions.csv'
		df = pd.read_csv(path)
		# Keep only one gaze position per frame
		df_unique = df.groupby('world_index', group_keys=False).apply(lambda df: df.sample(1))
		# Keep only relevant columns
		df_gaze = df_unique[['world_index', 'norm_pos_x', 'norm_pos_y']].reset_index(drop=True)
		return df_gaze

class SetSize(object):
	"""
	Transform object to set frame to desired size and create saliency map from gaze location
	"""

	def __init__(self, frame_size, map_size, gaussian_blur_size, class_size=None):
		self.frame_size = frame_size
		self.map_size = map_size
		self.gaussian_blur_size = gaussian_blur_size
		self.class_size = class_size

	def __call__(self, sample):
		frame, gaze_position = sample

		# Resize frame
		resized_frame = cv2.resize(frame, dsize=self.frame_size)

		# Create target
		gaze_norm_x, gaze_norm_y = gaze_position
		get_abs_pos = lambda x, upper: int(max(0, min(x * upper, upper-1)))
		# Saliency map with gaussian blur
		if self.gaussian_blur_size:
			height, width = self.map_size
			gaze_y = get_abs_pos(gaze_norm_y, height)
			gaze_x = get_abs_pos(gaze_norm_x, width)
			target = np.zeros((height, width))
			target[gaze_y, gaze_x] = 1
			target = cv2.GaussianBlur(target, self.gaussian_blur_size, 0)
			target = target.flatten()	
		# Class label
		elif self.class_size:
			height, width = self.class_size
			gaze_y = get_abs_pos(gaze_norm_y, height)
			gaze_x = get_abs_pos(gaze_norm_x, width)
			target = gaze_y * width + gaze_x
		# Normalized coordinates
		else:
			target = gaze_position

		return resized_frame, target

class ToTensor(object):
	"""
	Transform object to set frame and saliency map to tensor type
	"""

	def __call__(self, sample):
		frame, target = sample

		# numpy image: H x W x C
		# torch image: C x H x W
		frame = frame.transpose((2, 0, 1)) / 255 # 0 to 1 instead of 0 to 255
		return (torch.from_numpy(frame).float(), torch.from_numpy(target).float())

class SetSizeShiftedGrids(object):
	"""
	Transform object to set frame to desired size and 
	create target classes from gaze location for shifted grids method
	"""

	def __init__(self, frame_size, N, class_size=None):
		self.frame_size = frame_size
		self.N = N
		self.class_size = class_size

		shift = 1 / (2 * N)
		self.shifted_grids = [
			(0, 0),
			(-shift, 0),
			(shift, 0),
			(0, -shift),
			(0, shift)
		]

	def __call__(self, sample):
		frame, gaze_position = sample

		# Resize frame
		resized_frame = cv2.resize(frame, dsize=self.frame_size)

		# Create target saliency map with shifted grids
		targets = []
		gaze_norm_x, gaze_norm_y = gaze_position
		get_abs_pos = lambda x, upper: int(max(0, min(x * upper, upper-1)))
		for i in range(len(self.shifted_grids)):
			x_shift, y_shift = self.shifted_grids[i]
			gaze_norm_x_shifted = gaze_norm_x + x_shift
			gaze_norm_y_shifted = gaze_norm_y + y_shift
			if not self.class_size:
				gaze_y = get_abs_pos(gaze_norm_y_shifted, self.N)
				gaze_x = get_abs_pos(gaze_norm_x_shifted, self.N)
				target = gaze_y * self.N + gaze_x
			else:
				height, width = self.class_size
				gaze_y = get_abs_pos(gaze_norm_y, height)
				gaze_x = get_abs_pos(gaze_norm_x, width)
				target = gaze_y * width + gaze_x
			targets.append(target)

		return resized_frame, targets

class ToTensorShiftedGrids(object):
	"""
	Transform object to set frame to tensor type
	"""

	def __call__(self, sample):
		frame, targets = sample

		# numpy image: H x W x C
		# torch image: C x H x W
		frame = frame.transpose((2, 0, 1)) / 255 # 0 to 1 instead of 0 to 255
		frame = torch.from_numpy(frame).float()
		return (frame, targets)

def make_transform(gaussian_blur_size=(3,3), class_size=None):
    size_transform = SetSize((227,227), (13,13), gaussian_blur_size, class_size)
    tensor_transform = ToTensor()
    transform = transforms.Compose([size_transform, tensor_transform])
    return transform

def make_transform_shifted_grids(N=5, class_size=None):
    size_transform = SetSizeShiftedGrids((227,227), N, class_size)
    tensor_transform = ToTensorShiftedGrids()
    transform = transforms.Compose([size_transform, tensor_transform])
    return transform

def get_videos_list_from_file(filename):
    with open(filename, 'r') as f:
        videos_list = f.read().split("\n")
    return videos_list



if __name__ == '__main__':
	# print("Dummy test...")
	# frames = get_frames('world')
	# print("Number of frames:", len(frames))
	# gaze_positions = get_gaze_positions('gaze_positions')
	# print(gaze_positions)

	# size_transform = SetSize((227,227), (13,13), (3,3))
	# tensor_transform = ToTensor()
	# transform = transforms.Compose([size_transform, tensor_transform])
	# dataset = GazeFrameDataset("./data/", "2020-03-15_19-27-56-f2472745", transform=transform)
	# print(dataset[0][0].shape)
	# print(dataset[0][1].shape)
	# torch.utils.data.DataLoader(dataset)

	# data_path = "./data/"
	# videos_list = ["2020-03-15_19-27-56-f2472745", "2020-06-22_11-14-22-319eaf00", 
	# 			   "2020-06-25_17-25-16_alexl_everyday-tyingshoelaces-189703d3"]
	# dataset = GazeFrameDataset(data_path, videos_list, transform=transform)
	# for i, sample in enumerate(dataset):
	# 	if i % 100 == 99:
	# 		print(i+1)
	# torch.utils.data.DataLoader(dataset)

	print("Shifted grids test...")

	size_transform = SetSizeShiftedGrids((227,227), 5)
	tensor_transform = ToTensorShiftedGrids()
	transform = transforms.Compose([size_transform, tensor_transform])
	# dataset = GazeFrameDataset("./data/", "2020-03-15_19-27-56-f2472745", transform=transform)
	# print(dataset[0][0].shape)
	# print(dataset[0][1].shape)
	# torch.utils.data.DataLoader(dataset)

	data_path = "./data/"
	videos_list = ["2020-03-15_19-27-56-f2472745", "2020-06-22_11-14-22-319eaf00", 
				   "2020-06-25_17-25-16_alexl_everyday-tyingshoelaces-189703d3"]
	dataset = GazeFrameDataset(data_path, videos_list, transform=transform, shuffle=True)
	# for i, sample in enumerate(dataset):
	# 	if i % 100 == 99:
	# 		print(i+1)
	dataloader = torch.utils.data.DataLoader(dataset)
	for i, data in enumerate(dataloader, 0):
		if i % 100 == 99:
			print(i+1)


	print("Done!")

