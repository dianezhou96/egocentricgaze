import cv2
import mmcv
import numpy as np
import pandas as pd
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

	def __init__(self, data_path, videos_list, transform=None):

		self.data_path = data_path
		self.videos_list = videos_list
		self.transform = transform

		self.video_idx = -1
		self.gaze_idx = 0
		self.set_new_video()

	def __iter__(self):
		return self

	def __next__(self):

		# end of video; set new video
		if self.gaze_idx == len(self.df_gaze.index):
			self.set_new_video()

		# Find next frame
		# Sometimes last frames are empty so need to set new video
		found_frame = False
		while not found_frame:
			frame_idx = self.df_gaze.loc[self.gaze_idx, 'world_index']
			frame = self.video[frame_idx]
			if frame is None:
				self.set_new_video()
			else:
				found_frame = True

		# Extract gaze info for frame
		frame = frame.transpose(1, 0, 2) # make shape H x W
		gaze_position = (self.df_gaze.loc[self.gaze_idx, 'norm_pos_x'], 
						 self.df_gaze.loc[self.gaze_idx, 'norm_pos_y'])
		sample = (frame, gaze_position)

		if self.transform:
			sample = self.transform(sample)

		self.gaze_idx += 1

		return sample

	def set_new_video(self):

		if self.video_idx < len(self.videos_list) - 1:
			self.video_idx += 1
		else:
			raise StopIteration

		video_name = self.videos_list[self.video_idx]
		self.video = self.get_frames(video_name)
		self.df_gaze = self.get_gaze_positions(video_name)
		self.gaze_idx = 0

		print(video_name)

	def get_frames(self, video_name):
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

	def __init__(self, frame_size, map_size, gaussian_blur_size):
		self.frame_size = frame_size
		self.map_size = map_size
		self.gaussian_blur_size = gaussian_blur_size

	def __call__(self, sample):
		frame, gaze_position = sample

		# Resize frame
		resized_frame = cv2.resize(frame, dsize=self.frame_size)

		# Create target saliency map
		gaze_norm_x, gaze_norm_y = gaze_position
		height, width = self.map_size
		get_abs_pos = lambda x, upper: int(max(0, min(x * upper, upper-1)))
		gaze_y = get_abs_pos(gaze_norm_y, height)
		gaze_x = get_abs_pos(gaze_norm_x, width)
		target = np.zeros((height, width))
		target[gaze_y, gaze_x] = 1
		target = cv2.GaussianBlur(target, self.gaussian_blur_size, 0)

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
		target = np.expand_dims(target, 0)
		return (torch.from_numpy(frame).float(), torch.from_numpy(target).float())

class SetSizeShiftedGrids(object):
	"""
	Transform object to set frame to desired size and 
	create target classes from gaze location for shifted grids method
	"""

	def __init__(self, frame_size, N):
		self.frame_size = frame_size
		self.N = N

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
		for i in range(len(self.shifted_grids)):
			x_shift, y_shift = self.shifted_grids[i]
			gaze_norm_x_shifted = gaze_norm_x + x_shift
			gaze_norm_y_shifted = gaze_norm_y + y_shift
			get_abs_pos = lambda x, upper: int(max(0, min(x * upper, upper-1)))
			gaze_y = get_abs_pos(gaze_norm_y_shifted, self.N)
			gaze_x = get_abs_pos(gaze_norm_x_shifted, self.N)
			target = gaze_y * self.N + gaze_x
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
	dataset = GazeFrameDataset(data_path, videos_list, transform=transform)
	for i, sample in enumerate(dataset):
		if i % 100 == 99:
			print(i+1)
	torch.utils.data.DataLoader(dataset)


	print("Done!")

