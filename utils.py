import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, Dataset

import random

def get_loader_dict(batch_size = 128, num_workers = 8, loader_seed = 42, train_val_ratio = 0.9):
	# load the dataset 
	dataset_transforms = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
		download=True, transform=dataset_transforms)
	# Split the data to training and validation data
	# in the proportion 90/10
	# in other words, there are 45000 images in trainig data and 5000 images in the validation data
	train_size = int(train_val_ratio * len(dataset))
	val_size = len(dataset) - train_size
	train_data, val_data = random_split(dataset, [train_size, val_size], generator = torch.Generator().manual_seed(loader_seed))

	train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, 
		num_workers = num_workers, pin_memory = True, generator = torch.Generator().manual_seed(loader_seed))
	val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, 
		num_workers = num_workers, pin_memory = True, generator = torch.Generator().manual_seed(loader_seed) )

	return {'train': train_loader, 'val': val_loader}, train_size, val_size

def seed_everything(seed_val):
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)
	random.seed(seed_val)

# kindly taken from https://datascience.stackexchange.com/a/55964
def categorical_cross_entropy(y_pred, y_true):
	y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
	return -(y_true * torch.log(y_pred)).sum(dim=1).mean()