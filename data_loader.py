import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import random_split, DataLoader


if __name__ == "__main__":
    dataset_transforms = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	# size of dataset is 50000
	dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
		download=True, transform=dataset_transforms)
	# Split the data to training and validation data
	# in the proportion 9 to 1
	train_size = int(0.9 * len(dataset))
	val_size = len(dataset) - train_size
	train_data, val_data = random_split(dataset, [train_size, val_size], generator = torch.Generator().manual_seed(0))