import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

import sys
import os

from tqdm import tqdm

@torch.no_grad()
def run_test(model, loader, device):
	model.eval()
	correct, total = 0, 0
	for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
		inputs, targets = inputs.to(device), targets.to(device)
		outputs = model(inputs)
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
	return (correct/total)

if __name__ == "__main__":
	# pass arguments

	assert 2 <= len(sys.argv) <= 3, "Not enough arguments"
	assert os.path.exists(sys.argv[1]), "Path to load weights does not exist"

	path = sys.argv[1]
	gpu = sys.argv[2] if len(sys.argv) > 2 else 0
	device = torch.device("cuda:" + str(gpu))

	# data loaders
	dataset_transforms = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
		download=True, transform=dataset_transforms)
	loader = DataLoader(dataset, batch_size = 128, shuffle = False, 
		num_workers = 16, pin_memory = True)
	# define the model
	model = models.resnet18()
	model.fc = nn.Sequential(
		nn.Linear(512, 10),
		nn.Softmax(dim = 0),
	)
	# load the weights
	model = torch.load(path)
	model = model.to(device)
	# check the accuracy
	acc = run_test(model, loader, device)
	print(f'Accuracy is {100*acc}')