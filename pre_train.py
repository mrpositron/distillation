# Load PyTorch Framework
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import random_split, DataLoader

from tqdm import tqdm

import argparse
import os
from pathlib import Path


import copy



import wandb

def run_train(loader_dict, model, epochs, device, path2save, wb = False, val_epoch = 200):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = 0.001)

	best_model = []
	max_val_acc = .0

	for epoch in range(epochs):	
		cum_loss = {'train': .0, 'val': 0}
		correct = {'train': 0, 'val': 0}
		total = {'train': 0, 'val':0}

		for mode in ['train', 'val']:
			# no point of checking validation each time
			# will run inference on validation set only when epoch > val_epoch
			if mode == 'val' and epoch <= val_epoch:
				continue

			if mode == 'train':
				model.train()
			else:
				model.eval()

			loader = loader_dict[mode]
			for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
				batch_size = inputs.shape[0]
				inputs, targets = inputs.to(device), targets.to(device)
				if mode == 'train':
					optimizer.zero_grad()
				# adding torch.no_grad() speeds up inference a bit
				if mode == 'train':
					outputs = model(inputs)
				else:
					with torch.no_grad():
						outputs = model(inputs)
				loss = criterion(outputs, targets)
				if mode == 'train':
					loss.backward()
					optimizer.step()

				cum_loss[mode] += loss.item()
				_, predicted = outputs.max(1)
				total[mode] += targets.size(0)
				correct[mode] += predicted.eq(targets).sum().item()
			if wb:
				loss_logging = mode + "/loss"
				acc_logging = mode + "/acc"
				wandb.log({loss_logging: 1e3 * cum_loss[mode]/total[mode], acc_logging: correct[mode]/total[mode]}) 
				if mode == 'train':
					epoch_logging = mode + "/epoch"
					wandb.log({epoch_logging: epoch})
			print('Mode: %s | Epoch: %d/%d| Loss: %.3f | Acc: %.3f ' 
				% (mode, epoch+1, epochs, 1e3 * cum_loss[mode]/total[mode], 1e2 *  correct[mode]/total[mode]))


			# save the best model
			if mode == 'val' and (correct[mode]/total[mode]) > max_val_acc:
				temp_model = copy.deepcopy(model)
				max_val_acc = correct[mode]/total[mode]
				if best_model:
					best_model.pop()
				best_model.append((temp_model, max_val_acc))

	if epochs <= val_epoch:
		return
	model2save, acc = best_model.pop()
	print(f'Saved model has accuracy {acc}')
	torch.save(model2save, path2save)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pre-Training')
	parser.add_argument('--seed', default = 0, type = int)
	parser.add_argument('--path2save', default= 'model0', type = str )
	parser.add_argument('--wb', default = False, type = bool)
	parser.add_argument('--gpu', default = 0, type = int)
	parser.add_argument('--num_epochs', default = 300, type = int)
	parser.add_argument('--val_epoch', default=-1, type = int)
	args = parser.parse_args()

	torch.manual_seed(args.seed)

	# Assert that folder exists
	assert ".pt" in args.path2save or ".pth" in args.path2save, "The saved model should have following extensions: .pt ot .pth"
	path = Path(args.path2save)
	assert os.path.exists(path.parent), "The folder under which you want to save the weights does not exist"
	# Define the device
	device = torch.device("cuda:" + str(args.gpu))
	# Get the data
	dataset_transforms = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	# size of dataset is 50000
	dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
		download=True, transform=dataset_transforms)
	# Split the data to training and validation data
	# in the proportion 90/10
	# in other words, there are 45000 images in trainig data and 5000 images in the validation data
	train_size = int(0.9 * len(dataset))
	val_size = len(dataset) - train_size
	train_data, val_data = random_split(dataset, [train_size, val_size], generator = torch.Generator().manual_seed(42))

	train_loader = DataLoader(train_data, batch_size = 128, shuffle = True, 
		num_workers = 8, pin_memory = True, generator = torch.Generator().manual_seed(42))
	val_loader = DataLoader(val_data, batch_size = 128, shuffle = False, 
		num_workers = 8, pin_memory = True, generator = torch.Generator().manual_seed(42) )

	
	loader_dict = {'train': train_loader, 
				   'val': val_loader}
	# we should not load the model pretrained on ImageNet
	# model should learn from the randomly initialized weights
	model = models.resnet18()
	# we should cahnge the fully connected layer
	model.fc = nn.Sequential(
		nn.Linear(512, 10),
		nn.Softmax(dim = 0),
	)
	# send it to the device
	model = model.to(device)
	# initialize wandb session if is needed 
	if args.wb:
		# name the project whatever you want
		wandb.init(project="cifar10")
		wandb.config.update(args)
	# start the training
	run_train(loader_dict, model, epochs= args.num_epochs, device= device, path2save = args.path2save, wb = args.wb, val_epoch = args.val_epoch)


