# Load PyTorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import random_split, DataLoader

from tqdm import tqdm

import argparse

import copy


import wandb

def run(loader_dict, model, epochs, device, folder2save, wb = False, val_epoch = 200):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = 0.001)

	best_model = []
	min_loss = float('inf')

	for epoch in range(epochs):	
		cum_loss = .0
		correct = 0
		total = 0

		for mode in ['train', 'val']:
			# no point of checking validation each time
			if mode == 'val' and epoch <= val_epoch:
				continue

			if mode == 'train':
				model.train()
			else:
				model.eval()

			loader = loader_dict[mode]
			for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
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

				cum_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()
			if wb:
				loss_logging = mode + "/loss"
				acc_logging = mode + "/acc"
				wandb.log({loss_logging: cum_loss, acc_logging: correct/total}) 
				if mode == 'train':
					epoch_logging = mode + "/epoch"
					wandb.log({epoch_logging: epoch})
			print('Mode: %s | Epoch: %d/%d| Batch: %d/%d | Loss: %.3f | Acc: %.3f ' % (mode, epoch+1, epochs, batch_idx+1, len(loader), cum_loss, correct/total))


			# save the best model
			if mode == 'val' and cum_loss < min_loss:
				temp_model = copy.deepcopy(model)
				min_loss = cum_loss
				if best_model:
					best_model.pop()
				best_model.append((temp_model, correct/total))

	if epoch <= val_epoch:
		return
	model2save, acc = best_model.pop()
	path2save = './' + folder2save + '/best_model.pt'
	print(f'Saved model has accuracy {acc}')
	torch.save(model2save, path2save)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pre-Training')
	parser.add_argument('--seed', default = 0, type = int)
	parser.add_argument('--folder2save', default= 'model0', type = str )
	parser.add_argument('--wb', default = False, type = bool)
	parser.add_argument('--gpu', default = 0, type = int)
	parser.add_argument('--num_epochs', default = 300, type = int)
	args = parser.parse_args()

	torch.manual_seed(args.seed)

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
	# in the proportion 9 to 1
	train_size = int(0.9 * len(dataset))
	val_size = len(dataset) - train_size
	train_data, val_data = random_split(dataset, [train_size, val_size], generator = torch.Generator().manual_seed(0))

	train_loader = DataLoader(train_data, batch_size = 64, shuffle = True, num_workers = 8)
	val_loader = DataLoader(val_data, batch_size = 64, shuffle = False, num_workers = 8 )

	loader_dict = {'train': train_loader, 
				   'val': val_loader}
	# we should not load the model pretrained on ImageNet
	# model should learn from the randomly initialized weights
	model = models.resnet18()

	model.fc = nn.Sequential(
		nn.Linear(512, 10),
		nn.Softmax(dim = 0),
	)

	model = model.to(device)
	# initialize wandb session if is needed 
	if args.wb:
		wandb.init(project="cifar10")
		wandb.config.update(args)
	# start the training
	run(loader_dict, model, epochs= args.num_epochs, device= device, folder2save = args.folder2save, wb = args.wb)

		
    
