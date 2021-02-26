# Load PyTorch Framework
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import random_split, DataLoader

from tqdm import tqdm 
import copy
import argparse
import os
from pathlib import Path

import wandb


def run(loader_dict, new_model, target_model, epochs, device,
	path2save, wb = False, val_epoch = -1):

	stack2save = []
	max_val_acc = .0

	mse_criterion = nn.MSELoss()
	ce_criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(new_model.parameters(), lr = 0.001)

	target_model.eval()

	for epoch in range(epochs):
		# mse loss
		mse_cum_loss = {'train': .0, 'val': .0}
		# cross entropy loss
		ce_cum_loss  ={'train': .0, 'val': .0}
		correct = {'train': 0, 'val': 0}
		total = {'train': 0, 'val': 0}

		for mode in ['train', 'val']:
			if mode == 'val' and epoch < val_epoch:
				continue

			if mode == 'train':
				new_model.train()
			else:
				new_model.eval()

			loader = loader_dict[mode]
			for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
				batch_size = inputs.shape[0]
				inputs, targets = inputs.to(device), targets.to(device)
				if mode == 'train':
					optimizer.zero_grad()
				
				if mode == 'train':
					outputs = new_model(inputs)
				else:
					with torch.no_grad():
						outputs = new_model(inputs)

				with torch.no_grad():
					y = target_model(inputs)
				# compute mse loss
				mse_loss = mse_criterion(outputs, y)

				if mode == 'train':
					mse_loss.backward()
					optimizer.step()
				# compute cross entropy loss
				ce_loss = ce_criterion(outputs, targets)

				mse_cum_loss[mode] += mse_loss.item()
				ce_cum_loss[mode] += ce_loss.item()

				_, predicted = outputs.max(1)
				total[mode] += targets.size(0)
				correct[mode] += predicted.eq(targets).sum().item()
				#break
		
			print('Mode: %s | Epoch: %d/%d| MSE Loss: %.6f | Cross Entropy Loss %.3f | Acc: %.3f ' 
				% (mode, epoch+1, epochs, 1e6 *  (mse_cum_loss[mode] / total[mode]), 
					1e3 * (ce_cum_loss[mode] / total[mode]), (correct[mode]/total[mode]) ))
			# wandb logging

			if wb:
				mse_loss_log = mode + "/mse_loss"
				ce_loss_log = mode + "/ce_loss"
				acc_log = mode + "/acc"
				wandb.log({
					mse_loss_log: 1e6 *  (mse_cum_loss[mode] / total[mode]),
					ce_loss_log: 1e3 * (ce_cum_loss[mode] / total[mode]),
					acc_log: 1e2 * (correct[mode]/total[mode]),
				})
				if mode == "train":
					epoch_log = mode + "/epoch"
					wandb.log({epoch_log: epoch})
			
			# save the best model
			if mode == 'val' and (correct[mode]/total[mode]) > max_val_acc:
				temp_model = copy.deepcopy(new_model)
				max_val_acc = correct[mode]/total[mode]
				if stack2save:
					stack2save.pop()
				stack2save.append((temp_model, max_val_acc))
	# save the best model
	if epochs <= val_epoch:
		return
	model2save, acc = stack2save.pop()
	torch.save(model2save, path2save)
	print(f'Saved model has accuracy {acc}')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Self-Distillation')
	parser.add_argument('--seed', default = 42, type = int)
	parser.add_argument('--path2load', default= '', type = str )
	parser.add_argument('--path2save', default= '', type = str )
	parser.add_argument('--wb', default = False, type = bool)
	parser.add_argument('--gpu', default = 0, type = int)
	parser.add_argument('--num_epochs', default = 1, type = int)
	args = parser.parse_args()

	assert os.path.exists(args.path2load), "The pass to pre-trained model does not exist"
	assert ".pt" in args.path2load or ".pth" in args.path2load, "The pre-trained model is not of .pt ot .pth format"
	assert ".pt" in args.path2save or ".pth" in args.path2save, "The saved model should have following extensions: .pt ot .pth"
	path = Path(args.path2save)
	assert os.path.exists(path.parent), "The folder under which you want to save the weights does not exist"

	# For reproducibility
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	#torch.backends.cudnn.deterministic=True
	# Initialize wandb session if necessary
	if args.wb:
		wandb.init(project="cifar10_self_distill")
		wandb.config.update(args)
	# Set the device
	device = torch.device("cuda:" + str(args.gpu))
	# Define the target model
	target_model = models.resnet18()
	target_model.fc = nn.Sequential(
		nn.Linear(512, 10),
		nn.Softmax(dim = 0),
	)
	## Load the weights to target_model
	target_model = torch.load(args.path2load)
	target_model.to(device)
	## Freeze all the weights
	for params in target_model.parameters():
		params.requires_grad = False


	
	# Get the data
	dataset_transforms = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	## Size of dataset is 50000
	dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
		download=True, transform=dataset_transforms)
	## Split the data to training and validation data
	## in the proportion 90/10, as during pretraining
	train_size = int(0.9 * len(dataset))
	val_size = len(dataset) - train_size
	train_data, val_data = random_split(dataset, [train_size, val_size], generator = torch.Generator().manual_seed(42))

	train_loader = DataLoader(train_data, batch_size = 128, shuffle = True, 
		num_workers = 8, pin_memory = True, generator = torch.Generator().manual_seed(42))
	val_loader = DataLoader(val_data, batch_size = 128, shuffle = False, 
		num_workers = 8, pin_memory = True, generator = torch.Generator().manual_seed(42) )

	
	loader_dict = {'train': train_loader, 
		'val': val_loader}
	# Define the new model
	new_model = models.resnet18()
	new_model.fc = nn.Sequential(
		nn.Linear(512, 10),
		nn.Softmax(dim = 0),
	)
	new_model.to(device)
	# Run the training procedure
	run(loader_dict, new_model = new_model, target_model= target_model, epochs= args.num_epochs, device= device, 
		path2save = args.path2save, wb = args.wb, val_epoch = -1)