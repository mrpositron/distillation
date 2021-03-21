# Load PyTorch Framework
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from utils import *

def run(loader_dict, new_model, target_model, epochs, device,
	path2save, wb = False, val_epoch = -1):

	stack2save = []
	max_val_acc = .0

	ce_criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(new_model.parameters(), lr = 0.001)
	# target_model should be in eval state
	target_model.eval()

	for epoch in range(epochs):
		# cross entropy loss between outputs and targets
		cum_ce_loss_1 = {'train': .0, 'val': .0}
		# cross entropy loss between outputs and labels
		cum_ce_loss_2  ={'train': .0, 'val': .0}
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
			for batch_idx, (inputs, labels) in enumerate(tqdm(loader)):
				batch_size = inputs.shape[0]
				inputs, labels = inputs.to(device), labels.to(device)
				if mode == 'train':
					optimizer.zero_grad()
				
				if mode == 'train':
					outputs = new_model(inputs)
				else:
					with torch.no_grad():
						outputs = new_model(inputs)

				with torch.no_grad():
					targets = target_model(inputs)
				# compute cross entropy loss between outputs and targets
				ce_loss_1 = categorical_cross_entropy(F.softmax(outputs, dim = 1), F.softmax(targets, dim = 1))

				if mode == 'train':
					ce_loss_1.backward()
					optimizer.step()
				# compute cross entropy loss between outputs and true labels
				ce_loss_2 = ce_criterion(outputs, labels)

				cum_ce_loss_1[mode] += ce_loss_1.item()
				cum_ce_loss_2[mode] += ce_loss_2.item()

				_, predicted = outputs.max(1)
				total[mode] += labels.size(0)
				correct[mode] += predicted.eq(labels).sum().item()
			
			print('Mode: %s | Epoch: %d/%d| Self Cross Entropy Loss: %.6f | Cross Entropy Loss %.3f | Acc: %.3f ' 
				% (mode, epoch+1, epochs, 1e3 *  (cum_ce_loss_1[mode] / total[mode]), 
					1e3 * (cum_ce_loss_2[mode] / total[mode]), (correct[mode]/total[mode]) ))
			# wandb logging

			if wb:
				log_cum_ce_loss_1 = mode + "/ce_loss_1"
				log_cum_ce_loss_2 = mode + "/ce_loss_2"
				log_acc = mode + "/acc"
				wandb.log({
					log_cum_ce_loss_1: 1e3 *  (cum_ce_loss_1[mode] / total[mode]),
					log_cum_ce_loss_2: 1e3 * (cum_ce_loss_2[mode] / total[mode]),
					log_acc: 1e2 * (correct[mode]/total[mode]),
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
	seed_everything(args.seed)
	
	# Initialize wandb session if necessary
	if args.wb:
		wandb.init(project="cifar10_sd")
		wandb.config.update(args)
	# Set the device
	device = torch.device("cuda:" + str(args.gpu))
	# Define the target model
	target_model = models.resnet18()
	target_model.fc = nn.Linear(512, 10)
	target_model = torch.load(args.path2load)
	target_model.to(device)
	## Freeze all the weights
	for params in target_model.parameters():
		params.requires_grad = False
	# Get loader dict
	loader_dict = get_loader_dict()[0]
	# Define the new model
	new_model = models.resnet18()
	new_model.fc = nn.Linear(512, 10)
	new_model.to(device)
	# Run the training procedure
	run(loader_dict, new_model = new_model, target_model= target_model, epochs= args.num_epochs, device= device, 
		path2save = args.path2save, wb = args.wb, val_epoch = -1)