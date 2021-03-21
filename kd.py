from tqdm import tqdm
import copy
import wandb
from utils import *

import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim

def run(loader, device, models_list, new_model, wb = False, epochs = 100):
	# stack is needed to save the best model
	stack2save = []
	# max_val_acc keeps track of the current highest accuracy
	max_val_acc = .0
	ce_criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(new_model.parameters(), lr = 0.001)

	for epoch in range(epochs):
		print(f"Epoch {epoch+1}/{epochs}")
		# cross entropy loss between outputs and targets
		cum_ce_loss_1 = {'train': .0, 'val': .0}
		# cross entropy loss
		cum_ce_loss_2  ={'train': .0, 'val': .0}

		correct = {
			'train': [0 for _ in range(len(models_list) + 1)],
			'val': [0 for _ in range(len(models_list) + 1)]
		}
		total = {
			'train': 0,
			'val' :0,
		}
		for mode in ['train', 'val']:
			if mode == 'train':
				new_model.train()
			else:
				new_model.eval()
			for _, (inputs, labels) in enumerate(tqdm(loader[mode])):
				if mode == 'train':
					optimizer.zero_grad()
				targets = torch.zeros((inputs.size(0), 10))
				inputs, labels = inputs.to(device), labels.to(device)
				for i,model in enumerate(models_list):
					model.to(device)
					with torch.no_grad():
						outputs = model(inputs).detach().cpu()
					targets += F.softmax(outputs, dim = 1)
					model.cpu()
					_, predicted = outputs.max(1)
					correct[mode][i] += predicted.eq(labels.cpu()).sum().item()
				targets = targets.div(len(models_list))
				targets = targets.to(device)
				if mode == 'train':
					outputs = new_model(inputs)
				else:
					with torch.no_grad():
						outputs = new_model(inputs)
				ce_loss_1 = categorical_cross_entropy(F.softmax(outputs, dim = 1), targets)
				if mode == 'train':
					ce_loss_1.backward()
					optimizer.step()
				ce_loss_2 = ce_criterion(outputs.detach(), labels)
				cum_ce_loss_1[mode] += ce_loss_1.item()
				cum_ce_loss_2[mode] += ce_loss_2.item()
				_, predicted = outputs.max(1)
				total[mode] += inputs.size(0)
				correct[mode][-1] += predicted.eq(labels).sum().item()
			print(f'Epoch: {epoch+1}/{epochs} || Mode: {mode} || Accuracy : { 1e2 * correct[mode][-1]/total[mode]} || Loss 1: {1e3 * cum_ce_loss_1[mode] / total[mode]} || Loss 2: {1e3 * cum_ce_loss_2[mode]/ total[mode]}')
			if wb:
				log_cum_ce_loss_1 = mode + "/ce_loss_1"
				log_cum_ce_loss_2 = mode + "/ce_loss_2"
				log_acc = mode + "/acc"
				wandb.log({
					log_cum_ce_loss_1: 1e3 * (cum_ce_loss_1[mode] / total[mode]),
					log_cum_ce_loss_2: 1e3 * (cum_ce_loss_2[mode] / total[mode]),
					log_acc: 1e2 * (correct[mode][-1]/total[mode]),
				})
				if mode == "train":
					epoch_log = mode + "/epoch"
					wandb.log({epoch_log: epoch})
			if mode == 'val' and (correct[mode][-1]/total[mode]) > max_val_acc:
				temp_model = copy.deepcopy(new_model)
				max_val_acc = correct[mode][-1]/total[mode]
				if stack2save:
					stack2save.pop()
				stack2save.append((temp_model, max_val_acc))

		if epoch == 0:
			old_train = [correct['train'][i]/total['train'] for i in range(len(models_list))]
			old_val = [correct['val'][i]/total['val'] for i in range(len(models_list))]
			print(old_train,"\n", old_val)

	model2save, acc = stack2save.pop()
	torch.save(model2save, "kd1.pt")
	print(f'Saved model has accuracy {acc}')

if __name__ == "__main__":
	if True:
		wandb.init(project="cifar10_sd")
		#wandb.config.update(args)
	seed_everything(42)
	device = torch.device("cuda:0")
	new_model = models.resnet18()
	new_model.fc = nn.Linear(512, 10)
	new_model.to(device)

	weightsPath = ['model0.pt', 'model1.pt', 'model2.pt', 'model3.pt', 'model4.pt']
	models_list = []
	for path in weightsPath:
		model = models.resnet18()
		model.fc = nn.Linear(512, 10)
		model = torch.load(path, map_location = torch.device("cpu"))
		model.eval()
		models_list.append(model)
	loader = get_loader_dict()[0]
	run(loader, device, models_list, new_model, wb = True)