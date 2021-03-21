from tqdm import tqdm
import sys
from utils import *

import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

@torch.no_grad()
def run(loader, device, models_list):
	num_classes = 10
	correct = [0 for _ in range(len(models_list) + 1)]
	total = 0
	for _, (inputs, labels) in enumerate(tqdm(loader)):
		targets = torch.zeros((inputs.size(0), num_classes))
		inputs, labels, targets = inputs.to(device), labels.to(device), targets.to(device)
		for i,model in enumerate(models_list):
			model.to(device)
			outputs = model(inputs).detach()
			targets += F.softmax(outputs, dim = 1)
			model.cpu()
			_, predicted = outputs.max(1)
			correct[i] += predicted.eq(labels).sum().item()
		targets = targets.div(len(models_list))
		_, predicted = targets.max(1)
		total += inputs.size(0)
		correct[-1] += predicted.eq(labels).sum().item()

	old_train = [correct[i]/total for i in range(len(models_list))]

	print("Accuracies of all models: ", old_train)
	print("The accuracy of an ensemble model: ", 1e2 * correct[-1]/total)

if __name__ == "__main__":
	seed_everything(42)
	device = torch.device("cuda:0")
	weightsPath = []
	for i in range(1, len(sys.argv)):
		weightsPath.append(sys.argv[i])
	models_list = []
	for path in weightsPath:
		model = models.resnet18()
		model.fc = nn.Linear(512, 10)
		model = torch.load(path, map_location = torch.device("cpu"))
		model.eval()
		models_list.append(model)

	dataset_transforms = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
		download=True, transform=dataset_transforms)
	loader = DataLoader(dataset, batch_size = 128, shuffle = False, 
		num_workers = 16, pin_memory = True, generator = torch.Generator().manual_seed(42))
	run(loader, device, models_list)