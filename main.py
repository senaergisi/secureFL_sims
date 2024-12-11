#####Only train set and test set, no validation set######
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import copy
import numpy as np
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from torchvision.models import resnet18

import attacks
import defenses

torch.manual_seed(0)
np.random.seed(0)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

total_train_size = len(trainset)
total_test_size = len(testset)

classes = 10
num_clients = 10
rounds = 1000
batch_size = 200
epochs_per_client = 1
learning_rate = 0.1

total_train_size, total_test_size

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader(torch.utils.data.DataLoader):
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            for batch in self.dl:
                yield to_device(batch, self.device)

        def __len__(self):
            return len(self.dl)

device = get_device()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class FedResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(FedResNet18, self).__init__()
        
        # Load the pre-defined resnet18 model
        self.resnet18 = resnet18()
        
        # Modify the final layer if necessary (e.g., for a different number of classes)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
    
    def forward(self, x):
        # Define the forward pass using the resnet18 model
        x = self.resnet18(x)
        return x

       
    def batch_accuracy(self, outputs, labels):
        with torch.no_grad():
            _, predictions = torch.max(outputs, dim=1)
            return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))
    
    def _process_batch(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        accuracy = self.batch_accuracy(outputs, labels)
        return (loss, accuracy)
    
    def fit(self, dataset, epochs, lr, batch_size=32, opt=torch.optim.SGD):
        dataloader = DeviceDataLoader(torch.utils.data.DataLoader(dataset, batch_size, shuffle=True), device)
        optimizer = opt(self.parameters(), lr)
        history = []

        dataloader_iterator = iter(dataloader)
        for epoch in range(epochs):
            losses = []
            accs = []
            try:
                data, target = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                data, target = next(dataloader_iterator)
            # print(batch)
            loss, acc = self._process_batch([data, target])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss.detach()
            losses.append(loss)
            accs.append(acc)
            
            avg_loss = torch.stack(losses).mean().item()
            avg_acc = torch.stack(accs).mean().item()
            history.append((avg_loss, avg_acc))
        return history
        
    def evaluate(self, dataset, batch_size=32):
        dataloader = DeviceDataLoader(torch.utils.data.DataLoader(dataset, batch_size), device)
        losses = []
        accs = []
        with torch.no_grad():
            for batch in dataloader:
                loss, acc = self._process_batch(batch)
                losses.append(loss)
                accs.append(acc)
        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        return (avg_loss, avg_acc)



def get_parameters(model):
    stored_params = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            stored_params[name] = param.clone().detach()
    return stored_params

def change_parameters(model, stored_params):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in stored_params:
                param.copy_(stored_params[name])

def create_empty_parameter_dict(model):
    empty_params = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            empty_params[name] = torch.zeros_like(param)
    return empty_params
    
def create_random_parameter_dict(model):
    random_params = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            random_params[name] = torch.randn_like(param)
    return random_params

def reverse(params):
    reverse_params = {}   
    for name in params:
        reverse_params[name] = - params[name]
    return reverse_params
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
class Client:
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataset = dataset
    
    def get_dataset_size(self):
        return len(self.dataset)
    
    def get_client_id(self):
        return self.client_id
    
    def train(self, parameters_dict, lr):
        net = to_device(FedResNet18(), device)
        change_parameters(net, parameters_dict)
        train_history = net.fit(self.dataset, epochs_per_client, lr, batch_size)
        print('{}: Loss = {}, Accuracy = {}'.format(self.client_id, round(train_history[-1][0], 4), round(train_history[-1][1], 4)))
        return get_parameters(net)


f = 3
byz = [7,8,9]

examples_per_client = total_train_size // num_clients
client_datasets = torch.utils.data.random_split(trainset, [min(i + examples_per_client, 
           total_train_size) - i for i in range(0, total_train_size, examples_per_client)])

# ####### uncomment for LABEL-FLIP ATTACK ###############
# for j in range(num_clients):
#     if j in byz:
#         for idx in client_datasets[j].indices:  # Get the indices of the original dataset in the subset
#             trainset.targets[idx] = 9-trainset.targets[idx]
# ########################################################

clients = [Client('client_' + str(i), client_datasets[i]) for i in range(num_clients)]
global_net = to_device(FedResNet18(), device)
history = []
optimizer_process = torch.optim.SGD(global_net.parameters(), learning_rate)
test_loss, test_acc = global_net.evaluate(testset)
history.append(( test_loss, test_acc))

attack_round=0 #round number when the attack starts
c = [0,0,0,0,0,0,0,0,0,0] #for Sena_memory defense, initializing the identities as honest
for i in range(rounds):
    print('Start Round {} ... learning_rate {}'.format(i + 1, get_lr(optimizer_process)))
    curr_parameters = get_parameters(global_net)
    new_parameters = create_empty_parameter_dict(global_net)
    all_client_params = {}
    hon_client_params = {}
    byz_client_params = {}
    byz_params = {}
    if i<attack_round:
        for client in clients:
            client_parameters = client.train(curr_parameters, get_lr(optimizer_process))
            with torch.no_grad():
                all_client_params[client.client_id] = client_parameters
    else:
        for client in clients:
            if int(client.client_id.replace('client_','')) not in byz:
                client_parameters = client.train(curr_parameters, get_lr(optimizer_process))
                with torch.no_grad():
                    all_client_params[client.client_id] = client_parameters
                    hon_client_params[client.client_id] = client_parameters
            else:
                client_parameters = client.train(curr_parameters, get_lr(optimizer_process))
                with torch.no_grad():
                    byz_client_params[client.client_id] = client_parameters
                    all_client_params[client.client_id] = client_parameters
                    
            fraction = client.get_dataset_size() / total_train_size
            
        ############ FALL OF EMPIRES ATTACK ##############
        # byz_params = attacks.FOE(global_net, hon_client_params, 0.1) #omnicient, byzantines can obtain honest model parameters; attack_round can be changed above
        # for j in byz:
        #     all_client_params['client_'+str(j)] = byz_params
        ###################################################

        ############ A LITTLE IS ENOUGH ATTACK ##############
        # byz_params = attacks.ALIE(global_net, byz_client_params, 0.52) #not omnicient, byzantines can see their model parameters only
        # for j in byz:
        #     all_client_params['client_'+str(j)] = byz_params
        ###################################################
        
        ############ EASY MIMIC ATTACK ##############
        # byz_params = attacks.Mimic(global_net, hon_client_params, 0) #easy mimic attack without optimization
        # for j in byz:
        #     all_client_params['client_'+str(j)] = byz_params
        ###################################################

        ############ SIGN-FLIP ATTACK ##############
        # for j in byz:
        #     all_client_params['client_'+str(j)] = reverse(byz_client_params['client_'+str(j)])  #Sign-flip attack
        ###################################################
        
        ############ RANDOM GAUSSIAN ATTACK ##############
        # for j in byz:
        #     all_client_params['client_'+str(j)] = create_random_parameter_dict(global_net)  #random gaussian attack
        ###################################################

    
    
    ##### DEFENSES ######
    # new_parameters = defenses.KRUM(global_net, all_client_params, num_clients-f-2, 1) #change 1 to 2,3,4... for multikrum
    new_parameters = defenses.FedAvg(global_net, all_client_params, fraction) #change 1 to 2,3,4... for multikrum
    # new_parameters = defenses.median_trust(global_net, all_client_params)
    # new_parameters = defenses.Sena(global_net, all_client_params, f, 1) #change 1 to 2,3,4... for multi output
    # new_parameters = defenses.Sena_frac(global_net, all_client_params, f, 1) #change 1 to 2,3,4... for multi output
    # new_parameters = defenses.Sena_memory(global_net, all_client_params, f, 1, c) #change 1 to 2,3,4... for multi output
    #####################

    
    change_parameters(global_net, new_parameters)
    test_loss, test_acc = global_net.evaluate(testset)

    print('After round {}, test_loss = {}, test_acc = {} \n'.format(i + 1, round(test_loss, 4),round(test_acc, 4) ))
    history.append((test_loss, test_acc))

# np.savetxt("/simulation/history.txt", history)
