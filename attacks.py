#####Only train set and test set, no validation set######
#####with schedular reduce on plato ########
import torch
import copy
#import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision

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
    

            
def ALIE(net, byz_client_params, z_max): #not omniscient
    final_params = create_empty_parameter_dict(net)
    byz_client_params_concat = {}
    # n = len(all_params)
    for client_id in byz_client_params:
        params = byz_client_params[client_id]
        byz_client_params_concat[client_id] = copy.deepcopy(params)
        byz_client_params_concat[client_id] = np.concatenate([byz_client_params_concat[client_id][layer].ravel() for layer in byz_client_params_concat[client_id]])

    users_grads = []
    for client_id in byz_client_params:
        users_grads.append(byz_client_params_concat[client_id])
        # print(len(byz_client_params_concat[client_id]))
        
    grads_mean = np.mean(users_grads, axis=0)
    grads_stdev = np.var(users_grads, axis=0) ** 0.5
    final_grad = grads_mean + z_max*grads_stdev

    start_index = 0
    final_params_concat = copy.deepcopy(final_params)
    for layer in final_params:
        current_data = final_grad[start_index: start_index+len(final_params_concat[layer].ravel())]
        final_params[layer] = torch.from_numpy(current_data.reshape(final_params[layer].shape))
        start_index += len(final_params_concat[layer].ravel())
 
    return final_params
    
def FOE(net, hon_client_params, eps):
    final_params = create_empty_parameter_dict(net)
    hon_client_params_concat = {}
    # n = len(all_params)
    for client_id in hon_client_params:
        params = hon_client_params[client_id]
        hon_client_params_concat[client_id] = copy.deepcopy(params)
        hon_client_params_concat[client_id] = np.concatenate([hon_client_params_concat[client_id][layer].ravel() for layer in hon_client_params_concat[client_id]])

    users_grads = []
    for client_id in hon_client_params:
        users_grads.append(hon_client_params_concat[client_id])
        # print(len(byz_client_params_concat[client_id]))
        
    grads_mean = np.mean(users_grads, axis=0)
   
    final_grad = - eps*grads_mean

    start_index = 0
    final_params_concat = copy.deepcopy(final_params)
    for layer in final_params:
        current_data = final_grad[start_index: start_index+len(final_params_concat[layer].ravel())]
        final_params[layer] = torch.from_numpy(current_data.reshape(final_params[layer].shape))
        start_index += len(final_params_concat[layer].ravel())
 
    return final_params

def Mimic(net, hon_client_params, i):
    final_params = hon_client_params['client_' + str(i)]
    return final_params