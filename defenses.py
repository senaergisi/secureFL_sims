#####Only train set and test set, no validation set######
#####with schedular reduce on plato ########
# from random import randrange
import torch
import copy
import numpy as np
import torchvision


    
def create_random_parameter_dict(model):
    random_params = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            random_params[name] = torch.randn_like(param)
    return random_params

def create_empty_parameter_dict(model):
    empty_params = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            empty_params[name] = torch.zeros_like(param)
    return empty_params

def cos_sim(A,B):
    cevap = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
    if cevap > 1:
        cevap = 1
    elif cevap  < -1:
        cevap =-1
    return cevap

def R_same(A,B):
    return ( cos_sim(A,B)+ 1)/2

def R_diff(A,B):
    return (1- cos_sim(A,B) )/2



def FedAvg(net, all_client_params, fraction):
    final_params = create_empty_parameter_dict(net)
    for client_id in all_client_params:
        with torch.no_grad():
            for name in all_client_params[client_id]:
                final_params[name] += fraction * all_client_params[client_id][name]
    return final_params


def Sena(net, all_params, m, k=1):
        #k=1 KRUM, otherwise multikrum
    # m is the number of byzantine workers
    final_params = create_empty_parameter_dict(net)
    all_params_concat = {}
    n = len(all_params)
    for client_id in all_params:
        params = all_params[client_id]
        all_params_concat[client_id] = copy.deepcopy(params)
        all_params_concat[client_id] = np.concatenate([all_params_concat[client_id][layer].ravel() for layer in all_params_concat[client_id]])
    dist = {}
    scores = {}
    log_app = 0
    # prodprod_byz = 0
    i = 0
    for main_client in all_params:
        # for other clienmain_node
        dist[main_client] = {neighbor_client: cos_sim(all_params_concat[main_client],all_params_concat[neighbor_client]) for neighbor_client in all_params}
        dist[main_client].pop(main_client) 
        dist[main_client]=dict(sorted(dist[main_client].items(),key=lambda item: item[1], reverse=True))
        # prodprod_true = 1
        # prodprod_byz = 1
        # i = 0
        log_app = 0
        for neigh in dist[main_client]:
            log_app += np.log((1+cos_sim(all_params_concat[main_client],all_params_concat[neigh]))/2)
        scores[main_client] = log_app
    
    scores=dict(sorted(scores.items(),key=lambda item: item[1], reverse=True))
    print(scores)

    i = 0
    for main_client in scores:
        if i<k:
            for layer in all_params[main_client]:
                final_params[layer] += 1/k * all_params[main_client][layer]
        i +=1
    # print(scores)
    # print(dist)
    return final_params
    
def median_trust(net, all_params):
        #k=1 KRUM, otherwise multikrum
    # m is the number of byzantine workers
    final_params = create_empty_parameter_dict(net)
    all_params_concat = {}
    all_norms = {}
    n = len(all_params)
    for client_id in all_params:
        params = all_params[client_id]
        all_params_concat[client_id] = copy.deepcopy(params)
        all_params_concat[client_id] = np.concatenate([all_params_concat[client_id][layer].ravel() for layer in all_params_concat[client_id]])
        all_norms[client_id] = np.linalg.norm(all_params_concat[client_id])
        
    users_grads = []
    for client_id in all_params_concat:
        users_grads.append(all_params_concat[client_id])
        # print(len(byz_client_params_concat[client_id]))
        
    grads_median = np.median(users_grads, axis=0)
    norm_median = np.linalg.norm(grads_median)
    
    dist = {}
    sumsum = 0
    for main_client in all_params:
        # for other clienmain_node
        dist[main_client] = cos_sim(all_params_concat[main_client], grads_median)
        if dist[main_client] <0:
            dist[main_client] = 0
        sumsum += dist[main_client]
        
    for main_client in all_params:  
        with torch.no_grad():
            for name in all_params[main_client]:
                final_params[name] += dist[main_client]/sumsum * all_params[main_client][name] * norm_median/all_norms[main_client]
    return final_params
    
def KRUM(net, all_params, m, k=1):
    #k=1 KRUM, otherwise multikrum
    final_params = create_empty_parameter_dict(net)
    all_params_concat = {}
    n= len(all_params)
    for client_id in all_params:
        # order = int(client_id.replace('client_',''))
        params = all_params[client_id]
        all_params_concat[client_id] = copy.deepcopy(params)
        all_params_concat[client_id] = np.concatenate([all_params_concat[client_id][layer].ravel() for layer in all_params_concat[client_id]])
        # print(all_params_concat[client_id].dtype)
    dist = {}
    scores = {}
    sumsum = 0
    i = 0
    for main_client in all_params:
        # for other clienmain_node
        dist[main_client] = {neighbor_client: np.linalg.norm(all_params_concat[main_client]-all_params_concat[neighbor_client]) for neighbor_client in all_params}
        dist[main_client].pop(main_client) 
        dist[main_client]=dict(sorted(dist[main_client].items(),key=lambda item: item[1]))
        # scores[main_client] = sum()
        sumsum = 0;
        i = 0;
        for neigh in dist[main_client]:
            if i<m:
                sumsum += dist[main_client][neigh]
            i += 1
        scores[main_client] = sumsum
        scores=dict(sorted(scores.items(),key=lambda item: item[1]))

    i = 0
    print(scores)
    for main_client in scores:
        if i<k:
            for layer in all_params[main_client]:
                final_params[layer] += 1/k * all_params[main_client][layer]
        i +=1
    # print(scores)
    return final_params

def Sena_frac(net, all_params, m, k=1):
        #k=1 KRUM, otherwise multikrum
    # m is the number of byzantine workers
    final_params = create_empty_parameter_dict(net)
    all_params_concat = {}
    n = len(all_params)
    for client_id in all_params:
        params = all_params[client_id]
        all_params_concat[client_id] = copy.deepcopy(params)
        all_params_concat[client_id] = np.concatenate([all_params_concat[client_id][layer].ravel() for layer in all_params_concat[client_id]])
    dist = {}
    scores = {}
    log_app = 0
    # prodprod_byz = 0
    i = 0
    for main_client in all_params:
        # for other clienmain_node
        dist[main_client] = {neighbor_client: cos_sim(all_params_concat[main_client],all_params_concat[neighbor_client]) for neighbor_client in all_params}
        dist[main_client].pop(main_client) 
        dist[main_client]=dict(sorted(dist[main_client].items(),key=lambda item: item[1], reverse=True))
        # prodprod_true = 1
        # prodprod_byz = 1
        # i = 0
        log_app = 0
        for neigh in dist[main_client]:
            cs = 1-cos_sim(all_params_concat[main_client],all_params_concat[neigh])
            log_app += np.log((1+cs)/(1-cs))
        scores[main_client] = log_app
    
    scores=dict(sorted(scores.items(),key=lambda item: item[1], reverse=True))
    print(scores)

    i = 0
    for main_client in scores:
        if i<k:
            for layer in all_params[main_client]:
                final_params[layer] += 1/k * all_params[main_client][layer]
        i +=1
    # print(scores)
    # print(dist)
    return final_params


def Sena_memory(net, all_params, m, k, c):
    # k=1 KRUM, otherwise multikrum
    eps = 10^(-10)
    final_params = create_empty_parameter_dict(net)
    all_params_concat = {}
    n = len(all_params)
    for client_id in all_params:
        params = all_params[client_id]
        all_params_concat[client_id] = copy.deepcopy(params)
        all_params_concat[client_id] = np.concatenate([all_params_concat[client_id][layer].ravel() for layer in all_params_concat[client_id]])
    dist = {}
    scores = {}
    log_hon = 0
    log_byz = 0
    i = 0
    for main_client in all_params:
        # for other clienmain_node
        dist[main_client] = {neighbor_client: cos_sim(all_params_concat[main_client],all_params_concat[neighbor_client]) for neighbor_client in all_params}
        # dist[main_client].pop(main_client) 
        # dist[main_client]=dict(sorted(dist[main_client].items(),key=lambda item: item[1], reverse=True))
        # print(dist)
        log_hon = 0
        log_byz = 0
        i = 0
        for neigh in dist[main_client]:
            if main_client != neigh:
                cs = cos_sim(all_params_concat[main_client],all_params_concat[neigh])
                # print(cs)
                if c[i] == 0:
                    log_hon += np.log(1+cs)
                    log_byz += np.log(1-cs)
                else:
                    log_hon += np.log(1-cs)
                    log_byz += np.log(1+cs)
            # else:
                # print(main_client, neigh)
            i += 1
                
        scores[main_client] = log_hon-log_byz
    
    scores=dict(sorted(scores.items(),key=lambda item: item[1], reverse=True))
    print(scores)
    i = 0


    for main_client in scores:
        if i<k:
            for layer in all_params[main_client]:
                final_params[layer] += 1/k * all_params[main_client][layer]
        i+=1

    i = 0
    # c_new = [0,0,0,0,0,0,0,0,0,0]
    # print(n-m)
    for main_client in scores:
        index = int(main_client.replace('client_',''))
        # print(index)
        # print(c_new[index])
        if i<n-m:
            c[index] = 0
        else:
            c[index] = 1
        i +=1
    # print(c_new)
    # print(scores)
    # print(dist)
    return final_params



