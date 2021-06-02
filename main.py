#%%
import sys
import os
import csv
import argparse
import multiprocessing
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from checkpoint import (
    default_checkpoint,
    load_checkpoint,
    save_checkpoint,
    init_tensorboard,
    write_tensorboard,
)
from model import Encoder, Decoder
from dataset import CrohmeDataset, START, PAD, collate_batch,load_vocab
from data_tools.inkmltopng import inkml2png
from data_tools.extract_groundtruth import create_tsv


#%%
#inkml2png([0,"bonus_inkml",128,0])
create_tsv('bonus_inkml','bonus_inkml.tsv')

#%%
from train import parse_args
gt_test = "bonus_inkml.tsv"
tokensfile = "./data/tokens.tsv"
root = "./data_test_png"
use_cuda = torch.cuda.is_available()

input_size = (128, 128)
low_res_shape = (684, input_size[0] // 16, input_size[1] // 16)
high_res_shape = (792, input_size[0] // 8, input_size[1] // 8)

batch_size = 1
num_epochs = 1
print_epochs = 1
learning_rate = 1e-3
lr_epochs = 20
lr_factor = 0.1
weight_decay = 1e-4
max_grad_norm = 5.0
dropout_rate = 0.2
teacher_forcing_ratio = 0.5
seed = 1234

torch.manual_seed(seed)
is_cuda = use_cuda
hardware = "cuda" if is_cuda else "cpu"
device = torch.device(hardware)
transformers = transforms.Compose(
    [
        # Resize so all images have the same size
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ]
)
test_dataset = CrohmeDataset(
    gt_test, tokensfile, root=root, crop=False, transform=transformers
)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch,
)


#%%

enc = Encoder(
    img_channels=3, 
    dropout_rate=dropout_rate, 
    checkpoint=load_checkpoint("checkpoints/0009.pth")["model"].get("encoder"),
).to(device)
dec = Decoder(
    len(test_dataset.id_to_token),
    low_res_shape,
    high_res_shape,
    checkpoint=load_checkpoint("checkpoints/0009.pth")["model"].get("decoder"),
    device=device,
).to(device)
enc.eval()
dec.eval()

#%%
criterion = nn.CrossEntropyLoss().to(device)
losses = []
grad_norms = []
correct_symbols = 0
total_symbols = 0
token_to_id, id_to_token = load_vocab(tokensfile)
for d in test_data_loader:
    input = d["image"].to(device)
    # The last batch may not be a full batch
    curr_batch_size = len(input)
    expected = d["truth"]["encoded"].to(device)
    batch_max_len = expected.size(1)
    # Replace -1 with the PAD token
    #expected[expected == -1] = test_data_loader.dataset.token_to_id[PAD]
    enc_low_res, enc_high_res = enc(input)
    # Decoder needs to be reset, because the coverage attention (alpha)
    # only applies to the current image.
    dec.reset(curr_batch_size)
    hidden = dec.init_hidden(curr_batch_size).to(device)
    # Starts with a START token
    sequence = torch.full(
        (curr_batch_size, 1),
        test_data_loader.dataset.token_to_id[START],
        dtype=torch.long,
        device=device,
    )
    # The teacher forcing is done per batch, not symbol
    use_teacher_forcing = False
    decoded_values = []
    for i in range(batch_max_len - 1):
        previous = expected[:, i] if use_teacher_forcing else sequence[:, -1]
        previous = previous.view(-1, 1)
        out, hidden = dec(previous, hidden, enc_low_res, enc_high_res)
        hidden = hidden.detach()
        _, top1_id = torch.topk(out, 1)
        sequence = torch.cat((sequence, top1_id), dim=1)
        decoded_values.append(out)

    decoded_values = torch.stack(decoded_values, dim=2).to(device)
    
    # decoded_values does not contain the start symbol
    #loss = criterion(decoded_values, expected[:, 1:])

    
    #losses.append(loss.item())
    #correct_symbols += torch.sum(sequence == expected, dim=(0, 1)).item()
    #total_symbols += expected.numel()
    

    result = {

    #"loss": np.mean(losses),
    #"correct_symbols": correct_symbols,
    #"total_symbols": total_symbols,
    "shape_out": decoded_values.size(),
    #"shape_in": expected,
    "sequence":sequence,
    }

    
    filepath = os.path.join("./lg2", d["path"][0].split('/')[-1][0:-4]) + '.lg'

    lg = csv.writer(open(filepath, 'w', newline=''), delimiter = ',')
    

    start = 0
    end = 0
    labels =  result["sequence"][0].tolist()[1:]
    lg.writerow(['#', str(labels)])
    for id in labels:
        if (id == 117) | (id == 118):
            break
        
        end += 1
        if (end == len(labels)):
            ids = range(start, end)
            s_ids = []
            for _ in ids:
                s_ids.append(_)

            lg.writerow(['O','X', id_to_token[id] , 1.0] + s_ids)
            start = end
            break
        
        if (labels[end] != labels[start]):    
            ids = range(start, end)
            s_ids = []
            for _ in ids:
                s_ids.append(_)
            

            lg.writerow(['O','X', id_to_token[id] , 1.0] + s_ids)
            start = end

# %%
c = range(1,2)
for i in c:
    print(i)