import torch
import math
import random
from utils import char_tensor, CHUNK_LEN

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_bpc(model, string):
    criterion = torch.nn.CrossEntropyLoss()
    hidden, cell = model.init_hidden()
    start_idx = random.randint(0, len(string) - CHUNK_LEN - 1)
    end_idx = start_idx + CHUNK_LEN + 1
    chunk = string[start_idx:end_idx]
    eval_input = char_tensor(chunk[:-1]).unsqueeze(0).to(device)
    eval_target = char_tensor(chunk[1:]).unsqueeze(0).to(device)

    loss = []
    bpc = []

    for c in range(CHUNK_LEN):
        with torch.no_grad():
            output, (hidden, cell) = model(eval_input[:, c], hidden, cell)
        loss.append(criterion(output, eval_target[:, c].view(1)))
        bpc.append(loss[-1].item() / math.log(2))

    out_loss = sum([l.item() for l in loss]) / CHUNK_LEN # average LOSS
    out_bpc = sum([bl for bl in bpc]) / CHUNK_LEN # average BPC
    
    return out_bpc
    

# def compute_bpc(model, string):
#     criterion = torch.nn.CrossEntropyLoss()
#     avg_bpc = 0
#     bpc_losses = []
#     for i in range(1, len(string) - CHUNK_LEN, CHUNK_LEN):
#         hidden, cell = model.init_hidden()
#         chunk = string[i: i+CHUNK_LEN+1]
#         eval_input = char_tensor(chunk[:-1]).unsqueeze(0).to(device)
#         eval_target = char_tensor(chunk[1:]).unsqueeze(0).to(device)

#         loss = 0

#         for c in range(CHUNK_LEN):
#             with torch.no_grad():
#                 output, (hidden, cell) = model(eval_input[:, c], hidden, cell)
#             loss += criterion(output, eval_target[:, c].view(1))

#         loss = loss.item() / CHUNK_LEN
        
#         # BPC Loss = CrossEntropyLoss / log(2)
#         bpc = loss / math.log(2)
#         avg_bpc += bpc
#         bpc_losses.append(bpc)

#     total_iteration = len(range(1, len(string) - CHUNK_LEN, CHUNK_LEN))
#     return avg_bpc / total_iteration