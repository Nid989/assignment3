import torch
import math
import random
from utils import char_tensor, CHUNK_LEN

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_bpc(model, string):
    criterion = torch.nn.CrossEntropyLoss()
    avg_bpc = 0
    bpc_losses = []
    for i in range(1, len(string) - CHUNK_LEN, CHUNK_LEN):
        hidden, cell = model.init_hidden()
        chunk = string[i: i+CHUNK_LEN+1]
        eval_input = char_tensor(chunk[:-1]).unsqueeze(0).to(device)
        eval_target = char_tensor(chunk[1:]).unsqueeze(0).to(device)

        loss = 0

        for c in range(CHUNK_LEN):
            with torch.no_grad():
                output, (hidden, cell) = model(eval_input[:, c], hidden, cell)
            loss += criterion(output, eval_target[:, c].view(1))

        loss = loss.item() / CHUNK_LEN
        
        # BPC Loss = CrossEntropyLoss / log(2)
        bpc = loss / math.log(2)
        avg_bpc += bpc
        bpc_losses.append(bpc)

    total_iteration = len(range(1, len(string) - CHUNK_LEN, CHUNK_LEN))
    return avg_bpc / total_iteration
