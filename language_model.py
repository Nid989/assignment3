import torch
import torch.nn as nn
import string
import numpy as np
import time
import unidecode
import random
import matplotlib.pyplot as plt

from utils import char_tensor, random_training_set, time_since, random_chunk, CHUNK_LEN, get_optimizer
from evaluation import compute_bpc
from model.model import LSTM

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden, cell = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str
    all_characters = string.printable

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, (hidden, cell) = decoder(prime_input[p].view(1).to(device), hidden, cell) 
    inp = prime_input[-1]

    for p in range(predict_len):
        output, (hidden, cell) = decoder(inp.view(1).to(device), (hidden, cell))

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted


def train(decoder, decoder_optimizer, inp, target):

    # push PyTorch model and other associated params. and attr. to same device "cuda" OR "cpu" 
    decoder = decoder.to(device)
    inp = inp.to(device)
    target = target.to(device)

    hidden, cell = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    criterion = nn.CrossEntropyLoss()

    for c in range(CHUNK_LEN):
        output, (hidden, cell) = decoder(inp[:, c], hidden, cell)
        loss += criterion(output, target[:, c].view(1))

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / CHUNK_LEN


def tuner_train(decoder, decoder_optimizer, inp, target, criterion):

    # push PyTorch model and other associated params. and attr. to same device "cuda" OR "cpu" 
    decoder = decoder.to(device)
    inp = inp.to(device)
    target = target.to(device)

    hidden, cell = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(CHUNK_LEN):
        output, (hidden, cell) = decoder(inp[:, c], hidden, cell)
        loss += criterion(output, target[:, c].view(1))

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / CHUNK_LEN


def tuner(n_epochs=3000, print_every=100, plot_every=10, hidden_size=128, n_layers=2,
          lr=0.005, start_string='A', prediction_length=100, temperature=0.8, optimizer="Adam"):
        
        all_characters = string.printable
        n_characters = len(all_characters)
        decoder = LSTM(
            input_size=n_characters,
            hidden_size=hidden_size,
            num_layers=n_layers,
            output_size=n_characters
        )
        decoder_optimizer = get_optimizer(decoder=decoder, optim=optimizer, lr=lr)
        criterion = nn.CrossEntropyLoss()

        start = time.time()
        all_losses = []
        loss_avg = 0

        print(" -------------------------- STARTING TRAINING -------------------------- ")

        for epoch in range(1, n_epochs+1):
            loss = tuner_train(decoder, decoder_optimizer, *random_training_set(), criterion)
            loss_avg += loss

            if epoch % print_every == 0:
                print('[{} ({} {:.2f}%) {:.4f}]'.format(time_since(start), epoch, epoch/n_epochs * 100, loss))
                # print(generate(decoder, start_string, prediction_length), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0

        return decoder, all_losses


def plot_loss(lr_list):

    fig, ax = plt.subplots()
    for index, setting in enumerate(lr_list):
        model, all_losses = tuner(
            n_epochs=setting["num_epochs"],
            hidden_size=setting["hidden_size"],
            n_layers=setting["num_layers"],
            lr=setting["lr"]
        )
        x = np.arange(0, len(all_losses))
        ax.plot(x, np.array(all_losses), label=f"{setting['lr']}")
    plt.legend(loc="upper right")
    plt.xlabel(xlabel="Iterations")
    plt.ylabel(ylabel="CrossEntropyLoss")
    # plt.show() 
    fig.savefig("plot_loss.png")


def diff_temp(temp_list):

    model, _ = tuner()
    TEST_PATH = './data/dickens_test.txt'
    string = unidecode.unidecode(open(TEST_PATH, 'r').read())
    predict_len = 200
    for temperature in temp_list:
        # random_chunk
        start_index = random.randint(0, len(string) - CHUNK_LEN - 1)
        end_index = start_index + CHUNK_LEN + 1
        chunk = string[start_index: end_index]
        predicted_text = generate(model, prime_str=chunk[:10], predict_len=predict_len, temperature=temperature)    
        print(f"temperature: {temperature}")
        print(f"prediction: {predicte
d_text}")
        print(f"original: {chunk}")
        print("---------------------------------------------------------------")

def custom_train(hyperparam_list):
    TEST_PATH = './data/dickens_test.txt'
    string = unidecode.unidecode(open(TEST_PATH, 'r').read())
    bpc_scores = {}
    for index, setting in enumerate(hyperparam_list):
        model, _ = tuner(
            n_epochs=setting["num_epochs"],
            hidden_size=setting["hidden_size"],
            n_layers=setting["num_layers"],
            lr=setting["lr"],
        )

        bpc_score = compute_bpc(model, string)
        # print(f"Average BPC score: {bpc_score}")
        
        del model
        bpc_scores[f"attempt_{index}"] = bpc_score 
    
    return bpc_scores
