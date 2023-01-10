import argparse
import torch
import torch.nn as nn
import unidecode
import string
import time

from utils import char_tensor, random_training_set, time_since, CHUNK_LEN, get_optimizer, generate_hyperparameters
from language_model import plot_loss, diff_temp, custom_train, train, generate
from model.model import LSTM
from evaluation import compute_bpc

def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM'
    )

    parser.add_argument(
        '--default_train', dest='default_train',
        help='Train LSTM with default hyperparameter',
        action='store_true'
    )

    parser.add_argument(
        '--custom_train', dest='custom_train',
        help='Train LSTM while tuning hyperparameter',
        action='store_true'
    )

    parser.add_argument(
        '--plot_loss', dest='plot_loss',
        help='Plot losses chart with different learning rates',
        action='store_true'
    )

    parser.add_argument(
        '--diff_temp', dest='diff_temp',
        help='Generate strings by using different temperature',
        action='store_true'
    )

    args = parser.parse_args()

    all_characters = string.printable
    n_characters = len(all_characters)

    if args.default_train:
        n_epochs = 3000
        print_every = 100
        plot_every = 10
        hidden_size = 128
        n_layers = 2
        lr = 0.005

        decoder = LSTM(
            input_size=n_characters, 
            hidden_size=hidden_size,
            num_layers=n_layers, 
            output_size=n_characters)

        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

        start = time.time()
        all_losses = []
        loss_avg = 0

        print(" -------------------------- STARTING TRAINING -------------------------- ")

        for epoch in range(1, n_epochs+1):
            loss = train(decoder, decoder_optimizer, *random_training_set())
            loss_avg += loss

            if epoch % print_every == 0:
                print('[{} ({} {}%) {:.4f}]'.format(time_since(start), epoch, epoch/n_epochs * 100, loss))
                print(generate(decoder, 'A', 100), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0 

    if args.custom_train:
        
        # change these values as per your choice and tune only 1 param at time
        hyperparam_lists = [
            [{"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},], # tune epochs
            [{"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},], # tune hidden_size
            [{"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},], # tune n_layers
            [{"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},] # tune lr
        ]
        
        for hyperparam_list in hyperparam_lists:
            bpc = custom_train(hyperparam_list)
            for setting, (key, value) in zip(hyperparam_list, bpc.items()):
                print(f"{key} configuration\n{setting}")
                print("BPC {}: {}".format(key, value))

    if args.plot_loss:
        lr_list = [{"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},
            {"n_epochs": 3000, "hidden_size": 128, "n_layers": 2, "lr": 0.005},] # tune lr
        plot_loss(lr_list)

    if args.diff_temp:
        temp_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        diff_temp(temp_list)

if __name__ == "__main__":
    main()
