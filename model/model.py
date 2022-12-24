import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, hidden, cell):
        output = self.embedding(x)
        output, (hidden, cell) = self.lstm(output.unsqueeze(1), (hidden, cell))
        output = self.fc(output.reshape(output.shape[0], -1))
        return output, (hidden, cell)


    def init_hidden(self, batch_size=1):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return hidden, cell