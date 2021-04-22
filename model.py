import torch
from torch import nn
import random

class TimeSiries(nn.Module):
    def __init__(self, device, input_size, hidden_size, dropout):
        super(TimeSiries, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.rnn2 = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, future, targets, teacher_force_ratio):
        N = x.size(0)
        L = x.size(1) - 1
        h_t = torch.zeros(N, self.hidden_size).to(self.device)
        c_t = torch.zeros(N, self.hidden_size).to(self.device)
        h_t2 = torch.zeros(N, self.hidden_size).to(self.device)
        c_t2 = torch.zeros(N, self.hidden_size).to(self.device)
        outputs = []

        for i in x.split(1, dim=1):
            h_t, c_t = self.rnn(i, (h_t, c_t))
            h_t2, c_t2 = self.rnn2(h_t, (h_t2, c_t2))
            output = self.dropout(h_t2)
            output = self.fc(output)
            outputs.append(output)

        for i in range(future):
            if random.random() < teacher_force_ratio:
                output = targets[:, i].unsqueeze(1)
            h_t, c_t = self.rnn(output, (h_t, c_t))
            h_t2, c_t2 = self.rnn2(h_t, (h_t2, c_t2))
            output = self.dropout(h_t2)
            output = self.fc(output)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs


if __name__ == '__main__':
    future = 2
    data = torch.FloatTensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    targets = torch.FloatTensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    model = TimeSiries('cpu', 1, future, 0.0)
    print(model(data, 2, targets[:, -1-future:], 1))


